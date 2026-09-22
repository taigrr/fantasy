package kev

import (
	"fmt"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// engine wraps a loaded GGUF model and one inference context.
type engine struct {
	model llama.Model
	ctx   llama.Context
	vocab llama.Vocab
	nEmbd int

	mu sync.Mutex
}

// EngineOptions tune the llama.cpp context.
type EngineOptions struct {
	// Threads for decoding; zero uses llama.cpp's default.
	Threads int
	// GPULayers to offload. Zero keeps llama.cpp's default (all layers on
	// GPU when a GPU backend is present); -1 forces CPU-only.
	GPULayers int
	// ContextSize caps a single row. Zero uses MaxStateTokens+MaxBranchTokens.
	ContextSize int
}

func newEngine(modelPath string, opts EngineOptions) (*engine, error) {
	modelParams := llama.ModelDefaultParams()
	switch {
	case opts.GPULayers < 0:
		modelParams.NGpuLayers = 0
	case opts.GPULayers > 0:
		modelParams.NGpuLayers = int32(opts.GPULayers) //nolint:gosec // small user-provided count
	}
	model, err := llama.ModelLoadFromFile(modelPath, modelParams)
	if err != nil {
		return nil, fmt.Errorf("kev: load model %q: %w", modelPath, err)
	}

	ctxSize := opts.ContextSize
	if ctxSize <= 0 {
		ctxSize = MaxRowTokens
	}
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(ctxSize)
	ctxParams.NBatch = uint32(ctxSize)
	ctxParams.NUbatch = uint32(ctxSize)
	ctxParams.NSeqMax = 2 // seq 0 holds the state prefix, seq 1 the active branch
	ctxParams.Embeddings = 1
	ctxParams.PoolingType = llama.PoolingTypeNone
	if opts.Threads > 0 {
		ctxParams.NThreads = int32(opts.Threads)      //nolint:gosec // small user-provided count
		ctxParams.NThreadsBatch = int32(opts.Threads) //nolint:gosec // small user-provided count
	}
	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("kev: create context: %w", err)
	}
	return &engine{
		model: model,
		ctx:   ctx,
		vocab: llama.ModelGetVocab(model),
		nEmbd: int(llama.ModelNEmbd(model)),
	}, nil
}

func (e *engine) close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.ctx != 0 {
		_ = llama.Free(e.ctx)
		e.ctx = 0
	}
	if e.model != 0 {
		_ = llama.ModelFree(e.model)
		e.model = 0
	}
}

// Tokenize implements Tokenizer: no BOS, control tokens parsed.
func (e *engine) Tokenize(text string) []int32 {
	toks := llama.Tokenize(e.vocab, text, false, true)
	out := make([]int32, len(toks))
	for i, t := range toks {
		out[i] = int32(t)
	}
	return out
}

// SpecialToken implements Tokenizer.
func (e *engine) SpecialToken(text string) (int32, bool) {
	toks := llama.Tokenize(e.vocab, text, false, true)
	if len(toks) != 1 {
		return 0, false
	}
	return int32(toks[0]), true
}

// Detokenize renders tokens back to text with control tokens visible.
func (e *engine) Detokenize(tokens []int32) string {
	toks := make([]llama.Token, len(tokens))
	for i, t := range tokens {
		toks[i] = llama.Token(t)
	}
	return llama.Detokenize(e.vocab, toks, false, true)
}

// hiddenStates runs one causal row and returns the final-layer hidden states
// (post final norm) at the requested token indices, in order.
func (e *engine) hiddenStates(row Row, indices []int) ([][]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	mem, err := llama.GetMemory(e.ctx)
	if err != nil {
		return nil, fmt.Errorf("kev: get memory: %w", err)
	}
	if err := llama.MemoryClear(mem, true); err != nil {
		return nil, fmt.Errorf("kev: clear memory: %w", err)
	}

	n := len(row.Tokens)
	if n > MaxRowTokens {
		return nil, fmt.Errorf("kev: row of %d tokens exceeds %d", n, MaxRowTokens)
	}
	batch := llama.BatchInit(int32(n), 0, 1)
	defer llama.BatchFree(batch) //nolint:errcheck

	want := make(map[int]bool, len(indices))
	for _, idx := range indices {
		if idx < 0 || idx >= n {
			return nil, fmt.Errorf("kev: readout index %d out of range %d", idx, n)
		}
		want[idx] = true
	}
	for i, tok := range row.Tokens {
		if err := batch.Add(llama.Token(tok), llama.Pos(i), []llama.SeqId{seqID}, want[i]); err != nil {
			return nil, fmt.Errorf("kev: build batch: %w", err)
		}
	}
	if rc, err := llama.Decode(e.ctx, batch); err != nil || rc != 0 {
		if err == nil {
			err = fmt.Errorf("status %d", rc)
		}
		return nil, fmt.Errorf("kev: decode: %w", err)
	}

	// llama_get_embeddings_ith takes the batch position and resolves the
	// output slot internally (negative for "last"), so index by token.
	out := make([][]float32, len(indices))
	for i, idx := range indices {
		emb, err := llama.GetEmbeddingsIth(e.ctx, int32(idx), int32(e.nEmbd)) //nolint:gosec // idx < n <= MaxRowTokens; nEmbd is a model dim
		if err != nil {
			return nil, fmt.Errorf("kev: embeddings for token %d: %w", idx, err)
		}
		if emb == nil {
			return nil, fmt.Errorf("kev: no embeddings for token %d; was it flagged for output?", idx)
		}
		out[i] = append([]float32(nil), emb...)
	}
	return out, nil
}

const (
	stateSeq  = llama.SeqId(0)
	branchSeq = llama.SeqId(1)
)

// prefixSession decodes a shared state prefix once and then evaluates any
// number of branches against it, each as a fork of the cached prefix. This
// is the serving path Kev's Python server uses for hybrid backbones: exact
// by construction, since a branch only ever attends to the state and
// itself, and the state never sees a branch.
type prefixSession struct {
	engine *engine
	state  []int32
}

// beginPrefix clears memory and decodes state into stateSeq. No outputs are
// requested for the prefix.
func (e *engine) beginPrefix(state []int32) (*prefixSession, error) {
	e.mu.Lock()
	mem, err := llama.GetMemory(e.ctx)
	if err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("kev: get memory: %w", err)
	}
	if err := llama.MemoryClear(mem, true); err != nil {
		e.mu.Unlock()
		return nil, fmt.Errorf("kev: clear memory: %w", err)
	}
	if len(state) > MaxRowTokens {
		e.mu.Unlock()
		return nil, fmt.Errorf("kev: state of %d tokens exceeds %d", len(state), MaxRowTokens)
	}
	batch := llama.BatchInit(int32(len(state)), 0, 1) //nolint:gosec // bounded by MaxRowTokens above
	defer llama.BatchFree(batch)                      //nolint:errcheck
	for i, tok := range state {
		if err := batch.Add(llama.Token(tok), llama.Pos(i), []llama.SeqId{stateSeq}, false); err != nil {
			e.mu.Unlock()
			return nil, fmt.Errorf("kev: build state batch: %w", err)
		}
	}
	if rc, err := llama.Decode(e.ctx, batch); err != nil || rc != 0 {
		e.mu.Unlock()
		if err == nil {
			err = fmt.Errorf("status %d", rc)
		}
		return nil, fmt.Errorf("kev: decode state: %w", err)
	}
	return &prefixSession{engine: e, state: state}, nil
}

// end releases the engine lock taken by beginPrefix.
func (s *prefixSession) end() {
	s.engine.mu.Unlock()
}

// branch forks the cached state into branchSeq, decodes br after it and
// returns hidden states for the <decide> token first, then each </opt>.
func (s *prefixSession) branch(br branch) ([][]float32, error) {
	e := s.engine
	n := len(s.state) + len(br.tokens)
	if n > MaxRowTokens {
		return nil, fmt.Errorf("%w (%d tokens)", ErrBranchTooLong, n)
	}
	mem, err := llama.GetMemory(e.ctx)
	if err != nil {
		return nil, fmt.Errorf("kev: get memory: %w", err)
	}
	if _, err := llama.MemorySeqRm(mem, branchSeq, -1, -1); err != nil {
		return nil, fmt.Errorf("kev: reset branch sequence: %w", err)
	}
	if err := llama.MemorySeqCp(mem, stateSeq, branchSeq, -1, -1); err != nil {
		return nil, fmt.Errorf("kev: fork state prefix: %w", err)
	}

	want := make(map[int]bool, len(br.optEnds)+1)
	want[len(br.tokens)-1] = true
	for _, end := range br.optEnds {
		want[end] = true
	}
	batch := llama.BatchInit(int32(len(br.tokens)), 0, 1) //nolint:gosec // bounded by MaxRowTokens above
	defer llama.BatchFree(batch)                          //nolint:errcheck
	base := len(s.state)
	for i, tok := range br.tokens {
		if err := batch.Add(llama.Token(tok), llama.Pos(base+i), []llama.SeqId{branchSeq}, want[i]); err != nil { //nolint:gosec // bounded by MaxRowTokens above
			return nil, fmt.Errorf("kev: build branch batch: %w", err)
		}
	}
	if rc, err := llama.Decode(e.ctx, batch); err != nil || rc != 0 {
		if err == nil {
			err = fmt.Errorf("status %d", rc)
		}
		return nil, fmt.Errorf("kev: decode branch: %w", err)
	}

	indices := append([]int{len(br.tokens) - 1}, br.optEnds...)
	out := make([][]float32, len(indices))
	for i, idx := range indices {
		emb, err := llama.GetEmbeddingsIth(e.ctx, int32(idx), int32(e.nEmbd)) //nolint:gosec // idx < MaxRowTokens; nEmbd is a model dim
		if err != nil {
			return nil, fmt.Errorf("kev: embeddings for branch token %d: %w", idx, err)
		}
		if emb == nil {
			return nil, fmt.Errorf("kev: no embeddings for branch token %d", idx)
		}
		out[i] = append([]float32(nil), emb...)
	}
	return out, nil
}
