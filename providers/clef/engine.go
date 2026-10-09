package clef

import (
	"errors"
	"fmt"
	"math"
	"sync"

	"github.com/hybridgroup/yzma/pkg/llama"
)

// ErrNotDecisionModel is returned when the GGUF is not a Clef decision model
// (its output is not one score per token).
var ErrNotDecisionModel = errors.New("clef: GGUF is not a clef decision model")

// ErrInvalidScores is returned when llama.cpp rejects the span layout of a
// prompt, which it reports as NaN scores.
var ErrInvalidScores = errors.New("clef: llama.cpp rejected the decision layout")

// engine wraps a loaded GGUF model and one inference context. Clef has no
// KV cache: every call is one full prefill and the head scores every option
// in the same batch.
type engine struct {
	model     llama.Model
	ctx       llama.Context
	vocab     llama.Vocab
	maxTokens int

	mu sync.Mutex
}

// EngineOptions tune the llama.cpp context.
type EngineOptions struct {
	// Threads for the prefill; zero uses llama.cpp's default.
	Threads int
	// GPULayers to offload. Zero keeps llama.cpp's default (all layers on
	// GPU when a GPU backend is present); -1 forces CPU-only.
	GPULayers int
	// ContextSize caps one prompt (state, questions and template). Zero
	// uses DefaultMaxTokens. Clef attends over the whole prompt at once, so
	// memory grows with the square of this value.
	ContextSize int
}

// DefaultMaxTokens is the prompt budget when EngineOptions.ContextSize is
// zero. Clef's own reference uses 16384; half of that keeps the attention
// buffers modest on laptops while covering most states.
const DefaultMaxTokens = 8192

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
		return nil, fmt.Errorf("clef: load model %q: %w", modelPath, err)
	}
	if llama.ModelNEmbdOut(model) != 1 {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("%w: %q", ErrNotDecisionModel, modelPath)
	}

	maxTokens := opts.ContextSize
	if maxTokens <= 0 {
		maxTokens = DefaultMaxTokens
	}
	ctxParams := llama.ContextDefaultParams()
	ctxParams.NCtx = uint32(maxTokens)        //nolint:gosec // bounded user setting
	ctxParams.NBatch = uint32(maxTokens)      //nolint:gosec // the whole prompt must be one batch
	ctxParams.NUbatch = uint32(maxTokens)     //nolint:gosec // the head reads the whole ubatch
	ctxParams.NOutputsMax = uint32(maxTokens) //nolint:gosec // every token is an output
	ctxParams.NSeqMax = 1
	ctxParams.Embeddings = 1
	ctxParams.PoolingType = llama.PoolingTypeNone
	ctxParams.NoPerf = 1
	if opts.Threads > 0 {
		ctxParams.NThreads = int32(opts.Threads)      //nolint:gosec // small user-provided count
		ctxParams.NThreadsBatch = int32(opts.Threads) //nolint:gosec // small user-provided count
	}
	ctx, err := llama.InitFromModel(model, ctxParams)
	if err != nil {
		_ = llama.ModelFree(model)
		return nil, fmt.Errorf("clef: create context: %w", err)
	}
	return &engine{
		model:     model,
		ctx:       ctx,
		vocab:     llama.ModelGetVocab(model),
		maxTokens: maxTokens,
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

// scores runs one prompt and returns the head's score for every option, in
// prompt order.
func (e *engine) scores(p prompt, nOptions int) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	n := len(p.tokens)
	if n > e.maxTokens {
		return nil, fmt.Errorf("clef: prompt of %d tokens exceeds %d", n, e.maxTokens)
	}
	if mem, err := llama.GetMemory(e.ctx); err == nil && mem != 0 {
		if err := llama.MemoryClear(mem, true); err != nil {
			return nil, fmt.Errorf("clef: clear memory: %w", err)
		}
	}

	batch, err := llama.BatchExtInit(e.ctx)
	if err != nil {
		return nil, fmt.Errorf("clef: create batch: %w", err)
	}
	defer llama.BatchExtFree(batch) //nolint:errcheck
	for i, tok := range p.tokens {
		idx, err := llama.BatchExtAddToken(batch, 0, llama.Token(tok))
		if err != nil {
			return nil, fmt.Errorf("clef: add token %d: %w", i, err)
		}
		if err := llama.BatchExtSetPos(batch, idx, llama.Pos(i)); err != nil { //nolint:gosec // i < maxTokens
			return nil, fmt.Errorf("clef: set position %d: %w", i, err)
		}
		if err := llama.BatchExtSetOutputEmbd(batch, idx, true); err != nil {
			return nil, fmt.Errorf("clef: request output %d: %w", i, err)
		}
		if err := llama.BatchExtSetDecisionOrder(batch, idx, p.orders[i]); err != nil {
			return nil, fmt.Errorf("clef: set decision order %d: %w", i, err)
		}
	}
	if rc, err := llama.Process(e.ctx, llama.ProcessTypeDecode, batch); err != nil || rc != 0 {
		if err == nil {
			err = fmt.Errorf("status %d", rc)
		}
		return nil, fmt.Errorf("clef: decode: %w", err)
	}

	// The scores are the first nOptions output rows, one float each.
	out := make([]float32, nOptions)
	for i := range out {
		row, err := llama.GetEmbeddingsIth(e.ctx, int32(i), 1) //nolint:gosec // i < nOptions <= maxTokens
		if err != nil {
			return nil, fmt.Errorf("clef: score %d: %w", i, err)
		}
		if len(row) == 0 {
			return nil, fmt.Errorf("clef: no score for option %d", i)
		}
		if math.IsNaN(float64(row[0])) {
			return nil, ErrInvalidScores
		}
		out[i] = row[0]
	}
	return out, nil
}
