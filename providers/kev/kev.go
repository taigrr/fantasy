// Package kev runs Kev decision models fully in-process using llama.cpp via
// yzma. Kev (github.com/jaredpalmer/kev) is an open-weight reimplementation
// of TypeSafe's Jev: a LoRA-tuned Qwen3.5 backbone plus a small pointer head
// that scores each option's hidden state against a <decide> token.
//
// Checkpoints are downloaded on first use from Hugging Face as GGUF bundles
// (LoRA already merged) so binaries never embed weights; every file is
// SHA-256 checked against a manifest. The llama.cpp shared libraries are
// pinned to LlamaCPPVersion and, with WithAutoLibraries, installed the same
// way: manifest digest compiled in, every archive and file verified before
// load, and stale installs replaced in place.
//
//	p, err := kev.New(kev.WithCheckpoint(kev.Checkpoint4B), kev.WithAutoLibraries())
//	model, err := p.EvaluationModel(ctx, "")
//	resp, err := model.Evaluate(ctx, fantasy.EvaluationCall{...})
package kev

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/typesafe"
)

const (
	// Name is the provider name.
	Name = "kev"
	// ModelLatest is the alias accepted for the configured checkpoint.
	ModelLatest = typesafe.ModelKevLatest
)

// ErrLanguageModelUnsupported is returned by LanguageModel.
var ErrLanguageModelUnsupported = errors.New("kev: language models are not supported; use EvaluationModel")

var (
	_ fantasy.EvaluationProvider = (*provider)(nil)
	_ fantasy.EvaluationModel    = (*evaluationModel)(nil)
	_ fantasy.EvaluationModel    = (*lazyModel)(nil)
)

type options struct {
	name        string
	checkpoint  Checkpoint
	downloader  Downloader
	libDir      string
	autoLibs    bool
	processor   Processor
	engine      EngineOptions
	temperature float64
}

// Option configures the provider.
type Option = func(*options)

// WithName overrides the provider name.
func WithName(name string) Option { return func(o *options) { o.name = name } }

// WithCheckpoint selects which published bundle to load. A local directory
// containing manifest.json is also accepted.
func WithCheckpoint(ckpt Checkpoint) Option { return func(o *options) { o.checkpoint = ckpt } }

// WithDownloader customises where and how bundles are fetched.
func WithDownloader(d Downloader) Option { return func(o *options) { o.downloader = d } }

// WithLibDir sets the llama.cpp shared library directory.
func WithLibDir(dir string) Option { return func(o *options) { o.libDir = dir } }

// WithAutoLibraries installs the pinned llama.cpp release (LlamaCPPVersion)
// on first use when it is missing, outdated or fails verification. Every
// downloaded byte is checked against the release's digest manifest, whose
// own digest is compiled into this package, before anything is written or
// loaded. Without this option a missing install is ErrLibrariesMissing.
func WithAutoLibraries() Option {
	return func(o *options) { o.autoLibs = true }
}

// WithProcessor forces the llama.cpp backend flavour (cpu, metal, cuda,
// rocm, vulkan). The default detects one; $KEV_PROCESSOR also overrides.
func WithProcessor(p Processor) Option {
	return func(o *options) { o.processor = p }
}

// WithEngineOptions tunes threads, GPU offload and context size.
func WithEngineOptions(e EngineOptions) Option { return func(o *options) { o.engine = e } }

// WithTemperature overrides the checkpoint's calibration temperature; 1
// yields raw logits.
func WithTemperature(t float64) Option { return func(o *options) { o.temperature = t } }

type provider struct {
	options options

	mu    sync.Mutex
	model *evaluationModel
}

// New creates a Kev provider. Nothing is downloaded or loaded until the
// first EvaluationModel call.
func New(opts ...Option) (fantasy.Provider, error) {
	o := options{name: Name, checkpoint: DefaultCheckpoint}
	for _, opt := range opts {
		opt(&o)
	}
	return &provider{options: o}, nil
}

// Name implements fantasy.Provider.
func (p *provider) Name() string { return p.options.name }

// LanguageModel implements fantasy.Provider and always fails.
func (p *provider) LanguageModel(context.Context, string) (fantasy.LanguageModel, error) {
	return nil, ErrLanguageModelUnsupported
}

// EvaluationModel implements fantasy.EvaluationProvider. modelID may be
// empty, "kev-latest", or a Checkpoint name. Nothing is downloaded or
// loaded until the first Evaluate; call Load to warm up eagerly.
func (p *provider) EvaluationModel(_ context.Context, modelID string) (fantasy.EvaluationModel, error) {
	if modelID != "" && modelID != ModelLatest && Checkpoint(modelID) != p.options.checkpoint {
		return nil, fmt.Errorf("kev: provider is configured for %s, cannot serve %q", p.options.checkpoint, modelID)
	}
	return &lazyModel{provider: p}, nil
}

// Load downloads (if needed) and loads the model now instead of on first use.
func (p *provider) Load(ctx context.Context) error {
	_, err := p.loaded(ctx)
	return err
}

func (p *provider) loaded(ctx context.Context) (*evaluationModel, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.model != nil {
		return p.model, nil
	}
	model, err := p.load(ctx)
	if err != nil {
		return nil, err
	}
	p.model = model
	return model, nil
}

// lazyModel defers loading to the first Evaluate so constructing a model is
// free and never touches the network.
type lazyModel struct {
	provider *provider
}

// Provider implements fantasy.EvaluationModel.
func (m *lazyModel) Provider() string { return m.provider.options.name }

// Model implements fantasy.EvaluationModel.
func (m *lazyModel) Model() string { return string(m.provider.options.checkpoint) }

// Load downloads and loads the model now under ctx, so callers can give the
// install its own deadline and progress reporting before the first Evaluate.
func (m *lazyModel) Load(ctx context.Context) error { return m.provider.Load(ctx) }

// Evaluate implements fantasy.EvaluationModel. A first call may download
// gigabytes of weights; that work honours cancellation of ctx but not its
// deadline, which is sized for an evaluation, not an install. Call Load to
// control the install with its own context.
func (m *lazyModel) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	loadCtx, cancel := context.WithCancel(context.WithoutCancel(ctx))
	defer cancel()
	stop := context.AfterFunc(ctx, func() {
		if ctx.Err() == context.Canceled {
			cancel()
		}
	})
	defer stop()
	loaded, err := m.provider.loaded(loadCtx)
	if err != nil {
		return nil, err
	}
	return loaded.Evaluate(ctx, call)
}

func (p *provider) load(ctx context.Context) (*evaluationModel, error) {
	libDir := p.options.libDir
	if libDir == "" {
		dir, err := LibDir()
		if err != nil {
			return nil, err
		}
		libDir = dir
	}
	// Always verify what is on disk against the pinned release before
	// dlopen; only install when allowed.
	if _, err := EnsureLibraries(ctx, libDir, p.options.processor, p.options.autoLibs); err != nil {
		return nil, err
	}
	if err := loadLibraries(libDir); err != nil {
		return nil, err
	}

	bundle, err := p.options.downloader.Resolve(ctx, p.options.checkpoint)
	if err != nil {
		return nil, err
	}
	head, err := LoadHead(bundle.HeadPath)
	if err != nil {
		return nil, err
	}
	if p.options.temperature > 0 {
		head.Temperature = p.options.temperature
	}
	eng, err := newEngine(bundle.ModelPath, p.options.engine)
	if err != nil {
		return nil, err
	}
	if eng.nEmbd != head.HiddenSize {
		eng.close()
		return nil, fmt.Errorf("kev: model hidden size %d does not match head %d", eng.nEmbd, head.HiddenSize)
	}
	del, err := resolveDelimiters(eng)
	if err != nil {
		eng.close()
		return nil, err
	}
	return &evaluationModel{
		provider:   p.options.name,
		checkpoint: Checkpoint(cmp.Or(bundle.Manifest.Run, string(p.options.checkpoint))),
		engine:     eng,
		head:       head,
		del:        del,
	}, nil
}

// Close releases the loaded model, if any.
func (p *provider) Close() error {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.model != nil {
		p.model.engine.close()
		p.model = nil
	}
	return nil
}

type evaluationModel struct {
	provider   string
	checkpoint Checkpoint
	engine     *engine
	head       *Head
	del        delimiters
}

// Provider implements fantasy.EvaluationModel.
func (m *evaluationModel) Provider() string { return m.provider }

// Model implements fantasy.EvaluationModel.
func (m *evaluationModel) Model() string { return string(m.checkpoint) }

// Evaluate implements fantasy.EvaluationModel. Each question runs as its own
// causal row (state + branch), exactly as Kev serves hybrid backbones.
func (m *evaluationModel) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, fmt.Errorf("kev: invalid call: %w", err)
	}
	start := time.Now()
	text, err := stateText(call.State)
	if err != nil {
		return nil, err
	}
	state := stateTokens(m.engine, m.del, text)

	// Lower and render every question before touching the engine so a bad
	// question fails fast and the engine lock is held only for compute.
	type prepared struct {
		name    string
		lowered question
		br      branch
	}
	rows := make([]prepared, 0, len(call.Questions))
	for name, q := range call.Questions {
		lowered, err := lowerQuestion(q)
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", name, err)
		}
		br := renderBranch(m.engine, m.del, lowered.instructions, lowered.options)
		if len(state)+len(br.tokens) > MaxRowTokens {
			return nil, fmt.Errorf("question %q: %w (%d tokens)", name, ErrBranchTooLong, len(state)+len(br.tokens))
		}
		rows = append(rows, prepared{name: name, lowered: lowered, br: br})
	}

	answers := make(map[string]fantasy.EvaluationAnswer, len(rows))
	inputTokens := int64(len(state))
	for _, row := range rows {
		inputTokens += int64(len(row.br.tokens))
	}

	readout := func(row prepared, hidden [][]float32) error {
		probs, err := m.head.Probabilities(hidden[0], hidden[1:])
		if err != nil {
			return fmt.Errorf("question %q: %w", row.name, err)
		}
		answers[row.name] = row.lowered.read(probs)
		return nil
	}

	if len(rows) == 1 {
		// One decode of state+branch beats decode-state, fork, decode-branch
		// for a single question.
		row := rows[0]
		full := joinRow(state, row.br)
		hidden, err := m.engine.hiddenStates(full, append([]int{full.Decide}, full.OptEnds...))
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", row.name, err)
		}
		if err := readout(row, hidden); err != nil {
			return nil, err
		}
	} else {
		// Several questions share one decode of the state; each branch is
		// a fork of the cached prefix.
		session, err := m.engine.beginPrefix(state)
		if err != nil {
			return nil, err
		}
		defer session.end()
		for _, row := range rows {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			hidden, err := session.branch(row.br)
			if err != nil {
				return nil, fmt.Errorf("question %q: %w", row.name, err)
			}
			if err := readout(row, hidden); err != nil {
				return nil, err
			}
		}
	}

	return &fantasy.EvaluationResponse{
		Model:   cmp.Or(string(m.checkpoint), ModelLatest),
		Answers: answers,
		Usage:   fantasy.EvaluationUsage{InputTokens: inputTokens, TotalTokens: inputTokens},
		ProviderMetadata: fantasy.ProviderMetadata{
			m.provider: &typesafe.ProviderMetadata{LatencyMS: float64(time.Since(start).Microseconds()) / 1000},
		},
	}, nil
}

// joinRow concatenates the state prefix and a branch into one causal row.
func joinRow(state []int32, br branch) Row {
	tokens := make([]int32, 0, len(state)+len(br.tokens))
	tokens = append(tokens, state...)
	tokens = append(tokens, br.tokens...)
	optEnds := make([]int, len(br.optEnds))
	for i, e := range br.optEnds {
		optEnds[i] = len(state) + e
	}
	return Row{
		Tokens:  tokens,
		OptEnds: optEnds,
		Decide:  len(tokens) - 1,
		Options: br.options,
	}
}
