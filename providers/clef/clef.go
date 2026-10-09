// Package clef runs Cloudflare's Clef decision models fully in-process using
// llama.cpp via yzma. Clef (huggingface.co/Cloudflare/clef) is a Qwen3.5
// backbone with a joint schema head that scores every option of every
// question in one forward pass; llama.cpp implements the head natively, so
// a converted GGUF is all that is needed.
//
// Checkpoints are downloaded on first use from Hugging Face as GGUF bundles
// so binaries never embed weights; every file is SHA-256 checked against a
// manifest whose digest is compiled into this package. The llama.cpp shared
// libraries are pinned to LlamaCPPVersion and, with WithAutoLibraries,
// installed the same way.
//
//	p, err := clef.New(clef.WithAutoLibraries())
//	model, err := p.(fantasy.EvaluationProvider).EvaluationModel(ctx, "")
//	resp, err := model.Evaluate(ctx, fantasy.EvaluationCall{...})
//
// Images, a Clef extension on Workers AI, are not supported in-process:
// the bundles are text-only.
package clef

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"sort"
	"sync"
	"time"

	"github.com/taigrr/fantasy"
	"github.com/taigrr/fantasy/providers/internal/localgguf"
	"github.com/taigrr/fantasy/providers/typesafe"
)

const (
	// Name is the provider name.
	Name = "clef"
	// ModelLatest is the alias accepted for the configured checkpoint.
	ModelLatest = "clef-latest"

	// LlamaCPPVersion is the llama.cpp build this package is tested
	// against; see localgguf.LlamaCPPVersion.
	LlamaCPPVersion = localgguf.LlamaCPPVersion
	// EnvLibPath overrides where the llama.cpp shared libraries live.
	EnvLibPath = localgguf.EnvLibPath
	// EnvVerbose leaves llama.cpp native logging enabled when set to 1.
	EnvVerbose = "CLEF_VERBOSE"
	// EnvProcessor forces the llama.cpp backend flavour to install:
	// cpu, metal, cuda, rocm or vulkan.
	EnvProcessor = "CLEF_PROCESSOR"
)

// Processor is a llama.cpp backend flavour.
type Processor = localgguf.Processor

// Supported backend flavours.
const (
	ProcessorAuto   = localgguf.ProcessorAuto
	ProcessorCPU    = localgguf.ProcessorCPU
	ProcessorMetal  = localgguf.ProcessorMetal
	ProcessorCUDA   = localgguf.ProcessorCUDA
	ProcessorROCm   = localgguf.ProcessorROCm
	ProcessorVulkan = localgguf.ProcessorVulkan
)

// ErrLanguageModelUnsupported is returned by LanguageModel.
var ErrLanguageModelUnsupported = errors.New("clef: language models are not supported; use EvaluationModel")

// ErrTooManyQuestions is returned when a call carries more than MaxQuestions.
var ErrTooManyQuestions = errors.New("clef: too many questions")

var (
	_ fantasy.EvaluationProvider = (*provider)(nil)
	_ fantasy.EvaluationModel    = (*evaluationModel)(nil)
	_ fantasy.EvaluationModel    = (*lazyModel)(nil)
)

type options struct {
	name       string
	checkpoint Checkpoint
	downloader Downloader
	libDir     string
	autoLibs   bool
	processor  Processor
	engine     EngineOptions
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
// on first use when it is missing, outdated or fails verification. Without
// this option a missing install is an error.
func WithAutoLibraries() Option { return func(o *options) { o.autoLibs = true } }

// WithProcessor forces the llama.cpp backend flavour (cpu, metal, cuda,
// rocm, vulkan). The default detects one; $CLEF_PROCESSOR also overrides.
func WithProcessor(p Processor) Option { return func(o *options) { o.processor = p } }

// WithEngineOptions tunes threads, GPU offload and the prompt budget.
func WithEngineOptions(e EngineOptions) Option { return func(o *options) { o.engine = e } }

type provider struct {
	options options

	mu    sync.Mutex
	model *evaluationModel
}

// New creates a Clef provider. Nothing is downloaded or loaded until the
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
// empty, "clef-latest", or the configured Checkpoint name. Nothing is
// downloaded or loaded until the first Evaluate; call Load to warm up.
func (p *provider) EvaluationModel(_ context.Context, modelID string) (fantasy.EvaluationModel, error) {
	if modelID != "" && modelID != ModelLatest && Checkpoint(modelID) != p.options.checkpoint {
		return nil, fmt.Errorf("clef: provider is configured for %s, cannot serve %q", p.options.checkpoint, modelID)
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

// LibDir resolves the llama.cpp library directory: $YZMA_LIB, then
// $CLEF_CACHE/lib/<tag> when CLEF_CACHE is set, else the install shared by
// every in-process provider under the OS cache directory.
func LibDir() (string, error) {
	return localgguf.DefaultLibDir(EnvCacheDir)
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
	if _, err := localgguf.EnsureLibraries(ctx, libDir, p.options.processor, EnvProcessor, p.options.autoLibs); err != nil {
		return nil, fmt.Errorf("clef: %w", err)
	}
	if err := localgguf.LoadLibraries(libDir, EnvVerbose); err != nil {
		return nil, fmt.Errorf("clef: %w", err)
	}

	bundle, err := p.options.downloader.Resolve(ctx, p.options.checkpoint)
	if err != nil {
		return nil, fmt.Errorf("clef: %w", err)
	}
	eng, err := newEngine(bundle.ModelPath, p.options.engine)
	if err != nil {
		return nil, err
	}
	return &evaluationModel{
		provider:   p.options.name,
		checkpoint: Checkpoint(cmp.Or(bundle.Manifest.Run, string(p.options.checkpoint))),
		engine:     eng,
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

type evaluationModel struct {
	provider   string
	checkpoint Checkpoint
	engine     *engine
}

// Provider implements fantasy.EvaluationModel.
func (m *evaluationModel) Provider() string { return m.provider }

// Model implements fantasy.EvaluationModel.
func (m *evaluationModel) Model() string { return string(m.checkpoint) }

// Evaluate implements fantasy.EvaluationModel. All questions are decided
// jointly in one prefill. Questions are laid out in the prompt sorted by
// name, since Go maps carry no order; the hosted API uses request order,
// which can move probabilities slightly.
func (m *evaluationModel) Evaluate(ctx context.Context, call fantasy.EvaluationCall) (*fantasy.EvaluationResponse, error) {
	if err := call.Validate(); err != nil {
		return nil, fmt.Errorf("clef: invalid call: %w", err)
	}
	if len(call.Questions) > MaxQuestions {
		return nil, fmt.Errorf("%w: %d, maximum is %d", ErrTooManyQuestions, len(call.Questions), MaxQuestions)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	start := time.Now()
	text, err := stateText(call.State)
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(call.Questions))
	for name := range call.Questions {
		names = append(names, name)
	}
	sort.Strings(names)
	questions := make([]question, 0, len(names))
	nOptions := 0
	for _, name := range names {
		lowered, err := lowerQuestion(name, call.Questions[name])
		if err != nil {
			return nil, err
		}
		questions = append(questions, lowered)
		nOptions += len(lowered.options)
	}
	rendered, err := renderPrompt(m.engine, text, questions, m.engine.maxTokens)
	if err != nil {
		return nil, err
	}
	scores, err := m.engine.scores(rendered, nOptions)
	if err != nil {
		return nil, err
	}

	answers := make(map[string]fantasy.EvaluationAnswer, len(questions))
	offset := 0
	for _, q := range questions {
		probs := softmax(scores[offset : offset+len(q.options)])
		offset += len(q.options)
		answers[q.name] = q.read(probs)
	}
	inputTokens := int64(len(rendered.tokens))
	return &fantasy.EvaluationResponse{
		Model:   cmp.Or(string(m.checkpoint), ModelLatest),
		Answers: answers,
		Usage:   fantasy.EvaluationUsage{InputTokens: inputTokens, TotalTokens: inputTokens},
		ProviderMetadata: fantasy.ProviderMetadata{
			m.provider: &typesafe.ProviderMetadata{LatencyMS: float64(time.Since(start).Microseconds()) / 1000},
		},
	}, nil
}
