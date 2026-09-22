package kev

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"slices"
)

// Head is Kev's pointer head: a single scaled dot-product attention between
// the <decide> hidden state (query) and each </opt> hidden state (key).
// logit_i = (K h_opt_i) · (Q h_decide) / sqrt(dp) / T, then softmax.
type Head struct {
	HiddenSize  int
	HeadDim     int
	Temperature float64

	qWeight []float32 // [HeadDim][HiddenSize] row-major
	qBias   []float32 // [HeadDim]
	kWeight []float32 // [HeadDim][HiddenSize] row-major
	kBias   []float32 // [HeadDim]
}

// headFile is the on-disk head.json layout written by the conversion script.
type headFile struct {
	Format      int         `json:"format"`
	Base        string      `json:"base"`
	HiddenSize  int         `json:"hidden_size"`
	HeadDim     int         `json:"head_dim"`
	Temperature float64     `json:"temperature"`
	QWeight     [][]float32 `json:"q_weight"`
	QBias       []float32   `json:"q_bias"`
	KWeight     [][]float32 `json:"k_weight"`
	KBias       []float32   `json:"k_bias"`
}

const headFormat = 1

// LoadHead reads a head.json produced by the converter.
func LoadHead(path string) (*Head, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("kev: read head: %w", err)
	}
	return ParseHead(data)
}

// ParseHead decodes head.json bytes.
func ParseHead(data []byte) (*Head, error) {
	var file headFile
	if err := json.Unmarshal(data, &file); err != nil {
		return nil, fmt.Errorf("kev: decode head: %w", err)
	}
	if file.Format != headFormat {
		return nil, fmt.Errorf("kev: unsupported head format %d", file.Format)
	}
	if file.HeadDim == 0 || file.HiddenSize == 0 {
		return nil, errors.New("kev: head missing dimensions")
	}
	if len(file.QWeight) != file.HeadDim || len(file.KWeight) != file.HeadDim ||
		len(file.QBias) != file.HeadDim || len(file.KBias) != file.HeadDim {
		return nil, errors.New("kev: head weight shapes do not match head_dim")
	}
	head := &Head{
		HiddenSize:  file.HiddenSize,
		HeadDim:     file.HeadDim,
		Temperature: file.Temperature,
		qWeight:     flatten(file.QWeight, file.HiddenSize),
		qBias:       file.QBias,
		kWeight:     flatten(file.KWeight, file.HiddenSize),
		kBias:       file.KBias,
	}
	if head.qWeight == nil || head.kWeight == nil {
		return nil, errors.New("kev: head weight rows do not match hidden_size")
	}
	if head.Temperature <= 0 {
		head.Temperature = 1
	}
	return head, nil
}

func flatten(rows [][]float32, width int) []float32 {
	out := make([]float32, 0, len(rows)*width)
	for _, row := range rows {
		if len(row) != width {
			return nil
		}
		out = append(out, row...)
	}
	return out
}

// project computes W h + b for a row-major [HeadDim][HiddenSize] matrix.
func (h *Head) project(weight, bias, hidden []float32) []float64 {
	out := make([]float64, h.HeadDim)
	for i := range h.HeadDim {
		row := weight[i*h.HiddenSize : (i+1)*h.HiddenSize]
		sum := float64(bias[i])
		for j, w := range row {
			sum += float64(w) * float64(hidden[j])
		}
		out[i] = sum
	}
	return out
}

// Logits scores each option hidden state against the decide hidden state.
// Temperature is applied here, matching PointerHead.forward in eval mode.
func (h *Head) Logits(decide []float32, options [][]float32) ([]float64, error) {
	if len(decide) != h.HiddenSize {
		return nil, fmt.Errorf("kev: decide hidden size %d, head expects %d", len(decide), h.HiddenSize)
	}
	query := h.project(h.qWeight, h.qBias, decide)
	scale := 1 / math.Sqrt(float64(h.HeadDim))
	logits := make([]float64, len(options))
	for i, opt := range options {
		if len(opt) != h.HiddenSize {
			return nil, fmt.Errorf("kev: option hidden size %d, head expects %d", len(opt), h.HiddenSize)
		}
		key := h.project(h.kWeight, h.kBias, opt)
		dot := 0.0
		for j := range query {
			dot += key[j] * query[j]
		}
		logits[i] = dot * scale / h.Temperature
	}
	return logits, nil
}

// Probabilities returns softmax(Logits).
func (h *Head) Probabilities(decide []float32, options [][]float32) ([]float64, error) {
	logits, err := h.Logits(decide, options)
	if err != nil {
		return nil, err
	}
	return softmax(logits), nil
}

func softmax(z []float64) []float64 {
	if len(z) == 0 {
		return nil
	}
	maxZ := slices.Max(z)
	out := make([]float64, len(z))
	sum := 0.0
	for i, v := range z {
		out[i] = math.Exp(v - maxZ)
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// round2 matches kev/api.py's r2: two decimals on every reported number.
func round2(v float64) float64 {
	return math.Round(v*100) / 100
}
