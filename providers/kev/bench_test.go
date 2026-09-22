package kev

import (
	"fmt"
	"os"
	"testing"

	"github.com/taigrr/fantasy"
)

// Run with:
//
//	KEV_TEST_BUNDLE=/path/to/bundle go test -run xxx -bench . -benchtime 20x ./providers/kev
func BenchmarkEvaluate(b *testing.B) {
	dir := os.Getenv(envTestBundle)
	if dir == "" {
		b.Skipf("set %s", envTestBundle)
	}
	opts := []Option{WithCheckpoint(Checkpoint(dir)), WithAutoLibraries()}
	if os.Getenv("KEV_BENCH_CPU") == "1" {
		opts = append(opts, WithEngineOptions(EngineOptions{GPULayers: -1}))
	}
	p, err := New(opts...)
	if err != nil {
		b.Fatal(err)
	}
	defer p.(*provider).Close() //nolint:errcheck
	if err := p.(*provider).Load(b.Context()); err != nil {
		b.Fatal(err)
	}
	model, _ := p.(fantasy.EvaluationProvider).EvaluationModel(b.Context(), "")

	short := fantasy.EvaluationCall{
		State:     "I was charged twice for my subscription.",
		Questions: map[string]fantasy.EvaluationQuestion{"refund": fantasy.BoolQuestion("Is the customer asking for money back?")},
	}
	three := fantasy.EvaluationCall{
		State: "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card. Fix this today.",
		Questions: map[string]fantasy.EvaluationQuestion{
			"department":  fantasy.ChoiceQuestion("Which team should handle this?", map[string]string{"returns": "Exchanges, refunds, wrong or damaged items", "shipping": "Delivery status, delays, lost packages", "billing": "Charges, invoices, payment problems"}),
			"escalate":    fantasy.BoolQuestion("Does this need urgent human attention?"),
			"frustration": fantasy.ScoreQuestion("How frustrated is the customer?", "Calm", "Frustrated", "Very angry"),
		},
	}
	longState := make([]byte, 0, 6000)
	for len(longState) < 6000 {
		longState = append(longState, "The deploy script fetches the artifact, verifies its checksum, stops the old service, swaps the symlink and restarts. "...)
	}
	long := fantasy.EvaluationCall{
		State:     string(longState),
		Questions: map[string]fantasy.EvaluationQuestion{"safe": fantasy.BoolQuestion("Is this procedure safe to run unattended?")},
	}

	eight := fantasy.EvaluationCall{State: three.State, Questions: map[string]fantasy.EvaluationQuestion{}}
	for i := range 8 {
		eight.Questions[fmt.Sprintf("q%d", i)] = fantasy.BoolQuestion(fmt.Sprintf("Is aspect %d of this complaint about billing?", i))
	}
	longThree := fantasy.EvaluationCall{State: long.State, Questions: three.Questions}

	for _, bc := range []struct {
		name string
		call fantasy.EvaluationCall
	}{{"1q_short", short}, {"3q_medium", three}, {"8q_medium", eight}, {"1q_long_~1500tok", long}, {"3q_long_~1500tok", longThree}} {
		b.Run(bc.name, func(b *testing.B) {
			// warm once so shader/graph compilation is excluded
			if _, err := model.Evaluate(b.Context(), bc.call); err != nil {
				b.Fatal(err)
			}
			b.ResetTimer()
			for b.Loop() {
				if _, err := model.Evaluate(b.Context(), bc.call); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
