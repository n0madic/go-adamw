## AdamW in Go

AdamW / AdamWR optimizer for Go (float64 vectors) with decoupled weight decay, bias correction, cosine annealing with warm restarts, and optional normalized weight decay per Loshchilov & Hutter (ICLR 2019).

- Package: `github.com/n0madic/go-adamw`
- Go: 1.20+

### Features
- AdamW with strict hyperparameter validation and finite checks.
- Schedules: fixed multiplier or cosine annealing with warm restarts.
- Normalized weight decay (Appendix B.1) via `NormConfig`.
- Deterministic updates; resettable state for reproducibility.

### Install
```
go get github.com/n0madic/go-adamw
```

### Quick Start
```go
package main

import (
    "fmt"
    adamw "github.com/n0madic/go-adamw"
)

func main() {
    // Parameters θ we update in-place
    params := []float64{0.5, -1.0, 2.0}

    opt, err := adamw.New(params, adamw.Options{
        Alpha:       1e-3,     // base lr
        Beta1:       0.9,
        Beta2:       0.999,
        Eps:         1e-8,
        WeightDecay: 1e-2,     // decoupled λ; set <0 to disable
        Schedule:    adamw.NewFixedSchedule(1.0, 0), // η_t ≡ 1
    })
    if err != nil { panic(err) }

    // Dummy objective: ||θ||^2 → gradient = 2θ
    for step := 0; step < 1000; step++ {
        grad := make([]float64, len(params))
        for i := range params { grad[i] = 2 * params[i] }
        if err := opt.Step(params, grad); err != nil { panic(err) }
    }
    fmt.Printf("params: %#v\n", params)
}
```
Run the example in this repo: `go run ./example`.

### Schedules
- `FixedSchedule`: constant multiplier `η_t` (use `NewFixedSchedule(eta, 0)`).
- `CosineAnnealingWarmRestarts`: Eq. (15) with restarts; create via `NewCosineAnnealingWarmRestarts(initialPeriodSteps, tMult)`.

Query current period: `periodSteps, hasRestarts := sched.PeriodInfo()`.

### Normalized Weight Decay (B.1)
Provide `NormConfig` in `Options` to derive λ from `λ_norm`:
```
λ = λ_norm * sqrt(batch / (dataset * T))
```
- When using warm restarts, `T` is inferred from current period and `StepsPerEpoch`.
- Otherwise set `TotalEpochs` directly.

### API Surface (essentials)
- `type Optimizer`: `.Step(params, grad)`, `.ResetState()`, `.CurrentStep()`.
- `type Options`: `Alpha`, `Beta1`, `Beta2`, `Eps`, `WeightDecay` or `Norm *NormConfig`, `Schedule`.
- `type NormConfig`: `LambdaNorm`, `BatchSize`, `DatasetSize`, `TotalEpochs`, `StepsPerEpoch`.
- `type Schedule` and implementations: `FixedSchedule`, `CosineAnnealingWarmRestarts`.

### Testing & Benchmarks
- Unit tests: `go test ./...`
- Race detector: `go test -race ./...`
- Fuzz tests: `go test -fuzz=FuzzStepStability -run ^$ -fuzztime=30s`
- Benchmarks: `go test -bench . -benchmem`

### Notes
- Inputs must be finite; parameter/gradient lengths must match optimizer state.
- `ResetState` clears moments and schedule state for reproducibility.
