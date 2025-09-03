## AdamW in Go

AdamW / AdamWR optimizer for Go (float64 vectors) with decoupled weight decay, bias correction, cosine annealing with warm restarts, and optional normalized weight decay per Loshchilov & Hutter (ICLR 2019).

- Package: `github.com/n0madic/go-adamw`
- Go: 1.23+

### Features
- AdamW with strict hyperparameter validation and finite checks.
- Schedules: fixed multiplier or cosine annealing with warm restarts.
- Normalized weight decay (Appendix B.1) via `NormConfig`.
- Deterministic updates; resettable state for reproducibility.
- **🚀 Zero-copy matrix optimization** - work directly with `[][]float64` without data copying.
- **Advanced BLAS optimization** with direct BLAS calls for improved performance.
- **Optimized element-wise operations** for sqrt, division, and clamping.
- **Kernel fusion** to reduce memory bandwidth requirements.
- **Adaptive strategy selection** automatically chooses optimal implementation based on vector size.
- **Pure Go implementation** - no external dependencies or CGO requirements.

### Install
```
go get github.com/n0madic/go-adamw
```

### Quick Start

#### Traditional Flat Parameter Optimization
```go
package main

import (
    "fmt"
    "math"

    adamw "github.com/n0madic/go-adamw"
    "gonum.org/v1/gonum/floats"
)

func main() {
    // Parameters θ we update in-place
    params := make([]float64, 1024)
    for i := range params {
        params[i] = 0.1 * math.Sin(float64(i))
    }

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
    fmt.Printf("final norm: %g\n", math.Sqrt(floats.Dot(params, params)))
}
```

#### 🚀 Matrix-Based Optimization (Zero-Copy)
```go
package main

import (
    "fmt"
    adamw "github.com/n0madic/go-adamw"
)

func main() {
    // Neural network parameters as matrices
    weights := [][]float64{
        {0.1, 0.2, 0.3},    // Layer 1 weights
        {0.4, 0.5, 0.6},    // Layer 2 weights
    }
    biases := [][]float64{
        {0.0},              // Layer 1 bias
        {0.1},              // Layer 2 bias
    }

    // Parameters as matrices (no copying required!)
    paramMatrices := [][]float64{
        weights[0], weights[1], biases[0], biases[1],
    }

    // DecayMask: apply weight decay to weights, not to biases
    decayMask := []bool{true, true, true, true, true, true, false, false}

    opt, err := adamw.NewFromMatrices(paramMatrices, adamw.Options{
        Alpha:       1e-3,
        Beta1:       0.9,
        Beta2:       0.999,
        Eps:         1e-8,
        WeightDecay: 1e-2,
        Schedule:    adamw.NewFixedSchedule(1.0, 0),
        DecayMask:   decayMask,
    })
    if err != nil { panic(err) }

    for step := 0; step < 100; step++ {
        // Simulate gradients (same structure as parameters)
        gradMatrices := [][]float64{
            {2.0 * weights[0][0], 2.0 * weights[0][1], 2.0 * weights[0][2]},
            {2.0 * weights[1][0], 2.0 * weights[1][1], 2.0 * weights[1][2]},
            {2.0 * biases[0][0]},
            {2.0 * biases[1][0]},
        }

        // Optimize directly on matrix data - no copying!
        if err := opt.StepMatrices(paramMatrices, gradMatrices); err != nil {
            panic(err)
        }
    }

    fmt.Printf("Final weights[0]: %v\n", weights[0])
    fmt.Printf("Final biases[0]: %v (notice: less decay)\n", biases[0])
}
```

Run the comprehensive example: `go run ./example`.

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

### Performance Optimization Levels

The library automatically selects the optimal optimization strategy based on vector size:

#### Level 1: Direct BLAS Operations
All vector operations use optimized BLAS calls (`blas64.Scal`, `blas64.Axpy`, etc.) instead of `floats` package:
- Replaces 6 `floats.Scale` calls with `blas64.Scal`
- Replaces 5 `floats.AddScaled` calls with `blas64.Axpy`
- Universal performance improvement across all vector sizes

#### Level 2: Element-wise Operations
Critical scalar loops optimized for better performance:
- `elementWiseSquare()`: Gradient squaring (g²)
- `clampSqrtAddEps()`: Clamping + sqrt + epsilon addition
- `elementWiseDivide()`: Element-wise division

#### Level 3: Kernel Fusion
Multiple operations combined into single passes to reduce memory traffic:
- `momentUpdateFusion()`: Updates both m and v moments in one pass
- `biasCorrectClampSqrtFusion()`: Combines bias correction + clamp + sqrt operations
- Significant memory bandwidth reduction for large vectors

#### Level 4: Heavy Kernel Fusion
Aggressive optimization for large vectors (≥4096) with specialized fusion kernels:
- `fullMomentBiasFusion()`: Combines moment updates with immediate bias correction
- `adaptiveUpdateCompleteFusion()`: Fuses bias correction + clamp + sqrt + scale + divide
- `parameterUpdateFusion()`: Combines adaptive update + weight decay + parameter update
- **Memory bandwidth reduction**: ~60% fewer memory passes compared to standard approach

#### Level 5: Adaptive Strategy Selection
Automatic optimization strategy based on vector characteristics:
- **Small vectors (<512)**: `StrategyPureBLAS` - minimal overhead
- **Medium vectors (512-4096)**: `StrategyFusion` - moderate kernel fusion
- **Large vectors (≥4096)**: `StrategyHeavyFusion` - maximum fusion with specialized kernels
- **Pure Go**: All optimizations work without external dependencies

### API Surface (essentials)

#### Core Optimizer
- `type Optimizer`: `.Step(params, grad)`, `.StepMatrices(paramMatrices, gradMatrices)`, `.ResetState()`, `.CurrentStep()`.
- `type Options`: `Alpha`, `Beta1`, `Beta2`, `Eps`, `WeightDecay` or `Norm *NormConfig`, `Schedule`, `DecayMask`.

#### Constructors
- `New(params []float64, opt Options)` - traditional flat parameter optimization
- `NewFromMatrices(paramMatrices [][]float64, opt Options)` - zero-copy matrix optimization

#### Configuration
- `type NormConfig`: `LambdaNorm`, `BatchSize`, `DatasetSize`, `TotalEpochs`, `StepsPerEpoch`.
- `type Schedule` and implementations: `FixedSchedule`, `CosineAnnealingWarmRestarts`.
- Automatic optimization strategy selection based on vector size (no configuration needed).

#### Matrix Operations (Advanced)
- `type MatrixView`: `.Get(index)`, `.Set(index, value)`, `.ForEach(fn)`, `.ForEachMutable(fn)`.
- `ToFlat()` / `CopyFromFlat()` - available but discouraged (prefer zero-copy methods).

### Testing & Benchmarks
- Unit tests: `go test ./...`
- Race detector: `go test -race ./...`
- Fuzz tests: `go test -fuzz=FuzzStepStability -run ^$ -fuzztime=30s`
- Performance benchmarks: `go test -bench . -benchmem`

### Notes
- Inputs must be finite; parameter/gradient lengths must match optimizer state.
- `ResetState` clears moments and schedule state for reproducibility.
