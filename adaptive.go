package adamw

// OptimizationStrategy represents different optimization strategies
type OptimizationStrategy int

const (
	// StrategyPureBLAS uses only BLAS operations with minimal SIMD/fusion
	StrategyPureBLAS OptimizationStrategy = iota
	// StrategyFusion uses kernel fusion for medium-sized vectors
	StrategyFusion
	// StrategyHeavyFusion uses aggressive fusion + SIMD for large vectors
	StrategyHeavyFusion
)

// AdaptiveConfig holds configuration for adaptive optimization selection
type AdaptiveConfig struct {
	// Thresholds for different strategies based on vector size
	SmallVectorThreshold int // < this size: use StrategyPureBLAS
	LargeVectorThreshold int // >= this size: use StrategyHeavyFusion

	// Performance characteristics (can be measured at runtime)
	MemoryBandwidthGBps float64
	L3CacheSizeMB       int
}

// DefaultAdaptiveConfig returns sensible defaults for the current system
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		SmallVectorThreshold: 512,  // Below this: simple BLAS operations
		LargeVectorThreshold: 4096, // Above this: aggressive optimization
		MemoryBandwidthGBps:  25.0, // Conservative estimate
		L3CacheSizeMB:        16,   // Conservative estimate
	}
}

// SelectOptimizationStrategy chooses the best strategy for given vector size
func SelectOptimizationStrategy(vectorSize int, config AdaptiveConfig) OptimizationStrategy {
	if vectorSize < config.SmallVectorThreshold {
		// Small vectors: overhead of fusion/SIMD may not be worth it
		return StrategyPureBLAS
	}

	if vectorSize >= config.LargeVectorThreshold {
		// Large vectors: memory bandwidth becomes critical
		// Use aggressive fusion to reduce memory traffic
		return StrategyHeavyFusion
	}

	// Medium vectors: use moderate fusion
	return StrategyFusion
}

// OptimizeForCache estimates if vector fits in cache and adjusts strategy
func OptimizeForCache(vectorSize int, config AdaptiveConfig) bool {
	// Estimate memory usage: each float64 is 8 bytes
	// AdamW needs: params, grad, m, v, mhat, vhat, update = 7 vectors
	estimatedMemoryMB := float64(vectorSize*7*8) / (1024 * 1024)

	// If working set fits in L3 cache, prefer cache-friendly strategies
	return estimatedMemoryMB <= float64(config.L3CacheSizeMB)*0.8 // 80% utilization
}
