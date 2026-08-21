//go:build !arm64 && !goexperiment.simd

package util

func dotProductFloat32(a, b []float32) float64 {
	return dotProductUnrolled8Scalar(a, b)
}
