//go:build arm64

package util

//go:noescape
func dotProductFloat32NEON(a, b []float32) float64

func dotProductFloat32(a, b []float32) float64 {
	if len(a) < 32 {
		return dotProductUnrolled8Scalar(a, b)
	}
	return dotProductFloat32NEON(a, b)
}
