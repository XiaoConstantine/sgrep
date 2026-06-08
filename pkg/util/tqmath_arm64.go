//go:build arm64

package util

//go:noescape
func mulFloat32NEON(dst, a, b []float32)

//go:noescape
func scaleFloat32NEON(values []float32, scale float32)

//go:noescape
func fwhtStepFloat32NEON(values []float32, step int)

func mulFloat32Into(dst, a, b []float32) {
	n := len(a)
	main := n &^ 3
	if main > 0 {
		mulFloat32NEON(dst[:main], a[:main], b[:main])
	}
	if main < n {
		mulFloat32Scalar(dst[main:], a[main:], b[main:])
	}
}

func scaleFloat32(values []float32, scale float32) {
	main := len(values) &^ 3
	if main > 0 {
		scaleFloat32NEON(values[:main], scale)
	}
	if main < len(values) {
		scaleFloat32Scalar(values[main:], scale)
	}
}

func fwhtStepFloat32(values []float32, step int) {
	if step >= 4 && step&3 == 0 {
		fwhtStepFloat32NEON(values, step)
		return
	}
	fwhtStepFloat32Scalar(values, step)
}

func buildTQMSETable(table []float64, rotated, centroids []float32, levels int) {
	if levels == 16 {
		buildTQMSETable16Scalar(table, rotated, centroids)
		return
	}
	buildTQMSETableScalar(table, rotated, centroids, levels)
}
