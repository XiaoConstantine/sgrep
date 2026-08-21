//go:build !arm64 && !goexperiment.simd

package util

func mulFloat32Into(dst, a, b []float32) {
	mulFloat32Scalar(dst, a, b)
}

func scaleFloat32(values []float32, scale float32) {
	scaleFloat32Scalar(values, scale)
}

func fwhtStepFloat32(values []float32, step int) {
	fwhtStepFloat32Scalar(values, step)
}

func buildTQMSETable(table []float64, rotated, centroids []float32, levels int) {
	if levels == 16 {
		buildTQMSETable16Scalar(table, rotated, centroids)
		return
	}
	buildTQMSETableScalar(table, rotated, centroids, levels)
}
