package util

func mulFloat32Scalar(dst, a, b []float32) {
	for i, value := range a {
		dst[i] = value * b[i]
	}
}

func scaleFloat32Scalar(values []float32, scale float32) {
	for i := range values {
		values[i] *= scale
	}
}

func fwhtStepFloat32Scalar(values []float32, step int) {
	for start := 0; start < len(values); start += 2 * step {
		for i := 0; i < step; i++ {
			a := values[start+i]
			b := values[start+i+step]
			values[start+i] = a + b
			values[start+i+step] = a - b
		}
	}
}

func buildTQMSETableScalar(table []float64, rotated, centroids []float32, levels int) {
	for dim, value := range rotated {
		rotatedQ := float64(value)
		offset := dim * levels
		for code := 0; code < levels; code++ {
			table[offset+code] = rotatedQ * float64(centroids[code])
		}
	}
}

func buildTQMSETable16Scalar(table []float64, rotated, centroids []float32) {
	c0 := float64(centroids[0])
	c1 := float64(centroids[1])
	c2 := float64(centroids[2])
	c3 := float64(centroids[3])
	c4 := float64(centroids[4])
	c5 := float64(centroids[5])
	c6 := float64(centroids[6])
	c7 := float64(centroids[7])
	c8 := float64(centroids[8])
	c9 := float64(centroids[9])
	c10 := float64(centroids[10])
	c11 := float64(centroids[11])
	c12 := float64(centroids[12])
	c13 := float64(centroids[13])
	c14 := float64(centroids[14])
	c15 := float64(centroids[15])

	for dim, value := range rotated {
		rotatedQ := float64(value)
		offset := dim << 4
		table[offset] = rotatedQ * c0
		table[offset+1] = rotatedQ * c1
		table[offset+2] = rotatedQ * c2
		table[offset+3] = rotatedQ * c3
		table[offset+4] = rotatedQ * c4
		table[offset+5] = rotatedQ * c5
		table[offset+6] = rotatedQ * c6
		table[offset+7] = rotatedQ * c7
		table[offset+8] = rotatedQ * c8
		table[offset+9] = rotatedQ * c9
		table[offset+10] = rotatedQ * c10
		table[offset+11] = rotatedQ * c11
		table[offset+12] = rotatedQ * c12
		table[offset+13] = rotatedQ * c13
		table[offset+14] = rotatedQ * c14
		table[offset+15] = rotatedQ * c15
	}
}
