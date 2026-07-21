//go:build arm64

#include "textflag.h"

// func mulFloat32NEON(dst, a, b []float32)
TEXT ·mulFloat32NEON(SB), NOSPLIT|NOFRAME, $0-72
	MOVD dst_base+0(FP), R0
	MOVD dst_len+8(FP), R3
	MOVD a_base+24(FP), R1
	MOVD b_base+48(FP), R2

	LSR $2, R3, R3
	CBZ R3, mul_done

mul_loop:
	VLD1.P 16(R1), [V0.S4]
	VLD1.P 16(R2), [V1.S4]
	VEOR V2.B16, V2.B16, V2.B16
	VFMLA V1.S4, V0.S4, V2.S4
	VST1.P [V2.S4], 16(R0)
	SUB $1, R3, R3
	CBNZ R3, mul_loop

mul_done:
	RET

// func scaleFloat32NEON(values []float32, scale float32)
TEXT ·scaleFloat32NEON(SB), NOSPLIT|NOFRAME, $0-28
	MOVD values_base+0(FP), R0
	MOVD values_len+8(FP), R1
	FMOVS scale+24(FP), F0
	VDUP V0.S[0], V1.S4

	LSR $2, R1, R1
	CBZ R1, scale_done

scale_loop:
	VLD1 (R0), [V2.S4]
	VEOR V3.B16, V3.B16, V3.B16
	VFMLA V1.S4, V2.S4, V3.S4
	VST1.P [V3.S4], 16(R0)
	SUB $1, R1, R1
	CBNZ R1, scale_loop

scale_done:
	RET

// func fwhtStepFloat32NEON(values []float32, step int)
TEXT ·fwhtStepFloat32NEON(SB), NOSPLIT|NOFRAME, $0-32
	MOVD values_base+0(FP), R0
	MOVD values_len+8(FP), R1
	MOVD step+24(FP), R2

	LSL $2, R1, R1 // len bytes
	LSL $2, R2, R2 // step bytes
	LSL $1, R2, R3 // two-step block bytes
	MOVD R0, R4     // block pointer
	MOVD R1, R8     // remaining bytes
	MOVD $0x3f800000, R9
	FMOVS R9, F4
	VDUP V4.S[0], V4.S4

	CBZ R8, fwht_done

fwht_block:
	MOVD R4, R5
	ADD R2, R4, R6
	MOVD R2, R7

fwht_inner:
	VLD1 (R5), [V0.S4]
	VLD1 (R6), [V1.S4]
	VMOV V1.B16, V2.B16
	VFMLA V4.S4, V0.S4, V2.S4
	VMOV V0.B16, V3.B16
	VFMLS V4.S4, V1.S4, V3.S4
	VST1 [V2.S4], (R5)
	VST1 [V3.S4], (R6)
	ADD $16, R5, R5
	ADD $16, R6, R6
	SUB $16, R7, R7
	CBNZ R7, fwht_inner

	ADD R3, R4, R4
	SUB R3, R8, R8
	CBNZ R8, fwht_block

fwht_done:
	RET
