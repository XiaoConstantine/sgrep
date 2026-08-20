//go:build linux

package rerank

import (
	"encoding/binary"
	"runtime"
	"testing"

	"golang.org/x/sys/unix"
)

func TestRerankerSeccompPolicy(t *testing.T) {
	filters, err := rerankerSeccompFilters()
	if err != nil {
		t.Fatal(err)
	}
	architecture := uint32(unix.AUDIT_ARCH_AARCH64)
	if runtime.GOARCH == "amd64" {
		architecture = unix.AUDIT_ARCH_X86_64
	}
	deny := uint32(unix.SECCOMP_RET_ERRNO) | uint32(unix.EPERM)
	unsupported := uint32(unix.SECCOMP_RET_ERRNO) | uint32(unix.ENOSYS)

	tests := []struct {
		name string
		nr   uint32
		arch uint32
		arg0 uint64
		want uint32
	}{
		{name: "ordinary syscall", nr: uint32(unix.SYS_GETPID), arch: architecture, want: unix.SECCOMP_RET_ALLOW},
		{name: "setpgid", nr: uint32(unix.SYS_SETPGID), arch: architecture, want: deny},
		{name: "setsid", nr: uint32(unix.SYS_SETSID), arch: architecture, want: deny},
		{name: "clone process", nr: uint32(unix.SYS_CLONE), arch: architecture, want: deny},
		{name: "clone thread", nr: uint32(unix.SYS_CLONE), arch: architecture, arg0: unix.CLONE_THREAD, want: unix.SECCOMP_RET_ALLOW},
		{name: "clone3", nr: uint32(unix.SYS_CLONE3), arch: architecture, want: unsupported},
		{name: "wrong architecture", nr: uint32(unix.SYS_GETPID), arch: 0, want: unix.SECCOMP_RET_KILL_PROCESS},
	}
	if runtime.GOARCH == "amd64" {
		tests = append(tests, struct {
			name string
			nr   uint32
			arch uint32
			arg0 uint64
			want uint32
		}{name: "x32 ABI", nr: uint32(unix.SYS_GETPID) | 0x40000000, arch: architecture, want: unsupported})
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := evaluateRerankerSeccomp(t, filters, test.nr, test.arch, test.arg0); got != test.want {
				t.Fatalf("seccomp result = %#x, want %#x", got, test.want)
			}
		})
	}
}

func evaluateRerankerSeccomp(t *testing.T, filters []unix.SockFilter, nr, architecture uint32, arg0 uint64) uint32 {
	t.Helper()
	data := make([]byte, 64)
	binary.LittleEndian.PutUint32(data[0:4], nr)
	binary.LittleEndian.PutUint32(data[4:8], architecture)
	binary.LittleEndian.PutUint64(data[16:24], arg0)
	var accumulator uint32
	for pc := 0; pc < len(filters); pc++ {
		instruction := filters[pc]
		switch instruction.Code {
		case unix.BPF_LD | unix.BPF_W | unix.BPF_ABS:
			offset := int(instruction.K)
			if offset < 0 || offset+4 > len(data) {
				t.Fatalf("BPF load offset %d is outside seccomp_data", offset)
			}
			accumulator = binary.LittleEndian.Uint32(data[offset : offset+4])
		case unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K:
			if accumulator == instruction.K {
				pc += int(instruction.Jt)
			} else {
				pc += int(instruction.Jf)
			}
		case unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K:
			if accumulator&instruction.K != 0 {
				pc += int(instruction.Jt)
			} else {
				pc += int(instruction.Jf)
			}
		case unix.BPF_RET | unix.BPF_K:
			return instruction.K
		default:
			t.Fatalf("unsupported BPF instruction %#x", instruction.Code)
		}
	}
	t.Fatal("seccomp program reached the end without returning")
	return 0
}
