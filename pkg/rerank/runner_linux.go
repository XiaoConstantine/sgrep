//go:build linux

package rerank

import (
	"fmt"
	"runtime"
	"unsafe"

	"golang.org/x/sys/unix"
)

func preserveRerankerIdentityAfterExec() bool {
	return false
}

func prepareConstrainedReranker(executable string, args []string) (string, []string, error) {
	if err := denyRerankerProcessCreation(); err != nil {
		return "", nil, fmt.Errorf("confine reranker process creation: %w", err)
	}
	return executable, append([]string{executable}, args...), nil
}

func denyRerankerProcessCreation() error {
	filters, err := rerankerSeccompFilters()
	if err != nil {
		return err
	}
	program := unix.SockFprog{Len: uint16(len(filters)), Filter: &filters[0]}
	if err := unix.Prctl(unix.PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0); err != nil {
		return err
	}
	_, _, errno := unix.Syscall6(
		unix.SYS_SECCOMP,
		unix.SECCOMP_SET_MODE_FILTER,
		unix.SECCOMP_FILTER_FLAG_TSYNC,
		uintptr(unsafe.Pointer(&program)),
		0,
		0,
		0,
	)
	if errno != 0 {
		return errno
	}
	return nil
}

func rerankerSeccompFilters() ([]unix.SockFilter, error) {
	var auditArchitecture uint32
	var forkSyscall uint32
	var vforkSyscall uint32
	switch runtime.GOARCH {
	case "amd64":
		auditArchitecture = unix.AUDIT_ARCH_X86_64
		forkSyscall = 57
		vforkSyscall = 58
	case "arm64":
		auditArchitecture = unix.AUDIT_ARCH_AARCH64
		// arm64 has no dedicated fork or vfork syscalls; libc uses clone.
		forkSyscall = ^uint32(0)
		vforkSyscall = ^uint32(0) - 1
	default:
		return nil, fmt.Errorf("unsupported Linux architecture %s", runtime.GOARCH)
	}

	deny := uint32(unix.SECCOMP_RET_ERRNO) | uint32(unix.EPERM)
	unsupported := uint32(unix.SECCOMP_RET_ERRNO) | uint32(unix.ENOSYS)
	filters := []unix.SockFilter{
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 4},
		{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jt: 1, K: auditArchitecture},
		{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_KILL_PROCESS},
		{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 0},
	}
	if runtime.GOARCH == "amd64" {
		const x32SyscallBit = uint32(0x40000000)
		filters = append(filters,
			unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K, Jf: 1, K: x32SyscallBit},
			unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unsupported},
		)
	}
	filters = append(filters,
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: forkSyscall},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: deny},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: vforkSyscall},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: deny},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: uint32(unix.SYS_SETPGID)},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: deny},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: uint32(unix.SYS_SETSID)},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: deny},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jf: 1, K: uint32(unix.SYS_CLONE3)},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unsupported},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JEQ | unix.BPF_K, Jt: 1, K: uint32(unix.SYS_CLONE)},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ALLOW},
		unix.SockFilter{Code: unix.BPF_LD | unix.BPF_W | unix.BPF_ABS, K: 16},
		unix.SockFilter{Code: unix.BPF_JMP | unix.BPF_JSET | unix.BPF_K, Jt: 1, K: unix.CLONE_THREAD},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: deny},
		unix.SockFilter{Code: unix.BPF_RET | unix.BPF_K, K: unix.SECCOMP_RET_ALLOW},
	)
	return filters, nil
}
