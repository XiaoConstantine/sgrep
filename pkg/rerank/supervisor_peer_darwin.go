//go:build darwin

package rerank

import (
	"debug/macho"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"os"
	"runtime"
	"syscall"
	"time"
	"unsafe"

	"golang.org/x/sys/unix"
)

func rerankerProcessExecutableIdentity(pid int) (uint64, uint64, error) {
	type uniqueIdentifierInfo struct {
		UUID           [16]byte
		UniqueID       uint64
		ParentUniqueID uint64
		IDVersion      int32
		Reserved2      uint32
		Reserved3      uint64
		Reserved4      uint64
	}
	var identity uniqueIdentifierInfo
	const (
		procInfoCallPIDInfo         = 0x2
		procPIDUniqueIdentifierInfo = 17
	)
	//nolint:staticcheck // x/sys has no wrapper for Darwin's private proc_info flavors.
	result, _, errno := unix.Syscall6(
		unix.SYS_PROC_INFO,
		procInfoCallPIDInfo,
		uintptr(pid),
		procPIDUniqueIdentifierInfo,
		0,
		uintptr(unsafe.Pointer(&identity)),
		unsafe.Sizeof(identity),
	)
	if errno != 0 {
		return 0, 0, errno
	}
	if result != unsafe.Sizeof(identity) {
		return 0, 0, fmt.Errorf("unexpected Darwin executable identity size %d", result)
	}
	first := binary.LittleEndian.Uint64(identity.UUID[:8])
	second := binary.LittleEndian.Uint64(identity.UUID[8:])
	if first == 0 && second == 0 {
		return 0, 0, fmt.Errorf("darwin executable has no Mach-O UUID")
	}
	return first, second, nil
}

func configuredRerankerExecutableIdentity(path string) (uint64, uint64, error) {
	file, err := os.Open(path)
	if err != nil {
		return 0, 0, err
	}
	defer func() { _ = file.Close() }()
	return openedMachOExecutableIdentity(file)
}

func enableRerankerExecTrace() error {
	// PT_TRACE_ME stops this process at each following exec before the new
	// image can execute user code. The supervisor refreshes the guardian's
	// audit-token handle while the configured image is stopped.
	// Darwin applies this to the calling thread; runConstrainedReranker
	// locks the OS thread before arming and keeps it through exec.
	//nolint:staticcheck // x/sys exposes Darwin ptrace constants but no trace-me wrapper.
	_, _, errno := unix.Syscall6(unix.SYS_PTRACE, unix.PT_TRACE_ME, 0, 0, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func waitForStoppedRerankerExec(pid int, expectedDevice, expectedInode uint64, authorize func() error) (rerankerPIDState, func() error, bool, error) {
	deadline := time.Now().Add(supervisorSetupTimeout)
	if authorize != nil {
		if err := authorize(); err != nil {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			return rerankerPIDState{}, nil, false, err
		}
	}
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			_ = syscall.Kill(pid, syscall.SIGKILL)
			return rerankerPIDState{}, nil, false, fmt.Errorf("constrained reranker did not exec the configured binary")
		}
		status, reaped, err := waitForDarwinTraceEvent(pid, remaining)
		if err != nil {
			return rerankerPIDState{}, nil, reaped, err
		}
		if reaped || !status.Stopped() {
			return rerankerPIDState{}, nil, reaped, fmt.Errorf("constrained reranker exited before executing the configured binary")
		}
		state, inspectErr := inspectRerankerProcess(pid)
		if inspectErr == nil && state.ExecutableDevice == expectedDevice && state.ExecutableInode == expectedInode {
			return state, func() error {
				if err := detachDarwinTrace(pid); err != nil {
					return fmt.Errorf("detach constrained reranker exec trace: %w", err)
				}
				return nil
			}, false, nil
		}
		if err := continueDarwinTrace(pid); err != nil {
			return rerankerPIDState{}, nil, false, fmt.Errorf("continue intermediate constrained reranker exec: %w", err)
		}
	}
}

func waitForDarwinTraceEvent(pid int, timeout time.Duration) (syscall.WaitStatus, bool, error) {
	deadline := time.Now().Add(timeout)
	for {
		var status syscall.WaitStatus
		waited, err := syscall.Wait4(pid, &status, syscall.WUNTRACED|syscall.WNOHANG, nil)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if err != nil {
			return status, false, fmt.Errorf("wait for constrained reranker trace event: %w", err)
		}
		if waited == 0 {
			if time.Now().Before(deadline) {
				time.Sleep(5 * time.Millisecond)
				continue
			}
			// The immediately preceding non-reaping wait proved this PID is
			// still our direct child (possibly a zombie), so this numeric kill
			// cannot target a reused generation.
			_ = syscall.Kill(pid, syscall.SIGKILL)
			var reapedPID int
			for {
				reapedPID, err = syscall.Wait4(pid, &status, syscall.WUNTRACED, nil)
				if !errors.Is(err, syscall.EINTR) {
					break
				}
			}
			if err != nil {
				return status, false, fmt.Errorf("reap timed-out constrained reranker: %w", err)
			}
			if reapedPID != pid {
				return status, false, fmt.Errorf("reaped unexpected constrained reranker PID %d", reapedPID)
			}
			return status, status.Exited() || status.Signaled(), fmt.Errorf("timed out waiting for constrained reranker trace event")
		}
		if waited != pid {
			return status, false, fmt.Errorf("waited for unexpected constrained reranker PID %d", waited)
		}
		return status, status.Exited() || status.Signaled(), nil
	}
}

func continueDarwinTrace(pid int) error {
	return ptraceDarwin(unix.PT_CONTINUE, pid)
}

func detachDarwinTrace(pid int) error {
	// x/sys's PtraceDetach passes addr=0. Darwin treats that like a null PC
	// on detach and delivers the exec-stop SIGTRAP as a fatal trace/BPT trap.
	return ptraceDarwin(unix.PT_DETACH, pid)
}

func ptraceDarwin(request, pid int) error {
	//nolint:staticcheck // x/sys exposes attach/detach but no Darwin PT_CONTINUE wrapper.
	_, _, errno := unix.Syscall6(unix.SYS_PTRACE, uintptr(request), uintptr(pid), 1, 0, 0, 0)
	if errno != 0 {
		return errno
	}
	return nil
}

func openedMachOExecutableIdentity(file *os.File) (uint64, uint64, error) {
	if fat, err := macho.NewFatFile(file); err == nil {
		var wanted macho.Cpu
		switch runtime.GOARCH {
		case "amd64":
			wanted = macho.CpuAmd64
		case "arm64":
			wanted = macho.CpuArm64
		}
		for _, architecture := range fat.Arches {
			if architecture.Cpu == wanted {
				return machoExecutableUUID(architecture.File)
			}
		}
		return 0, 0, fmt.Errorf("Mach-O has no %s architecture", runtime.GOARCH)
	}
	thin, err := macho.NewFile(file)
	if err != nil {
		return 0, 0, err
	}
	return machoExecutableUUID(thin)
}

func openCurrentSupervisorExecutable(path string) (*os.File, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	openedDevice, openedInode, err := openedMachOExecutableIdentity(file)
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	runningDevice, runningInode, err := rerankerProcessExecutableIdentity(os.Getpid())
	if err != nil {
		_ = file.Close()
		return nil, err
	}
	if openedDevice != runningDevice || openedInode != runningInode {
		_ = file.Close()
		return nil, fmt.Errorf("executable path no longer names the running host image")
	}
	return file, nil
}

func machoExecutableUUID(file *macho.File) (uint64, uint64, error) {
	const loadCommandUUID = 0x1b
	for _, load := range file.Loads {
		data := load.Raw()
		if len(data) >= 24 && file.ByteOrder.Uint32(data[:4]) == loadCommandUUID {
			first := binary.LittleEndian.Uint64(data[8:16])
			second := binary.LittleEndian.Uint64(data[16:24])
			if first != 0 || second != 0 {
				return first, second, nil
			}
		}
	}
	return 0, 0, fmt.Errorf("Mach-O executable has no UUID")
}

func rerankerNativeProcessStartIdentity(pid int) string {
	process, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil || int(process.Proc.P_pid) != pid {
		return ""
	}
	started := process.Proc.P_starttime
	return fmt.Sprintf("darwin-start-time:%d:%d", started.Sec, started.Usec)
}

type rerankerProcessHandle struct {
	token [8]uint32
}

func openRerankerProcessHandle(pid int, started string, identity *os.File) (*rerankerProcessHandle, error) {
	token, err := peerAuditToken(identity)
	if err != nil {
		return nil, err
	}
	if int(token[5]) != pid || rerankerProcessStartIdentity(pid) != started {
		return nil, fmt.Errorf("reranker process generation changed while acquiring audit token")
	}
	return &rerankerProcessHandle{token: token}, nil
}

func (h *rerankerProcessHandle) encodeGuardianHandle() (string, error) {
	if h == nil {
		return "", os.ErrProcessDone
	}
	data := make([]byte, len(h.token)*4)
	for index, value := range h.token {
		binary.LittleEndian.PutUint32(data[index*4:], value)
	}
	return hex.EncodeToString(data), nil
}

func restoreRerankerProcessHandle(pid int, started, encoded string, identity *os.File) (*rerankerProcessHandle, error) {
	if encoded == "" {
		return openRerankerProcessHandle(pid, started, identity)
	}
	data, err := hex.DecodeString(encoded)
	if err != nil || len(data) != 8*4 {
		return nil, fmt.Errorf("invalid encoded reranker audit token")
	}
	var token [8]uint32
	for index := range token {
		token[index] = binary.LittleEndian.Uint32(data[index*4:])
	}
	if int(token[5]) != pid || rerankerProcessStartIdentity(pid) != started {
		return nil, fmt.Errorf("reranker process generation changed while restoring audit token")
	}
	return &rerankerProcessHandle{token: token}, nil
}

func (h *rerankerProcessHandle) Signal(signal os.Signal) error {
	if h == nil {
		return os.ErrProcessDone
	}
	sig, ok := signal.(syscall.Signal)
	if !ok {
		return fmt.Errorf("unsupported process signal %v", signal)
	}
	const procInfoCallSignalAuditToken = 0x11
	//nolint:staticcheck // x/sys has no generation-safe proc_signal_with_audittoken wrapper.
	_, _, errno := unix.Syscall6(
		unix.SYS_PROC_INFO,
		procInfoCallSignalAuditToken,
		0,
		uintptr(sig),
		0,
		uintptr(unsafe.Pointer(&h.token[0])),
		unsafe.Sizeof(h.token),
	)
	if errors.Is(errno, unix.ESRCH) {
		return os.ErrProcessDone
	}
	if errno != 0 {
		return errno
	}
	return nil
}

func (h *rerankerProcessHandle) Close() error {
	return nil
}

func peerAuditToken(identity *os.File) ([8]uint32, error) {
	var token [8]uint32
	if identity == nil {
		return token, fmt.Errorf("reranker identity socket is unavailable")
	}
	size := uint32(unsafe.Sizeof(token))
	//nolint:staticcheck // x/sys has no LOCAL_PEERTOKEN wrapper that returns audit_token_t.
	_, _, errno := unix.Syscall6(
		unix.SYS_GETSOCKOPT,
		identity.Fd(),
		unix.SOL_LOCAL,
		unix.LOCAL_PEERTOKEN,
		uintptr(unsafe.Pointer(&token[0])),
		uintptr(unsafe.Pointer(&size)),
		0,
	)
	if errno != 0 {
		return token, errno
	}
	if uintptr(size) != unsafe.Sizeof(token) {
		return token, fmt.Errorf("unexpected Darwin peer audit token size %d", size)
	}
	return token, nil
}

func supervisorPeerPID(conn *net.UnixConn) (int, error) {
	raw, err := conn.SyscallConn()
	if err != nil {
		return 0, err
	}
	var pid int
	var controlErr error
	if err := raw.Control(func(fd uintptr) {
		pid, controlErr = unix.GetsockoptInt(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERPID)
	}); err != nil {
		return 0, err
	}
	return pid, controlErr
}
