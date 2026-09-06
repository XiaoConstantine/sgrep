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

type uniqueIdentifierInfo struct {
	UUID           [16]byte
	UniqueID       uint64
	ParentUniqueID uint64
	IDVersion      int32
	Reserved2      uint32
	Reserved3      uint64
	Reserved4      uint64
}

func darwinProcessIdentity(pid int) (uniqueIdentifierInfo, error) {
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
		return identity, errno
	}
	if result != unsafe.Sizeof(identity) {
		return identity, fmt.Errorf("unexpected Darwin executable identity size %d", result)
	}
	return identity, nil
}

func rerankerProcessExecutableIdentity(pid int) (uint64, uint64, error) {
	identity, err := darwinProcessIdentity(pid)
	if err != nil {
		return 0, 0, err
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
	// macOS 26 delivers an uncaught exec-stop SIGTRAP as a fatal trace/BPT
	// trap unless wait4 consumes it immediately. Keep the identity socket
	// open across exec instead and poll for the new image like Linux.
	return nil
}

func waitForStoppedRerankerExec(pid int, expectedDevice, expectedInode uint64, authorize func() error) (rerankerPIDState, func() error, bool, error) {
	if authorize != nil {
		if err := authorize(); err != nil {
			return rerankerPIDState{}, nil, false, err
		}
	}
	state, err := waitForRerankerExec(pid, expectedDevice, expectedInode)
	return state, func() error { return nil }, false, err
}

func waitForRerankerExec(pid int, expectedDevice, expectedInode uint64) (rerankerPIDState, error) {
	deadline := time.Now().Add(supervisorSetupTimeout)
	var lastErr error
	for time.Now().Before(deadline) {
		state, err := inspectRerankerProcess(pid)
		if err == nil && state.ExecutableDevice == expectedDevice && state.ExecutableInode == expectedInode {
			return state, nil
		}
		if err != nil {
			lastErr = err
		}
		time.Sleep(10 * time.Millisecond)
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("process never executed the configured binary")
	}
	return rerankerPIDState{}, fmt.Errorf("identify constrained reranker PID %d: %w", pid, lastErr)
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
	if errors.Is(err, unix.ENOTCONN) {
		// The executed program may close inherited descriptors before we
		// observe its image. Darwin's proc_find_audit_token (used by signal)
		// checks only PID and pidversion; obtain both from kernel process
		// information rather than falling back to a racy kill(pid).
		info, infoErr := darwinProcessIdentity(pid)
		if infoErr != nil {
			return nil, infoErr
		}
		token = [8]uint32{5: uint32(pid), 7: uint32(info.IDVersion)}
	} else if err != nil {
		return nil, err
	}
	if int(token[5]) != pid || started == "" || rerankerProcessStartIdentity(pid) != started {
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
