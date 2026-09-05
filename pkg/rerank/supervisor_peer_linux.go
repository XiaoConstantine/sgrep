//go:build linux

package rerank

import (
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

func rerankerNativeProcessStartIdentity(pid int) string {
	_, started, err := linuxRerankerProcessStat(pid)
	if err != nil {
		return ""
	}
	return started
}

func rerankerProcessExecutableIdentity(pid int) (uint64, uint64, error) {
	var stat unix.Stat_t
	if err := unix.Stat(filepath.Join("/proc", strconv.Itoa(pid), "exe"), &stat); err != nil {
		return 0, 0, err
	}
	return uint64(stat.Dev), stat.Ino, nil
}

func configuredRerankerExecutableIdentity(path string) (uint64, uint64, error) {
	var stat unix.Stat_t
	if err := unix.Stat(path, &stat); err != nil {
		return 0, 0, err
	}
	return uint64(stat.Dev), stat.Ino, nil
}

func enableRerankerExecTrace() error {
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

func openCurrentSupervisorExecutable(_ string) (*os.File, error) {
	// Opening /proc/self/exe binds registration to the image executing this
	// process, even if its original path is atomically replaced concurrently.
	return os.Open("/proc/self/exe")
}

type rerankerProcessHandle struct {
	fd int
}

func openRerankerProcessHandle(pid int, started string, _ *os.File) (*rerankerProcessHandle, error) {
	fd, err := unix.PidfdOpen(pid, 0)
	if err != nil {
		return nil, err
	}
	handle := &rerankerProcessHandle{fd: fd}
	if rerankerProcessStartIdentity(pid) != started {
		_ = handle.Close()
		return nil, fmt.Errorf("reranker process generation changed while acquiring pidfd")
	}
	return handle, nil
}

func (h *rerankerProcessHandle) encodeGuardianHandle() (string, error) {
	if h == nil || h.fd < 0 {
		return "", os.ErrProcessDone
	}
	// A pidfd cannot be serialized as data. The guardian reacquires its own
	// pidfd while the direct child is alive and unreaped.
	return "", nil
}

func restoreRerankerProcessHandle(pid int, started, _ string, identity *os.File) (*rerankerProcessHandle, error) {
	return openRerankerProcessHandle(pid, started, identity)
}

func (h *rerankerProcessHandle) Signal(signal os.Signal) error {
	if h == nil || h.fd < 0 {
		return os.ErrProcessDone
	}
	sig, ok := signal.(syscall.Signal)
	if !ok {
		return fmt.Errorf("unsupported process signal %v", signal)
	}
	err := unix.PidfdSendSignal(h.fd, unix.Signal(sig), nil, 0)
	if errors.Is(err, unix.ESRCH) {
		return os.ErrProcessDone
	}
	return err
}

func (h *rerankerProcessHandle) Close() error {
	if h == nil || h.fd < 0 {
		return nil
	}
	fd := h.fd
	h.fd = -1
	return unix.Close(fd)
}

func linuxRerankerProcessStat(pid int) (int, string, error) {
	data, err := os.ReadFile(filepath.Join("/proc", strconv.Itoa(pid), "stat"))
	if err != nil {
		return 0, "", err
	}
	endCommand := strings.LastIndexByte(string(data), ')')
	if endCommand < 0 {
		return 0, "", fmt.Errorf("invalid process stat")
	}
	fields := strings.Fields(string(data)[endCommand+1:])
	// Fields after the command start at proc field 3. Parent PID is field 4,
	// and starttime is field 22.
	if len(fields) <= 19 {
		return 0, "", fmt.Errorf("short process stat")
	}
	parent, err := strconv.Atoi(fields[1])
	if err != nil {
		return 0, "", err
	}
	return parent, "linux-start-ticks:" + fields[19], nil
}

func supervisorPeerPID(conn *net.UnixConn) (int, error) {
	raw, err := conn.SyscallConn()
	if err != nil {
		return 0, err
	}
	var pid int
	var controlErr error
	if err := raw.Control(func(fd uintptr) {
		var credentials *unix.Ucred
		credentials, controlErr = unix.GetsockoptUcred(int(fd), unix.SOL_SOCKET, unix.SO_PEERCRED)
		if controlErr == nil {
			pid = int(credentials.Pid)
		}
	}); err != nil {
		return 0, err
	}
	return pid, controlErr
}
