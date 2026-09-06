//go:build darwin

package rerank

import (
	"errors"
	"os"
	"os/exec"
	"syscall"
	"testing"

	"golang.org/x/sys/unix"
)

func TestRerankerProcessHandleAfterIdentitySocketCloses(t *testing.T) {
	cmd := exec.Command("/bin/sleep", "30")
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	})
	fds, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		t.Fatal(err)
	}
	identity := os.NewFile(uintptr(fds[0]), "closed-peer")
	defer func() { _ = identity.Close() }()
	if err := unix.Close(fds[1]); err != nil {
		t.Fatal(err)
	}
	if _, err := peerAuditToken(identity); !errors.Is(err, unix.ENOTCONN) {
		t.Fatalf("peer audit token error = %v, want ENOTCONN", err)
	}
	pid := cmd.Process.Pid
	started := rerankerProcessStartIdentity(pid)
	if started == "" {
		t.Fatal("missing process start identity")
	}
	for _, wrong := range []string{"", started + "-stale"} {
		if _, err := openRerankerProcessHandle(pid, wrong, identity); err == nil {
			t.Fatal("accepted incorrect process generation")
		}
	}
	handle, err := openRerankerProcessHandle(pid, started, identity)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = handle.Close() }()
	stale := *handle
	stale.token[7]++
	if err := stale.Signal(syscall.SIGKILL); !errors.Is(err, os.ErrProcessDone) {
		t.Fatalf("stale pidversion signal = %v, want ErrProcessDone", err)
	}
	if err := syscall.Kill(pid, 0); err != nil {
		t.Fatalf("stale handle killed the live process: %v", err)
	}
	if err := handle.Signal(syscall.SIGKILL); err != nil {
		t.Fatalf("terminate through handle: %v", err)
	}
}
