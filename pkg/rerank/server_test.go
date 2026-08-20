package rerank

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

func init() {
	if os.Getenv("GO_WANT_FAKE_RERANKER_SERVER") != "1" || os.Getenv(supervisorModeEnv) == "1" {
		return
	}
	port := ""
	for index, argument := range os.Args {
		if argument == "--port" && index+1 < len(os.Args) {
			port = os.Args[index+1]
		}
	}
	if port == "" {
		os.Exit(2)
	}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	if err := http.ListenAndServe("127.0.0.1:"+port, mux); err != nil {
		os.Exit(2)
	}
	os.Exit(0)
}

func TestMain(m *testing.M) {
	if handled, err := RunSupervisorCommand(os.Args[1:]); handled {
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	os.Exit(m.Run())
}

func TestRerankerManagerStopRefusesUnownedLivePID(t *testing.T) {
	dir := t.TempDir()
	pid := os.Getpid()
	mgr := &RerankerManager{
		sgrepHome: dir,
		port:      59994,
		host:      "localhost",
		processCheck: func(got int) bool {
			return got == pid && false
		},
	}
	state := rerankerPIDState{PID: pid, Started: rerankerProcessStartIdentity(pid)}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}

	err := mgr.Stop()
	if err == nil || !strings.Contains(err.Error(), "refusing to stop") {
		t.Fatalf("Stop error = %v, want ownership refusal", err)
	}
	if !rerankerProcessAlive(pid) {
		t.Fatal("Stop signaled an unowned process")
	}
}

func TestRerankerManagerStartWorksForEmbeddedHost(t *testing.T) {
	home := t.TempDir()
	binDir := t.TempDir()
	llamaPath := filepath.Join(binDir, "llama-server")
	pathSpoof := filepath.Join(binDir, "sgrep")
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(executable, llamaPath); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(pathSpoof, []byte("#!/bin/sh\nexit 77\n"), 0755); err != nil {
		t.Fatal(err)
	}
	modelPath := filepath.Join(home, "model.gguf")
	if err := os.WriteFile(modelPath, []byte("test model"), 0600); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := listener.Addr().(*net.TCPAddr).Port
	_ = listener.Close()
	t.Setenv("PATH", binDir+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("SGREP_RERANK_MODEL", modelPath)
	t.Setenv("GO_WANT_FAKE_RERANKER_SERVER", "1")

	mgr := &RerankerManager{
		sgrepHome: home,
		port:      port,
		host:      "localhost",
	}
	if got, err := mgr.findSupervisorExecutable(); err != nil || got.Path != executable {
		t.Fatalf("findSupervisorExecutable() = %+v, %v; want registered host %q", got, err, executable)
	}
	t.Cleanup(func() { _ = mgr.Stop() })
	if err := mgr.Start(); err != nil {
		t.Fatal(err)
	}
	state, err := mgr.readPIDState()
	if err != nil {
		t.Fatal(err)
	}
	if state.SupervisorPID <= 0 || state.SupervisorPID == os.Getpid() {
		t.Fatalf("supervisor PID = %d, want a separate re-exec process", state.SupervisorPID)
	}
	if !mgr.IsRunning() {
		t.Fatal("embedded host did not retain a manageable reranker")
	}
	if err := mgr.Stop(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(mgr.pidPath()); !os.IsNotExist(err) {
		t.Fatalf("PID state still exists after embedded-host Stop: %v", err)
	}
}

func TestRerankerManagerSerializesLifecycleOperations(t *testing.T) {
	home := t.TempDir()
	first := &RerankerManager{sgrepHome: home}
	second := &RerankerManager{sgrepHome: home}
	firstEntered := make(chan struct{})
	secondEntered := make(chan struct{})
	releaseFirst := make(chan struct{})
	firstDone := make(chan error, 1)
	secondDone := make(chan error, 1)
	defer func() {
		select {
		case <-releaseFirst:
		default:
			close(releaseFirst)
		}
	}()

	go func() {
		firstDone <- first.withLifecycleLock("test", func(*os.File) error {
			close(firstEntered)
			<-releaseFirst
			return nil
		})
	}()
	select {
	case <-firstEntered:
	case <-time.After(2 * time.Second):
		t.Fatal("first lifecycle operation did not acquire the lock")
	}
	go func() {
		secondDone <- second.withLifecycleLock("test", func(*os.File) error {
			close(secondEntered)
			return nil
		})
	}()
	select {
	case <-secondEntered:
		t.Fatal("second lifecycle operation entered while the first held the lock")
	case <-time.After(100 * time.Millisecond):
	}
	close(releaseFirst)
	select {
	case err := <-firstDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("first lifecycle operation did not release the lock")
	}
	select {
	case err := <-secondDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second lifecycle operation did not acquire the released lock")
	}
}

func TestRerankerSupervisorRetainsLifecycleLockUntilPublication(t *testing.T) {
	home := t.TempDir()
	lockPath := filepath.Join(home, ".reranker.lock")
	launcherFD, err := unix.Open(lockPath, unix.O_CREAT|unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0600)
	if err != nil {
		t.Fatal(err)
	}
	if err := unix.Flock(launcherFD, unix.LOCK_EX); err != nil {
		_ = unix.Close(launcherFD)
		t.Fatal(err)
	}
	supervisorFD, err := unix.Dup(launcherFD)
	if err != nil {
		_ = unix.Close(launcherFD)
		t.Fatal(err)
	}
	supervisorLock := os.NewFile(uintptr(supervisorFD), "inherited-reranker-lifecycle-lock")
	// Model abrupt launcher death: close its descriptor without LOCK_UN. The
	// duplicate inherited by the supervisor must retain the open-file lock.
	if err := unix.Close(launcherFD); err != nil {
		t.Fatal(err)
	}

	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59978
	publicationReached := make(chan struct{})
	releasePublication := make(chan struct{})
	done := make(chan error, 1)
	go func() {
		done <- runRerankerSupervisor(supervisorConfig{
			Home:          home,
			Control:       control,
			Token:         token,
			Port:          port,
			Executable:    "/bin/sleep",
			Args:          []string{"30"},
			Environment:   os.Environ(),
			Stdout:        io.Discard,
			Stderr:        io.Discard,
			LifecycleLock: supervisorLock,
			BeforePublish: func() {
				close(publicationReached)
				<-releasePublication
			},
		})
	}()
	select {
	case <-publicationReached:
	case <-time.After(2 * time.Second):
		close(releasePublication)
		t.Fatal("supervisor did not reach publication")
	}
	contenderFD, err := unix.Open(lockPath, unix.O_RDWR|unix.O_CLOEXEC|unix.O_NOFOLLOW, 0600)
	if err != nil {
		close(releasePublication)
		t.Fatal(err)
	}
	defer func() { _ = unix.Close(contenderFD) }()
	if err := unix.Flock(contenderFD, unix.LOCK_EX|unix.LOCK_NB); !errors.Is(err, syscall.EWOULDBLOCK) {
		close(releasePublication)
		t.Fatalf("competing launcher acquired lock before publication: %v", err)
	}
	close(releasePublication)

	mgr := &RerankerManager{sgrepHome: home, port: port}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), done)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = requestRerankerSupervisor(state, "stop") })
	deadline := time.Now().Add(2 * time.Second)
	for {
		err = unix.Flock(contenderFD, unix.LOCK_EX|unix.LOCK_NB)
		if err == nil {
			break
		}
		if !errors.Is(err, syscall.EWOULDBLOCK) || time.Now().After(deadline) {
			t.Fatalf("lifecycle lock was not released after publication: %v", err)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := requestRerankerSupervisor(state, "stop"); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor did not stop")
	}
}

func TestOpenCurrentSupervisorExecutableUsesRunningImage(t *testing.T) {
	file, err := openCurrentSupervisorExecutable("/bin/sleep")
	if runtime.GOOS == "darwin" {
		if err == nil {
			_ = file.Close()
			t.Fatal("Darwin accepted a path naming a different executable image")
		}
		return
	}
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	info, err := file.Stat()
	if err != nil {
		t.Fatal(err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		t.Fatal("opened supervisor image has no native file identity")
	}
	device, inode, err := rerankerProcessExecutableIdentity(os.Getpid())
	if err != nil {
		t.Fatal(err)
	}
	if uint64(stat.Dev) != device || uint64(stat.Ino) != inode {
		t.Fatal("Linux registration did not open /proc/self/exe")
	}
}

func TestOpenCurrentSupervisorExecutableAcceptsCurrentImage(t *testing.T) {
	file, err := openCurrentSupervisorExecutable(mustExecutable(t))
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = file.Close() }()
	if _, err := newSupervisorExecutableRegistration(mustExecutable(t), file); err != nil {
		t.Fatal(err)
	}
}

func TestRerankerManagerCleansReusedPIDGenerations(t *testing.T) {
	operations := map[string]func(*RerankerManager) error{
		"start cleanup": (*RerankerManager).cleanStalePID,
		"stop":          (*RerankerManager).Stop,
	}
	for name, operation := range operations {
		t.Run(name, func(t *testing.T) {
			home := t.TempDir()
			token, err := newRerankerInstanceToken()
			if err != nil {
				t.Fatal(err)
			}
			control, err := createRerankerControlPath(token)
			if err != nil {
				t.Fatal(err)
			}
			listener, _, err := listenSupervisorSocket(control)
			if err != nil {
				t.Fatal(err)
			}
			if err := listener.Close(); err != nil {
				t.Fatal(err)
			}
			pid := os.Getpid()
			state := rerankerPIDState{
				Protocol:          supervisorProtocolVersion,
				PID:               pid,
				Started:           rerankerProcessStartIdentity(pid) + "-reused",
				Instance:          token,
				Control:           control,
				Port:              59977,
				SupervisorPID:     pid,
				SupervisorStarted: rerankerProcessStartIdentity(pid) + "-reused",
			}
			mgr := &RerankerManager{sgrepHome: home, port: state.Port, host: "localhost"}
			if err := mgr.writePIDState(state); err != nil {
				t.Fatal(err)
			}
			if err := operation(mgr); err != nil {
				t.Fatal(err)
			}
			if _, err := os.Stat(mgr.pidPath()); !os.IsNotExist(err) {
				t.Fatalf("stale PID state remains: %v", err)
			}
			if _, err := os.Stat(filepath.Dir(control)); !os.IsNotExist(err) {
				t.Fatalf("stale control directory remains: %v", err)
			}
			if !rerankerProcessAlive(pid) {
				t.Fatal("generation cleanup signaled the reused process")
			}
		})
	}
}

func TestRerankerManagerDoesNotDiscoverSupervisorOnPATH(t *testing.T) {
	pathDirectory := t.TempDir()
	pathSpoof := filepath.Join(pathDirectory, "sgrep")
	if err := os.WriteFile(pathSpoof, []byte("#!/bin/sh\nexit 77\n"), 0755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", pathDirectory+string(os.PathListSeparator)+os.Getenv("PATH"))
	mgr := &RerankerManager{}
	got, err := mgr.findSupervisorExecutable()
	if err != nil {
		t.Fatal(err)
	}
	if got.Path == pathSpoof {
		t.Fatalf("findSupervisorExecutable() selected PATH spoof %q", pathSpoof)
	}
	if got.Path != absoluteExecutablePath(mustExecutable(t)) {
		t.Fatalf("findSupervisorExecutable() = %+v, want current registered executable", got)
	}
}

func TestRerankerManagerIsRunningRequiresOwnedPID(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	dir := t.TempDir()
	pid := os.Getpid()
	state := strconv.Itoa(pid) + "\ntest-process-start\n"
	if err := os.WriteFile(filepath.Join(dir, "reranker.pid"), []byte(state), 0644); err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{
		sgrepHome: dir,
		port:      testServerPort(t, srv.URL),
		host:      "localhost",
		processCheck: func(got int) bool {
			return got == pid
		},
	}
	if !mgr.IsRunning() {
		t.Fatal("owned healthy reranker was not reported running")
	}
	mgr.processCheck = func(int) bool { return false }
	if mgr.IsRunning() {
		t.Fatal("unowned healthy service was reported as the managed reranker")
	}
}

func TestRerankerManagerReadPIDState(t *testing.T) {
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir}
	want := rerankerPIDState{PID: 1234, PGID: 1234, Started: "process start identity", Executable: "/bin/llama-server", Instance: "token", Control: "/tmp/control.sock", Port: 8081}
	if err := mgr.writePIDState(want); err != nil {
		t.Fatal(err)
	}
	pid, err := mgr.readPID()
	if err != nil || pid != 1234 {
		t.Fatalf("readPID() = %d, %v", pid, err)
	}
	identity, err := mgr.readProcessStartIdentity()
	if err != nil || identity != "process start identity" {
		t.Fatalf("readProcessStartIdentity() = %q, %v", identity, err)
	}
}

func TestRerankerManagerRejectsLegacyPIDState(t *testing.T) {
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: 8081}
	if err := os.WriteFile(mgr.pidPath(), []byte(strconv.Itoa(os.Getpid())), 0600); err != nil {
		t.Fatal(err)
	}
	if mgr.ownsProcess(os.Getpid()) {
		t.Fatal("legacy PID-only state was accepted as proof of ownership")
	}
}

func TestRerankerManagerRejectsUnrelatedExecutableWithMatchingArgs(t *testing.T) {
	dir := t.TempDir()
	script := filepath.Join(dir, "definitely-not-llama")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nwhile :; do sleep 1; done\n"), 0755); err != nil {
		t.Fatal(err)
	}
	const port = 59993
	const instance = "adversarial-instance-token"
	cmd := exec.Command(script, "--port", strconv.Itoa(port), "--pooling", "rank")
	cmd.Env = append(os.Environ(), rerankerInstanceEnv+"="+instance)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL)
		_, _ = cmd.Process.Wait()
	}()

	var current rerankerPIDState
	var err error
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		current, err = inspectRerankerProcess(cmd.Process.Pid)
		if err == nil {
			break
		}
	}
	if err != nil {
		t.Fatal(err)
	}
	current.Executable = filepath.Join(dir, "llama-server")
	current.Instance = instance
	current.Port = port
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	if err := mgr.writePIDState(current); err != nil {
		t.Fatal(err)
	}
	if mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("unrelated executable was classified as the managed reranker")
	}
}

func TestRerankerManagerRejectsStartIdentityMismatch(t *testing.T) {
	cmd, current := startRerankerHelperProcess(t, false)
	current.Started += "-different-generation"
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: current.Port}
	if err := mgr.writePIDState(current); err != nil {
		t.Fatal(err)
	}
	if mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("mismatched process generation was accepted")
	}
}

func TestRerankerManagerRejectsLookalikeArguments(t *testing.T) {
	const port = 59990
	tests := map[string][]string{
		"port":    {"--port", strconv.Itoa(port) + "0", "--pooling", "rank"},
		"pooling": {"--port", strconv.Itoa(port), "--pooling", "ranker"},
	}
	for name, args := range tests {
		t.Run(name, func(t *testing.T) {
			cmd, state := startRerankerHelperProcessWith(t, os.Args[0], false, port, args)
			dir := t.TempDir()
			mgr := &RerankerManager{sgrepHome: dir, port: port}
			if err := mgr.writePIDState(state); err != nil {
				t.Fatal(err)
			}
			if mgr.ownsProcess(cmd.Process.Pid) {
				t.Fatalf("lookalike %s argument was accepted", name)
			}
		})
	}
}

func TestRerankerManagerOwnsProductionProcessIdentity(t *testing.T) {
	cmd, state := startRerankerHelperProcess(t, false)
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: state.Port, host: "localhost"}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if !mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("process with the persisted executable, generation, arguments, and instance token was rejected")
	}
}

func TestRerankerManagerStopEscalatesOnlyOwnedProcess(t *testing.T) {
	state, supervisorDone := startTestRerankerSupervisor(t, true)
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: state.Port, host: "localhost"}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if err := mgr.Stop(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(mgr.pidPath()); !os.IsNotExist(err) {
		t.Fatalf("PID state still exists after Stop: %v", err)
	}
	select {
	case err := <-supervisorDone:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor survived owned-process SIGTERM/SIGKILL escalation")
	}
}

func TestRerankerManagerStopDoesNotSignalUnvalidatedGroupMember(t *testing.T) {
	state, _ := startTestRerankerSupervisor(t, true)
	other := exec.Command("/bin/sleep", "30")
	other.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: state.PGID}
	if err := other.Start(); err != nil {
		t.Fatal(err)
	}
	otherDone := make(chan error, 1)
	go func() { otherDone <- other.Wait() }()
	t.Cleanup(func() {
		_ = other.Process.Kill()
		select {
		case <-otherDone:
		case <-time.After(time.Second):
		}
	})

	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: state.Port, host: "localhost"}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if err := mgr.Stop(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-otherDone:
		t.Fatalf("Stop signaled an unvalidated process-group member: %v", err)
	case <-time.After(200 * time.Millisecond):
	}
}

func TestRerankerSupervisorSurvivesDisconnectedClient(t *testing.T) {
	state, supervisorDone := startTestRerankerSupervisor(t, false)
	conn, err := net.Dial("unix", state.Control)
	if err != nil {
		t.Fatal(err)
	}
	_ = conn.Close()
	time.Sleep(50 * time.Millisecond)

	response, err := requestRerankerSupervisor(state, "status")
	if err != nil {
		t.Fatalf("supervisor stopped serving after a disconnected client: %v", err)
	}
	if !response.Running || response.PID != state.PID {
		t.Fatalf("status after disconnected client = %+v, want running PID %d", response, state.PID)
	}
	select {
	case err := <-supervisorDone:
		t.Fatalf("supervisor exited after disconnected client: %v", err)
	default:
	}
}

func TestRerankerSupervisorSilentClientsDoNotBlockControl(t *testing.T) {
	state, _ := startTestRerankerSupervisor(t, false)
	directory, err := os.Stat(filepath.Dir(state.Control))
	if err != nil || directory.Mode().Perm() != 0700 {
		t.Fatalf("control directory mode = %v, %v; want 0700", directory, err)
	}
	socket, err := os.Stat(state.Control)
	if err != nil || socket.Mode().Perm() != 0600 {
		t.Fatalf("control socket mode = %v, %v; want 0600", socket, err)
	}
	var silent []net.Conn
	for range 2 {
		conn, err := net.Dial("unix", state.Control)
		if err != nil {
			t.Fatal(err)
		}
		silent = append(silent, conn)
	}
	defer func() {
		for _, conn := range silent {
			_ = conn.Close()
		}
	}()

	started := time.Now()
	response, err := requestRerankerSupervisor(state, "status")
	if err != nil {
		t.Fatalf("authenticated status was blocked by silent clients: %v", err)
	}
	if elapsed := time.Since(started); elapsed >= supervisorReadTimeout {
		t.Fatalf("authenticated status took %v with silent clients", elapsed)
	}
	if !response.Running || response.PID != state.PID {
		t.Fatalf("status with silent clients = %+v", response)
	}
}

func TestRerankerManagerRejectsReplacementControlSocket(t *testing.T) {
	cmd, state := startRerankerHelperProcess(t, false)
	control, err := createRerankerControlPath(state.Instance)
	if err != nil {
		t.Fatal(err)
	}
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: control, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = listener.Close()
		cleanupAbandonedControlPath(control, state.Instance)
	}()
	state.Control = control
	state.SupervisorPID = os.Getpid()
	state.SupervisorStarted = rerankerProcessStartIdentity(os.Getpid())
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: state.Port, host: "localhost"}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}

	received := make(chan supervisorRequest, 1)
	go func() {
		conn, acceptErr := listener.AcceptUnix()
		if acceptErr != nil {
			return
		}
		defer func() { _ = conn.Close() }()
		var request supervisorRequest
		if json.NewDecoder(conn).Decode(&request) == nil {
			received <- request
			_ = json.NewEncoder(conn).Encode(supervisorResponse{PID: state.PID, Running: false})
		}
	}()

	if err := mgr.Stop(); err == nil || !strings.Contains(err.Error(), "authentication failed") {
		t.Fatalf("Stop error = %v, want forged-response rejection", err)
	}
	select {
	case request := <-received:
		if request.Nonce == "" || request.Proof == "" {
			t.Fatalf("replacement received malformed challenge: %+v", request)
		}
	case <-time.After(time.Second):
		t.Fatal("replacement control server received no challenge")
	}
	if _, err := os.Stat(mgr.pidPath()); err != nil {
		t.Fatalf("Stop removed state after forged response: %v", err)
	}
	if !rerankerProcessAlive(cmd.Process.Pid) {
		t.Fatal("Stop terminated the managed process after forged response")
	}
}

func TestSupervisorSocketCleanupPreservesReplacement(t *testing.T) {
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	original, originalInfo, err := listenSupervisorSocket(control)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(control); err != nil {
		t.Fatal(err)
	}
	replacement, err := net.ListenUnix("unix", &net.UnixAddr{Name: control, Net: "unix"})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		_ = replacement.Close()
		_ = os.Remove(control)
		_ = os.Remove(filepath.Dir(control))
	}()

	cleanupSupervisorSocket(original, control, originalInfo)
	info, err := os.Lstat(control)
	if err != nil || info.Mode()&os.ModeSocket == 0 {
		t.Fatalf("cleanup removed replacement socket: %v", err)
	}
}

func TestRerankerSupervisorUnadoptedLaunchCleansUp(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59987
	done := make(chan error, 1)
	go func() {
		done <- runRerankerSupervisor(supervisorConfig{
			Home:            dir,
			Control:         control,
			Token:           token,
			Port:            port,
			Executable:      "/bin/sleep",
			Args:            []string{"30"},
			Environment:     os.Environ(),
			Stdout:          io.Discard,
			Stderr:          io.Discard,
			AdoptionTimeout: 500 * time.Millisecond,
		})
	}()
	mgr := &RerankerManager{sgrepHome: dir, port: port}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), done)
	if err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("unadopted supervisor did not terminate")
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("unadopted supervisor left its child alive")
	}
	if _, err := os.Stat(mgr.pidPath()); !os.IsNotExist(err) {
		t.Fatalf("unadopted supervisor left PID state: %v", err)
	}
}

func TestRerankerSupervisorUncommittedAdoptionCleansUp(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59979
	done := make(chan error, 1)
	go func() {
		done <- runRerankerSupervisor(supervisorConfig{
			Home:            dir,
			Control:         control,
			Token:           token,
			Port:            port,
			Executable:      "/bin/sleep",
			Args:            []string{"30"},
			Environment:     os.Environ(),
			Stdout:          io.Discard,
			Stderr:          io.Discard,
			AdoptionTimeout: time.Second,
			AdoptionLease:   300 * time.Millisecond,
		})
	}()
	mgr := &RerankerManager{sgrepHome: dir, port: port}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("uncommitted adoption disabled supervisor cleanup")
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("uncommitted adoption left reranker running")
	}
}

func TestRerankerManagerStartAdoptsPublishedSupervisor(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	port := testServerPort(t, srv.URL)
	ready := filepath.Join(dir, "ready")
	supervisorDone := make(chan error, 1)
	go func() {
		supervisorDone <- runRerankerSupervisor(supervisorConfig{
			Home:       dir,
			Control:    control,
			Token:      token,
			Port:       port,
			Executable: os.Args[0],
			Args: []string{
				"-test.run=TestRerankerHelperProcess", "--",
				"--port", strconv.Itoa(port), "--pooling", "rank",
			},
			Environment: append(os.Environ(),
				"GO_WANT_RERANKER_HELPER=1",
				"GO_RERANKER_HELPER_READY="+ready,
			),
			Stdout:          io.Discard,
			Stderr:          io.Discard,
			AdoptionTimeout: 400 * time.Millisecond,
		})
	}()
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), supervisorDone)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = requestRerankerSupervisor(state, "stop") })

	if err := mgr.Start(); err != nil {
		t.Fatalf("Start did not adopt published supervisor: %v", err)
	}
	time.Sleep(600 * time.Millisecond)
	response, err := requestRerankerSupervisor(state, "status")
	if err != nil || !response.Running || response.PID != state.PID {
		t.Fatalf("adopted supervisor after original deadline = %+v, %v", response, err)
	}
	select {
	case err := <-supervisorDone:
		t.Fatalf("adopted supervisor exited at original deadline: %v", err)
	default:
	}
}

func TestRerankerManagerFailedAdoptionStopsSupervisor(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59982
	ready := filepath.Join(dir, "ready")
	supervisorDone := make(chan error, 1)
	go func() {
		supervisorDone <- runRerankerSupervisor(supervisorConfig{
			Home:       dir,
			Control:    control,
			Token:      token,
			Port:       port,
			Executable: os.Args[0],
			Args: []string{
				"-test.run=TestRerankerHelperProcess", "--",
				"--port", strconv.Itoa(port), "--pooling", "rank",
			},
			Environment: append(os.Environ(),
				"GO_WANT_RERANKER_HELPER=1",
				"GO_RERANKER_HELPER_READY="+ready,
			),
			Stdout:          io.Discard,
			Stderr:          io.Discard,
			AdoptionTimeout: 2 * time.Second,
		})
	}()
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost", startupTimeout: 200 * time.Millisecond}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), supervisorDone)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = requestRerankerSupervisor(state, "stop") })

	adopted, err := mgr.adoptExistingSupervisor()
	if !adopted || err == nil {
		t.Fatalf("adoptExistingSupervisor() = %v, %v; want adopted readiness failure", adopted, err)
	}
	select {
	case <-supervisorDone:
	case <-time.After(2 * time.Second):
		t.Fatal("failed adoption left supervisor running")
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("failed adoption left reranker running")
	}
	if _, err := os.Stat(mgr.pidPath()); !os.IsNotExist(err) {
		t.Fatalf("failed adoption left PID state: %v", err)
	}
}

func TestRerankerSupervisorSIGKILLTerminatesEscapedRoot(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59981
	ready := filepath.Join(dir, "ready")
	t.Setenv("GO_WANT_RERANKER_HELPER", "1")
	t.Setenv("GO_RERANKER_HELPER_READY", ready)
	t.Setenv("GO_RERANKER_HELPER_IGNORE_TERM", "true")
	if runtime.GOOS == "darwin" {
		t.Setenv("GO_RERANKER_HELPER_SETPGID_PARENT", "true")
		t.Setenv("GO_RERANKER_HELPER_CLOSE_IDENTITY", "true")
	}
	t.Setenv("GO_RERANKER_HELPER_SETSID", "true")
	t.Setenv("GO_RERANKER_HELPER_REQUIRE_SETSID_DENIAL", "true")
	supervisor, done, err := launchRerankerSupervisor(currentSupervisorRegistration(t), dir, control, token, port, os.Args[0], []string{
		"-test.run=TestRerankerHelperProcess", "--",
		"--port", strconv.Itoa(port), "--pooling", "rank",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, supervisor.Process.Pid, done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = supervisor.Process.Kill()
		mgr.removePIDFileIfOwned(state)
		cleanupAbandonedControlPath(control, token)
	})
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("adversarial reranker did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	pgid, err := syscall.Getpgid(state.PID)
	if err != nil {
		t.Fatal(err)
	}
	if runtime.GOOS == "darwin" && pgid == state.PID {
		t.Fatalf("reranker did not exercise the adversarial process-group escape: PGID %d", pgid)
	}
	if runtime.GOOS != "darwin" && pgid != state.PID {
		t.Fatalf("Linux reranker escaped its seccomp-protected process group: PGID %d", pgid)
	}
	if session, err := unix.Getsid(state.PID); err != nil || session == state.PID {
		t.Fatalf("reranker escaped into a new session: SID %d, %v", session, err)
	}

	if err := supervisor.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor did not exit after SIGKILL")
	}
	deadline = time.Now().Add(2 * time.Second)
	for rerankerProcessAlive(state.PID) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("supervisor SIGKILL orphaned the escaped reranker root")
	}
}

func TestRerankerSupervisorSIGKILLTerminatesForkConfinedChild(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59986
	descendantPIDPath := filepath.Join(dir, "descendant.pid")
	deniedPath := filepath.Join(dir, "fork-denied")
	t.Setenv("GO_WANT_RERANKER_DESCENDANT_HELPER", "1")
	t.Setenv("GO_RERANKER_DESCENDANT_PID", descendantPIDPath)
	t.Setenv("GO_RERANKER_DESCENDANT_DENIED", deniedPath)
	t.Setenv("GO_RERANKER_DESCENDANT_SETSID", "true")
	supervisor, done, err := launchRerankerSupervisor(currentSupervisorRegistration(t), dir, control, token, port, os.Args[0], []string{
		"-test.run=TestRerankerDescendantHelper", "--",
		"--port", strconv.Itoa(port), "--pooling", "rank",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{sgrepHome: dir, port: port}
	state, err := waitForPublishedRerankerState(mgr, control, token, supervisor.Process.Pid, done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(deniedPath); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("reranker process creation was not denied")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(descendantPIDPath); !os.IsNotExist(err) {
		t.Fatalf("fork-confined reranker created a descendant: %v", err)
	}
	t.Cleanup(func() {
		mgr.removePIDFileIfOwned(state)
		cleanupAbandonedControlPath(control, token)
	})

	if err := supervisor.Process.Kill(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor did not exit after SIGKILL")
	}
	deadline = time.Now().Add(2 * time.Second)
	for rerankerProcessAlive(state.PID) && time.Now().Before(deadline) {
		time.Sleep(10 * time.Millisecond)
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("supervisor SIGKILL orphaned its retained child")
	}
}

func TestRerankerSupervisorStopPreventsDetachedDescendants(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59985
	descendantPIDPath := filepath.Join(dir, "descendant.pid")
	deniedPath := filepath.Join(dir, "fork-denied")
	t.Setenv("GO_WANT_RERANKER_DESCENDANT_HELPER", "1")
	t.Setenv("GO_RERANKER_DESCENDANT_PID", descendantPIDPath)
	t.Setenv("GO_RERANKER_DESCENDANT_DENIED", deniedPath)
	t.Setenv("GO_RERANKER_DESCENDANT_SETSID", "true")
	supervisor, done, err := launchRerankerSupervisor(currentSupervisorRegistration(t), dir, control, token, port, os.Args[0], []string{
		"-test.run=TestRerankerDescendantHelper", "--",
		"--port", strconv.Itoa(port), "--pooling", "rank",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, supervisor.Process.Pid, done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = requestRerankerSupervisor(state, "stop") })

	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(deniedPath); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("reranker process creation was not denied")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(descendantPIDPath); !os.IsNotExist(err) {
		t.Fatalf("fork-confined reranker created a descendant: %v", err)
	}

	if err := mgr.Stop(); err != nil {
		t.Fatal(err)
	}
	select {
	case err := <-done:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor did not exit after Stop")
	}
}

func TestRerankerSupervisorGuardianFailureTerminatesChild(t *testing.T) {
	dir := t.TempDir()
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59984
	ready := filepath.Join(dir, "ready")
	t.Setenv("GO_WANT_RERANKER_HELPER", "1")
	t.Setenv("GO_RERANKER_HELPER_READY", ready)
	t.Setenv("GO_RERANKER_HELPER_IGNORE_TERM", "true")
	supervisor, done, err := launchRerankerSupervisor(currentSupervisorRegistration(t), dir, control, token, port, os.Args[0], []string{
		"-test.run=TestRerankerHelperProcess", "--",
		"--port", strconv.Itoa(port), "--pooling", "rank",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, supervisor.Process.Pid, done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	if state.PGID != state.PID || state.GuardianPID <= 1 || state.GuardianPID == state.PID || state.GuardianPID == state.SupervisorPID {
		t.Fatalf("protected process group = %d, child = %d, guardian = %d, supervisor = %d", state.PGID, state.PID, state.GuardianPID, state.SupervisorPID)
	}
	if guardianGroup, err := syscall.Getpgid(state.GuardianPID); err != nil || guardianGroup != state.PGID {
		t.Fatalf("guardian process group = %d, %v; want %d", guardianGroup, err, state.PGID)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("supervised child did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Cleanup(func() {
		_ = supervisor.Process.Kill()
		mgr.removePIDFileIfOwned(state)
		cleanupAbandonedControlPath(control, token)
	})

	guardian, err := os.FindProcess(state.GuardianPID)
	if err != nil {
		t.Fatal(err)
	}
	if err := guardian.Kill(); err != nil {
		t.Fatal(err)
	}
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("supervisor did not exit after guardian failure")
	}
	deadline = time.Now().Add(2 * time.Second)
	for rerankerProcessAlive(state.PID) && time.Now().Before(deadline) {
		time.Sleep(20 * time.Millisecond)
	}
	if rerankerProcessAlive(state.PID) {
		t.Fatal("guardian failure left the retained child alive")
	}
}

func TestRerankerGuardianStableHandleSurvivesProcessGroupEscape(t *testing.T) {
	identities, err := newRerankerIdentitySockets()
	if err != nil {
		t.Fatal(err)
	}
	defer identities.Close()
	gateChild, gateParent, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = gateParent.Close() }()
	defer func() { _ = gateChild.Close() }()
	escaped := filepath.Join(t.TempDir(), "escaped")
	cmd := exec.Command(os.Args[0], "-test.run=TestRerankerGroupEscapeHelper")
	cmd.Env = append(os.Environ(),
		"GO_WANT_RERANKER_GROUP_ESCAPE_HELPER=1",
		"GO_RERANKER_ESCAPE_TARGET_GROUP="+strconv.Itoa(syscall.Getpgrp()),
		"GO_RERANKER_ESCAPE_READY="+escaped,
	)
	cmd.ExtraFiles = []*os.File{identities.root, gateChild}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	cmd.Stdout = io.Discard
	cmd.Stderr = io.Discard
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	_ = identities.root.Close()
	identities.root = nil
	_ = gateChild.Close()
	childDone := make(chan error, 1)
	go func() { childDone <- cmd.Wait() }()
	childWaited := false
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		if childWaited {
			return
		}
		select {
		case <-childDone:
		case <-time.After(time.Second):
		}
	})
	if err := waitForRunnerReady(identities.supervisor); err != nil {
		t.Fatal(err)
	}

	guardian, err := launchRerankerGuardian(io.Discard, cmd.Process.Pid, identities.guardian)
	if err != nil {
		t.Fatal(err)
	}
	_ = identities.guardian.Close()
	identities.guardian = nil
	if err := guardian.waitReady(); err != nil {
		t.Fatal(err)
	}
	started := rerankerProcessStartIdentity(cmd.Process.Pid)
	if started == "" {
		select {
		case childErr := <-childDone:
			childWaited = true
			t.Fatalf("escape helper exited before registration: %v", childErr)
		default:
		}
		t.Fatal("could not identify escape helper generation")
	}
	probe, err := openRerankerProcessHandle(cmd.Process.Pid, started, identities.supervisor)
	if err != nil {
		t.Fatalf("test identity socket cannot acquire root handle: %v", err)
	}
	defer func() { _ = probe.Close() }()
	if err := guardian.registerRoot(cmd.Process.Pid, started); err != nil {
		select {
		case <-guardian.done:
		case <-time.After(time.Second):
		}
		t.Fatal(err)
	}
	if err := guardian.refreshRoot(cmd.Process.Pid, started, probe); err != nil {
		t.Fatal(err)
	}
	_ = probe.Close()
	if _, err := gateParent.Write([]byte{1}); err != nil {
		t.Fatal(err)
	}

	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(escaped); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("helper did not escape its protected process group")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if pgid, err := syscall.Getpgid(cmd.Process.Pid); err != nil || pgid == cmd.Process.Pid {
		t.Fatalf("escape helper PGID = %d, %v; want a different group", pgid, err)
	}

	_ = guardian.lifetime.Close()
	select {
	case <-guardian.done:
	case <-time.After(2 * time.Second):
		t.Fatal("guardian did not react to supervisor lifetime closure")
	}
	select {
	case <-childDone:
		childWaited = true
	case <-time.After(2 * time.Second):
		t.Fatal("stable guardian handle did not kill the escaped root")
	}
	if rerankerProcessGenerationAlive(cmd.Process.Pid, started) {
		t.Fatal("escaped root generation survived guardian shutdown")
	}
}

func TestRerankerSupervisorDoesNotInheritHostDescriptors(t *testing.T) {
	dir := t.TempDir()
	secretPath := filepath.Join(dir, "host-secret")
	const secretContents = "host capability must not leak"
	if err := os.WriteFile(secretPath, []byte(secretContents), 0600); err != nil {
		t.Fatal(err)
	}
	secret, err := os.Open(secretPath)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = secret.Close() }()
	flags, err := unix.FcntlInt(secret.Fd(), unix.F_GETFD, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := unix.FcntlInt(secret.Fd(), unix.F_SETFD, flags&^unix.FD_CLOEXEC); err != nil {
		t.Fatal(err)
	}
	defer func() { _, _ = unix.FcntlInt(secret.Fd(), unix.F_SETFD, flags) }()

	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control, err := createRerankerControlPath(token)
	if err != nil {
		t.Fatal(err)
	}
	const port = 59983
	checked := filepath.Join(dir, "descriptor-checked")
	leaked := filepath.Join(dir, "descriptor-leaked")
	t.Setenv("GO_WANT_RERANKER_FD_HELPER", "1")
	t.Setenv("GO_RERANKER_FD", strconv.Itoa(int(secret.Fd())))
	t.Setenv("GO_RERANKER_FD_SECRET", secretContents)
	t.Setenv("GO_RERANKER_FD_CHECKED", checked)
	t.Setenv("GO_RERANKER_FD_LEAKED", leaked)
	supervisor, done, err := launchRerankerSupervisor(currentSupervisorRegistration(t), dir, control, token, port, os.Args[0], []string{
		"-test.run=TestRerankerFDHelper", "--",
		"--port", strconv.Itoa(port), "--pooling", "rank",
	}, io.Discard)
	if err != nil {
		t.Fatal(err)
	}
	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, supervisor.Process.Pid, done)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _, _ = requestRerankerSupervisor(state, "stop") })

	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(checked); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("reranker did not inspect inherited descriptors")
		}
		time.Sleep(10 * time.Millisecond)
	}
	if _, err := os.Stat(leaked); err == nil {
		t.Fatalf("reranker inherited host descriptor %d", secret.Fd())
	}
}

func TestRerankerManagerDarwinIdentityIgnoresTimezone(t *testing.T) {
	if runtime.GOOS != "darwin" {
		t.Skip("Darwin process identity regression")
	}
	t.Setenv("TZ", "UTC")
	cmd, state := startRerankerHelperProcess(t, false)
	if err := os.Setenv("TZ", "Pacific/Honolulu"); err != nil {
		t.Fatal(err)
	}
	dir := t.TempDir()
	mgr := &RerankerManager{sgrepHome: dir, port: state.Port}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if !mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("same process lost identity when the caller timezone changed")
	}
}

func TestRerankerManagerKeepsStableExecutableIdentityAcrossSymlinkRetarget(t *testing.T) {
	dir := t.TempDir()
	link := filepath.Join(dir, "llama-server")
	target, err := filepath.Abs(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(target, link); err != nil {
		t.Fatal(err)
	}
	const port = 59989
	cmd, state := startRerankerHelperProcessWith(t, link, false, port, []string{
		"--port", strconv.Itoa(port),
		"--pooling", "rank",
	})
	mgr := &RerankerManager{sgrepHome: dir, port: port}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if !mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("initial process ownership failed")
	}
	if err := os.Remove(link); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("/bin/sleep", link); err != nil {
		t.Fatal(err)
	}
	if !mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("live process lost ownership solely because its launcher symlink was retargeted")
	}
}

func TestRerankerManagerKeepsLinuxExecutableIdentityAfterUnlink(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("Linux /proc executable unlink regression")
	}
	dir := t.TempDir()
	executable := filepath.Join(dir, "llama-server")
	data, err := os.ReadFile(os.Args[0])
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(executable, data, 0700); err != nil {
		t.Fatal(err)
	}
	const port = 59980
	cmd, state := startRerankerHelperProcessWith(t, executable, false, port, []string{
		"--port", strconv.Itoa(port),
		"--pooling", "rank",
	})
	mgr := &RerankerManager{sgrepHome: dir, port: port}
	if err := mgr.writePIDState(state); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(executable); err != nil {
		t.Fatal(err)
	}
	if !mgr.ownsProcess(cmd.Process.Pid) {
		t.Fatal("live process lost ownership after its executable was unlinked")
	}
}

func startRerankerHelperProcess(t *testing.T, ignoreTerm bool) (*exec.Cmd, rerankerPIDState) {
	t.Helper()
	const port = 59992
	return startRerankerHelperProcessWith(t, os.Args[0], ignoreTerm, port, []string{
		"--port", strconv.Itoa(port),
		"--pooling", "rank",
	})
}

func startRerankerHelperProcessWith(t *testing.T, executable string, ignoreTerm bool, port int, rerankerArgs []string) (*exec.Cmd, rerankerPIDState) {
	t.Helper()
	instance := "test-instance-" + strings.ReplaceAll(t.Name(), "/", "-")
	ready := filepath.Join(t.TempDir(), "ready")
	args := []string{"-test.run=TestRerankerHelperProcess", "--"}
	args = append(args, rerankerArgs...)
	cmd := exec.Command(executable, args...)
	cmd.Env = append(os.Environ(),
		"GO_WANT_RERANKER_HELPER=1",
		"GO_RERANKER_HELPER_READY="+ready,
		"GO_RERANKER_HELPER_IGNORE_TERM="+strconv.FormatBool(ignoreTerm),
		rerankerInstanceEnv+"="+instance,
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("helper process did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	state, err := inspectRerankerProcess(cmd.Process.Pid)
	if err != nil {
		t.Fatal(err)
	}
	state.Instance = instance
	state.Port = port
	return cmd, state
}

func startTestRerankerSupervisor(t *testing.T, ignoreTerm bool) (rerankerPIDState, <-chan error) {
	t.Helper()
	dir := t.TempDir()
	const port = 59988
	token, err := newRerankerInstanceToken()
	if err != nil {
		t.Fatal(err)
	}
	control := rerankerControlPath(token)
	if _, err := createRerankerControlPath(token); err != nil {
		t.Fatal(err)
	}
	ready := filepath.Join(dir, "ready")
	environment := append(os.Environ(),
		"GO_WANT_RERANKER_HELPER=1",
		"GO_RERANKER_HELPER_READY="+ready,
		"GO_RERANKER_HELPER_IGNORE_TERM="+strconv.FormatBool(ignoreTerm),
	)
	supervisorDone := make(chan error, 1)
	go func() {
		supervisorDone <- runRerankerSupervisor(supervisorConfig{
			Home:        dir,
			Control:     control,
			Token:       token,
			Port:        port,
			Executable:  os.Args[0],
			Args:        []string{"-test.run=TestRerankerHelperProcess", "--", "--port", strconv.Itoa(port), "--pooling", "rank"},
			Environment: environment,
			Stdout:      io.Discard,
			Stderr:      io.Discard,
		})
	}()

	mgr := &RerankerManager{sgrepHome: dir, port: port, host: "localhost"}
	state, err := waitForPublishedRerankerState(mgr, control, token, os.Getpid(), supervisorDone)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := requestRerankerSupervisor(state, "adopt"); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(2 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("supervised helper process did not become ready")
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Cleanup(func() {
		_, _ = requestRerankerSupervisor(state, "stop")
	})
	return state, supervisorDone
}

func TestRerankerHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_RERANKER_HELPER") != "1" {
		return
	}
	if os.Getenv("GO_RERANKER_HELPER_CLOSE_IDENTITY") == "true" {
		if identity := os.NewFile(3, "reranker-identity"); identity != nil {
			_ = identity.Close()
		}
	}
	if os.Getenv("GO_RERANKER_HELPER_SETPGID_PARENT") == "true" {
		err := syscall.Setpgid(0, os.Getppid())
		if os.Getenv("GO_RERANKER_HELPER_REQUIRE_SETPGID_DENIAL") == "true" {
			if err == nil {
				os.Exit(2)
			}
		} else if err != nil {
			os.Exit(2)
		}
	}
	if os.Getenv("GO_RERANKER_HELPER_SETSID") == "true" {
		_, err := syscall.Setsid()
		if os.Getenv("GO_RERANKER_HELPER_REQUIRE_SETSID_DENIAL") == "true" {
			if err == nil {
				os.Exit(2)
			}
		} else if err != nil {
			os.Exit(2)
		}
	}
	if os.Getenv("GO_RERANKER_HELPER_IGNORE_TERM") == "true" {
		signal.Ignore(syscall.SIGTERM)
	}
	if os.Getenv("GO_RERANKER_HELPER_IGNORE_HUP") == "true" {
		signal.Ignore(syscall.SIGHUP)
	}
	if err := os.WriteFile(os.Getenv("GO_RERANKER_HELPER_READY"), []byte("ready"), 0600); err != nil {
		os.Exit(2)
	}
	for {
		time.Sleep(time.Hour)
	}
}

func TestRerankerGroupEscapeHelper(t *testing.T) {
	if os.Getenv("GO_WANT_RERANKER_GROUP_ESCAPE_HELPER") != "1" {
		return
	}
	identity := os.NewFile(3, "reranker-escape-identity")
	if identity == nil {
		fmt.Fprintln(os.Stderr, "identity descriptor unavailable")
		os.Exit(2)
	}
	if _, err := identity.Write([]byte{runnerReady}); err != nil {
		fmt.Fprintln(os.Stderr, "write identity handshake:", err)
		os.Exit(2)
	}
	gate := os.NewFile(4, "reranker-escape-gate")
	if gate == nil {
		fmt.Fprintln(os.Stderr, "gate descriptor unavailable")
		os.Exit(2)
	}
	var command [1]byte
	if _, err := io.ReadFull(gate, command[:]); err != nil {
		fmt.Fprintln(os.Stderr, "read escape gate:", err)
		os.Exit(2)
	}
	_ = identity.Close()
	target, err := strconv.Atoi(os.Getenv("GO_RERANKER_ESCAPE_TARGET_GROUP"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "parse escape target:", err)
		os.Exit(2)
	}
	if err := syscall.Setpgid(0, target); err != nil {
		fmt.Fprintln(os.Stderr, "escape process group:", err)
		os.Exit(2)
	}
	signal.Ignore(syscall.SIGTERM, syscall.SIGHUP)
	if err := os.WriteFile(os.Getenv("GO_RERANKER_ESCAPE_READY"), []byte("escaped"), 0600); err != nil {
		os.Exit(2)
	}
	for {
		time.Sleep(time.Hour)
	}
}

func TestRerankerDescendantHelper(t *testing.T) {
	if os.Getenv("GO_WANT_RERANKER_DESCENDANT_HELPER") != "1" {
		return
	}
	signal.Ignore(syscall.SIGHUP, syscall.SIGTERM)
	descendant := exec.Command("/bin/sleep", "30")
	if os.Getenv("GO_RERANKER_DESCENDANT_SETSID") == "true" {
		descendant.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	}
	if err := descendant.Start(); err != nil {
		if writeErr := os.WriteFile(os.Getenv("GO_RERANKER_DESCENDANT_DENIED"), []byte(err.Error()), 0600); writeErr != nil {
			os.Exit(2)
		}
		for {
			time.Sleep(time.Hour)
		}
	}
	if err := os.WriteFile(os.Getenv("GO_RERANKER_DESCENDANT_PID"), []byte(strconv.Itoa(descendant.Process.Pid)), 0600); err != nil {
		os.Exit(2)
	}
	for {
		time.Sleep(time.Hour)
	}
}

func TestRerankerFDHelper(t *testing.T) {
	if os.Getenv("GO_WANT_RERANKER_FD_HELPER") != "1" {
		return
	}
	descriptor, _ := strconv.Atoi(os.Getenv("GO_RERANKER_FD"))
	file := os.NewFile(uintptr(descriptor), "inherited-host-descriptor")
	if file != nil {
		buffer := make([]byte, 128)
		if count, err := file.Read(buffer); err == nil && string(buffer[:count]) == os.Getenv("GO_RERANKER_FD_SECRET") {
			_ = os.WriteFile(os.Getenv("GO_RERANKER_FD_LEAKED"), []byte("leaked"), 0600)
		}
	}
	if err := os.WriteFile(os.Getenv("GO_RERANKER_FD_CHECKED"), []byte("checked"), 0600); err != nil {
		os.Exit(2)
	}
	for {
		time.Sleep(time.Hour)
	}
}

func testServerPort(t *testing.T, rawURL string) int {
	t.Helper()
	index := strings.LastIndexByte(rawURL, ':')
	if index < 0 {
		t.Fatalf("server URL %q has no port", rawURL)
	}
	port, err := strconv.Atoi(rawURL[index+1:])
	if err != nil {
		t.Fatal(err)
	}
	return port
}

func mustExecutable(t *testing.T) string {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	return executable
}

func currentSupervisorRegistration(t *testing.T) supervisorExecutableRegistration {
	t.Helper()
	executable := mustExecutable(t)
	file, err := os.Open(executable)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = file.Close() })
	registration, err := newSupervisorExecutableRegistration(executable, file)
	if err != nil {
		t.Fatal(err)
	}
	return registration
}
