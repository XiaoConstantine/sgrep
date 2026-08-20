package rerank

import (
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"
)

const (
	rerankerSupervisorCommand = "__sgrep-reranker-supervisor"
	rerankerGuardianCommand   = "__sgrep-reranker-guardian"
	rerankerRunnerCommand     = "__sgrep-reranker-runner"
	supervisorModeEnv         = "SGREP_RERANKER_SUPERVISOR_MODE"
	supervisorProtocolEnv     = "SGREP_RERANKER_SUPERVISOR_PROTOCOL"
	supervisorEphemeralEnv    = "SGREP_RERANKER_SUPERVISOR_EPHEMERAL"
	supervisorControlEnv      = "SGREP_RERANKER_SUPERVISOR_CONTROL"
	supervisorTokenEnv        = "SGREP_RERANKER_SUPERVISOR_TOKEN"
	supervisorHomeEnv         = "SGREP_RERANKER_SUPERVISOR_HOME"
	supervisorPortEnv         = "SGREP_RERANKER_SUPERVISOR_PORT"
	supervisorLifecycleEnv    = "SGREP_RERANKER_SUPERVISOR_LIFECYCLE_LOCK"
	supervisorProtocolVersion = 1
	guardianCleanShutdown     = byte(0x73)
	guardianReady             = byte(0x61)
	guardianRegister          = byte(0x72)
	guardianRefresh           = byte(0x66)
	guardianRegistered        = byte(0x64)
	runnerReady               = byte(0x67)
	runnerArmExec             = byte(0x74)
	runnerExecArmed           = byte(0x65)
	runnerContinue            = byte(0x63)
	supervisorStopTimeout     = 500 * time.Millisecond
	supervisorSetupTimeout    = 5 * time.Second
	supervisorAdoptionTimeout = 10 * time.Second
	supervisorReadTimeout     = 500 * time.Millisecond
	supervisorRequestLimit    = 4 << 10
	supervisorMaxClients      = 16
	guardianRegistrationLimit = 4 << 10
)

var registeredSupervisorExecutable struct {
	sync.RWMutex
	supervisorExecutableRegistration
}

type supervisorExecutableRegistration struct {
	Path   string
	Device uint64
	Inode  uint64
	Size   int64
	Digest [sha256.Size]byte
	File   *os.File
}

type supervisorConfig struct {
	Home                 string
	Control              string
	Token                string
	Port                 int
	Executable           string
	Args                 []string
	Environment          []string
	Stdout               io.Writer
	Stderr               io.Writer
	HandleSignals        bool
	GuardProcessLifetime bool
	AdoptionTimeout      time.Duration
	AdoptionLease        time.Duration
	LifecycleLock        *os.File
	BeforePublish        func() // test hook
}

type supervisorRequest struct {
	Protocol int    `json:"protocol"`
	Nonce    string `json:"nonce"`
	Action   string `json:"action"`
	Proof    string `json:"proof"`
}

type supervisorResponse struct {
	Protocol int    `json:"protocol"`
	PID      int    `json:"pid,omitempty"`
	Running  bool   `json:"running,omitempty"`
	Error    string `json:"error,omitempty"`
	Proof    string `json:"proof"`
}

type supervisorCommand struct {
	action   string
	response chan supervisorResponse
	written  chan struct{}
}

type rerankerGuardian struct {
	process    *os.Process
	lifetime   *os.File
	done       <-chan error
	registered bool
}

type rerankerIdentitySockets struct {
	supervisor *os.File
	guardian   *os.File
	root       *os.File
}

type guardianRegistration struct {
	Protocol int    `json:"protocol"`
	PID      int    `json:"pid"`
	Started  string `json:"started"`
	Handle   string `json:"handle,omitempty"`
}

// RunSupervisorCommand handles sgrep's private reranker subprocess modes.
// The sgrep command must call it before dispatching normal CLI commands.
func RunSupervisorCommand(args []string) (bool, error) {
	if len(args) > 0 && (args[0] == rerankerSupervisorCommand || args[0] == rerankerGuardianCommand || args[0] == rerankerRunnerCommand) {
		if os.Getenv(supervisorModeEnv) != "1" || os.Getenv(supervisorProtocolEnv) != strconv.Itoa(supervisorProtocolVersion) {
			return true, fmt.Errorf("private reranker command cannot be invoked directly")
		}
		if args[0] == rerankerGuardianCommand {
			if err := closeInheritedDescriptors(3, 4); err != nil {
				return true, err
			}
			return true, runGuardianFromEnvironment(args)
		}
		if args[0] == rerankerRunnerCommand {
			if err := closeInheritedDescriptors(3); err != nil {
				return true, err
			}
			return true, runConstrainedReranker(args)
		}
		var kept []int
		if os.Getenv(supervisorLifecycleEnv) == "1" {
			kept = append(kept, 3)
		}
		if err := closeInheritedDescriptors(kept...); err != nil {
			return true, err
		}
		return true, runSupervisorFromEnvironment(args)
	}
	if os.Getenv(supervisorModeEnv) == "1" {
		return true, fmt.Errorf("invalid private reranker command")
	}
	executable, err := os.Executable()
	if err != nil {
		return false, fmt.Errorf("register reranker supervisor executable: %w", err)
	}
	file, err := openCurrentSupervisorExecutable(executable)
	if err != nil {
		return false, fmt.Errorf("register reranker supervisor executable: %w", err)
	}
	registration, err := newSupervisorExecutableRegistration(executable, file)
	if err != nil {
		_ = file.Close()
		return false, fmt.Errorf("register reranker supervisor executable: %w", err)
	}
	registeredSupervisorExecutable.Lock()
	if registeredSupervisorExecutable.File == nil {
		registeredSupervisorExecutable.supervisorExecutableRegistration = registration
		file = nil
	}
	registeredSupervisorExecutable.Unlock()
	if file != nil {
		_ = file.Close()
	}
	return false, nil
}

func newSupervisorExecutableRegistration(path string, file *os.File) (supervisorExecutableRegistration, error) {
	if file == nil {
		return supervisorExecutableRegistration{}, fmt.Errorf("executable handle is unavailable")
	}
	info, err := file.Stat()
	if err != nil {
		return supervisorExecutableRegistration{}, err
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return supervisorExecutableRegistration{}, fmt.Errorf("unsupported executable file identity")
	}
	hash := sha256.New()
	if _, err := io.Copy(hash, io.NewSectionReader(file, 0, info.Size())); err != nil {
		return supervisorExecutableRegistration{}, err
	}
	var digest [sha256.Size]byte
	copy(digest[:], hash.Sum(nil))
	return supervisorExecutableRegistration{
		Path: absoluteExecutablePath(path), Device: uint64(stat.Dev), Inode: uint64(stat.Ino),
		Size: info.Size(), Digest: digest, File: file,
	}, nil
}

func runSupervisorFromEnvironment(args []string) error {
	if len(args) < 3 || args[1] != "--" {
		return fmt.Errorf("invalid reranker supervisor command")
	}
	port, err := strconv.Atoi(os.Getenv(supervisorPortEnv))
	if err != nil || port <= 0 {
		return fmt.Errorf("invalid reranker supervisor port")
	}
	home := os.Getenv(supervisorHomeEnv)
	lifecycleLock, err := inheritedLifecycleLock(home, os.Getenv(supervisorLifecycleEnv) == "1")
	if err != nil {
		return err
	}
	if lifecycleLock != nil {
		defer func() { _ = lifecycleLock.Close() }()
	}

	err = runRerankerSupervisor(supervisorConfig{
		Home:                 home,
		Control:              os.Getenv(supervisorControlEnv),
		Token:                os.Getenv(supervisorTokenEnv),
		Port:                 port,
		Executable:           args[2],
		Args:                 args[3:],
		Environment:          os.Environ(),
		Stdout:               os.Stdout,
		Stderr:               os.Stderr,
		HandleSignals:        true,
		GuardProcessLifetime: true,
		LifecycleLock:        lifecycleLock,
	})
	if os.Getenv(supervisorEphemeralEnv) == "1" {
		cleanupStagedSupervisorExecutable(os.Getenv(supervisorControlEnv))
	}
	return err
}

func inheritedLifecycleLock(home string, required bool) (*os.File, error) {
	if !required {
		return nil, nil
	}
	if _, err := unix.FcntlInt(3, unix.F_GETFD, 0); errors.Is(err, unix.EBADF) {
		return nil, fmt.Errorf("required reranker lifecycle lock was not inherited")
	} else if err != nil {
		return nil, fmt.Errorf("inspect inherited reranker lifecycle lock: %w", err)
	}
	lock := os.NewFile(3, "reranker-lifecycle-lock")
	if lock == nil {
		return nil, fmt.Errorf("reranker lifecycle lock descriptor is unavailable")
	}
	info, err := lock.Stat()
	if err != nil {
		_ = lock.Close()
		return nil, fmt.Errorf("inspect inherited reranker lifecycle lock: %w", err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !info.Mode().IsRegular() || !ok || int(stat.Uid) != os.Getuid() || stat.Nlink != 1 || info.Mode().Perm()&0077 != 0 {
		_ = lock.Close()
		return nil, fmt.Errorf("inherited reranker lifecycle lock has unsafe identity or permissions")
	}
	expected, err := os.Stat(filepath.Join(home, ".reranker.lock"))
	if err != nil {
		_ = lock.Close()
		return nil, fmt.Errorf("inspect SGREP_HOME reranker lifecycle lock: %w", err)
	}
	if !os.SameFile(info, expected) {
		_ = lock.Close()
		return nil, fmt.Errorf("inherited reranker lifecycle lock does not match SGREP_HOME")
	}
	syscall.CloseOnExec(3)
	return lock, nil
}

func runConstrainedReranker(args []string) error {
	if len(args) < 3 || args[1] != "--" {
		return fmt.Errorf("invalid constrained reranker command")
	}
	identity := os.NewFile(3, "reranker-identity")
	if identity == nil {
		return fmt.Errorf("reranker identity descriptor is unavailable")
	}
	defer func() { _ = identity.Close() }()
	if !preserveRerankerIdentityAfterExec() {
		syscall.CloseOnExec(3)
	}
	executable, commandArgs, err := prepareConstrainedReranker(args[2], args[3:])
	if err != nil {
		return err
	}
	if _, err := identity.Write([]byte{runnerReady}); err != nil {
		return fmt.Errorf("announce constrained reranker: %w", err)
	}
	var command [1]byte
	if _, err := io.ReadFull(identity, command[:]); err != nil {
		return fmt.Errorf("wait for constrained reranker authorization: %w", err)
	}
	if command[0] != runnerArmExec {
		return fmt.Errorf("invalid constrained reranker trace authorization")
	}
	if err := enableRerankerExecTrace(); err != nil {
		return fmt.Errorf("arm constrained reranker exec trace: %w", err)
	}
	if _, err := identity.Write([]byte{runnerExecArmed}); err != nil {
		return fmt.Errorf("announce constrained reranker exec trace: %w", err)
	}
	if _, err := io.ReadFull(identity, command[:]); err != nil {
		return fmt.Errorf("wait for constrained reranker exec authorization: %w", err)
	}
	if command[0] != runnerContinue {
		return fmt.Errorf("invalid constrained reranker authorization")
	}
	environment := withoutSupervisorEnvironment(os.Environ())
	if err := syscall.Exec(executable, commandArgs, environment); err != nil {
		return fmt.Errorf("exec constrained reranker: %w", err)
	}
	return fmt.Errorf("constrained reranker exec returned unexpectedly")
}

func runGuardianFromEnvironment(args []string) error {
	if len(args) != 3 || args[2] != "--" {
		return fmt.Errorf("invalid reranker guardian command")
	}
	rootPID, err := strconv.Atoi(args[1])
	if err != nil || rootPID <= 1 {
		return fmt.Errorf("invalid reranker guardian root PID")
	}
	actualPGID, err := syscall.Getpgid(0)
	if err != nil || actualPGID != rootPID {
		return fmt.Errorf("reranker guardian is outside its protected process group")
	}
	lifetime := os.NewFile(3, "reranker-supervisor-lifetime")
	if lifetime == nil {
		return fmt.Errorf("reranker guardian lifetime descriptor is unavailable")
	}
	defer func() { _ = lifetime.Close() }()
	if _, err := lifetime.Stat(); err != nil {
		return fmt.Errorf("inspect reranker guardian lifetime descriptor: %w", err)
	}
	syscall.CloseOnExec(3)
	identity := os.NewFile(4, "reranker-guardian-identity")
	if identity == nil {
		return fmt.Errorf("reranker guardian identity descriptor is unavailable")
	}
	defer func() { _ = identity.Close() }()
	if _, err := identity.Stat(); err != nil {
		return fmt.Errorf("inspect reranker guardian identity descriptor: %w", err)
	}
	syscall.CloseOnExec(4)
	written, err := lifetime.Write([]byte{guardianReady})
	if err != nil {
		return fmt.Errorf("announce reranker guardian: %w", err)
	}
	if written != 1 {
		return fmt.Errorf("announce reranker guardian: %w", io.ErrShortWrite)
	}

	var command [1]byte
	if _, err := io.ReadFull(lifetime, command[:]); err != nil {
		return killAbandonedRerankerGroup(rootPID, err)
	}
	if command[0] == guardianCleanShutdown {
		return nil
	}
	if command[0] != guardianRegister {
		return killAbandonedRerankerGroup(rootPID, fmt.Errorf("invalid reranker guardian registration command"))
	}
	var registration guardianRegistration
	if err := json.NewDecoder(io.LimitReader(lifetime, guardianRegistrationLimit)).Decode(&registration); err != nil {
		return killAbandonedRerankerGroup(rootPID, fmt.Errorf("decode reranker guardian registration: %w", err))
	}
	if registration.Protocol != supervisorProtocolVersion || registration.PID != rootPID || registration.Started == "" {
		return killAbandonedRerankerGroup(rootPID, fmt.Errorf("invalid reranker guardian registration"))
	}
	handle, err := restoreRerankerProcessHandle(rootPID, registration.Started, registration.Handle, identity)
	if err != nil {
		return killAbandonedRerankerGroup(rootPID, fmt.Errorf("acquire stable guardian process handle: %w", err))
	}
	defer func() { _ = handle.Close() }()
	if _, err := lifetime.Write([]byte{guardianRegistered}); err != nil {
		_ = handle.Signal(syscall.SIGKILL)
		return fmt.Errorf("acknowledge reranker guardian registration: %w", err)
	}

	var readErr error
	for {
		_, readErr = io.ReadFull(lifetime, command[:])
		if readErr != nil {
			break
		}
		switch command[0] {
		case guardianCleanShutdown:
			return nil
		case guardianRefresh:
			var refresh guardianRegistration
			if err := json.NewDecoder(io.LimitReader(lifetime, guardianRegistrationLimit)).Decode(&refresh); err != nil {
				readErr = fmt.Errorf("decode reranker guardian handle refresh: %w", err)
				break
			}
			if refresh.Protocol != supervisorProtocolVersion || refresh.PID != rootPID || refresh.Started != registration.Started || refresh.Handle == "" {
				readErr = fmt.Errorf("invalid reranker guardian handle refresh")
				break
			}
			refreshed, err := restoreRerankerProcessHandle(rootPID, refresh.Started, refresh.Handle, identity)
			if err != nil {
				readErr = fmt.Errorf("refresh stable guardian process handle: %w", err)
				break
			}
			_ = handle.Close()
			handle = refreshed
			if _, err := lifetime.Write([]byte{guardianRegistered}); err != nil {
				readErr = fmt.Errorf("acknowledge reranker guardian handle refresh: %w", err)
				break
			}
			continue
		default:
			readErr = fmt.Errorf("invalid reranker guardian command")
		}
		break
	}
	// Reacquire at the moment of abandonment. Darwin's audit token can change
	// across exec, while Linux pidfds remain stable; this gives both platforms
	// a post-exec, generation-bound kill capability without a supervisor-owned
	// refresh window.
	refreshErr := error(nil)
	if refreshed, err := openRerankerProcessHandle(rootPID, registration.Started, identity); err == nil {
		_ = handle.Close()
		handle = refreshed
	} else {
		refreshErr = fmt.Errorf("refresh stable guardian process handle: %w", err)
	}
	if err := handle.Signal(syscall.SIGKILL); err != nil {
		if !errors.Is(err, os.ErrProcessDone) || rerankerProcessGenerationAlive(rootPID, registration.Started) {
			return killAbandonedRerankerGroup(rootPID, errors.Join(readErr, refreshErr, fmt.Errorf("kill abandoned reranker process: %w", err)))
		}
	}
	return errors.Join(fmt.Errorf("reranker guardian terminated abandoned process"), readErr, refreshErr)
}

func killAbandonedRerankerGroup(rootPID int, reason error) error {
	// Before stable-handle registration the constrained runner cannot create
	// processes or leave its group. The guardian anchors that group against
	// reuse, so this is a safe failure-path fallback.
	if err := syscall.Kill(-rootPID, syscall.SIGKILL); err != nil && !errors.Is(err, syscall.ESRCH) {
		return errors.Join(reason, fmt.Errorf("kill abandoned reranker process group %d: %w", rootPID, err))
	}
	return reason
}

func launchRerankerSupervisor(supervisorExecutable supervisorExecutableRegistration, home, control, token string, port int, rerankerExecutable string, args []string, output io.Writer) (*exec.Cmd, <-chan error, error) {
	return launchRerankerSupervisorWithLifecycleLock(supervisorExecutable, home, control, token, port, rerankerExecutable, args, output, nil)
}

func launchRerankerSupervisorWithLifecycleLock(supervisorExecutable supervisorExecutableRegistration, home, control, token string, port int, rerankerExecutable string, args []string, output io.Writer, lifecycleLock *os.File) (*exec.Cmd, <-chan error, error) {
	stagedExecutable, err := stageSupervisorExecutable(supervisorExecutable, control)
	if err != nil {
		return nil, nil, err
	}
	commandArgs := []string{rerankerSupervisorCommand, "--", rerankerExecutable}
	commandArgs = append(commandArgs, args...)
	cmd := exec.Command(stagedExecutable, commandArgs...)
	cmd.Env = environmentForSupervisor(os.Environ(), home, control, token, port)
	cmd.Env = append(cmd.Env, supervisorEphemeralEnv+"=1")
	cmd.Stdout = output
	cmd.Stderr = output
	if lifecycleLock != nil {
		cmd.ExtraFiles = []*os.File{lifecycleLock}
		cmd.Env = append(cmd.Env, supervisorLifecycleEnv+"=1")
	}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
	if err := cmd.Start(); err != nil {
		_ = os.Remove(stagedExecutable)
		return nil, nil, err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	return cmd, done, nil
}

func stageSupervisorExecutable(registration supervisorExecutableRegistration, control string) (string, error) {
	if registration.File == nil {
		return "", fmt.Errorf("registered reranker supervisor has no retained executable handle")
	}
	info, err := registration.File.Stat()
	if err != nil {
		return "", fmt.Errorf("inspect registered reranker supervisor: %w", err)
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || uint64(stat.Dev) != registration.Device || uint64(stat.Ino) != registration.Inode || info.Size() != registration.Size {
		return "", fmt.Errorf("registered reranker supervisor was replaced before launch")
	}

	destination := stagedSupervisorPath(control)
	staged, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0700)
	if err != nil {
		return "", fmt.Errorf("create private reranker supervisor copy: %w", err)
	}
	complete := false
	defer func() {
		_ = staged.Close()
		if !complete {
			_ = os.Remove(destination)
		}
	}()
	source := io.NewSectionReader(registration.File, 0, registration.Size)
	hash := sha256.New()
	if _, err := io.Copy(io.MultiWriter(staged, hash), source); err != nil {
		return "", fmt.Errorf("copy private reranker supervisor: %w", err)
	}
	if !hmac.Equal(hash.Sum(nil), registration.Digest[:]) {
		return "", fmt.Errorf("registered reranker supervisor contents changed before launch")
	}
	if err := staged.Sync(); err != nil {
		return "", fmt.Errorf("sync private reranker supervisor: %w", err)
	}
	if err := staged.Close(); err != nil {
		return "", fmt.Errorf("close private reranker supervisor: %w", err)
	}
	complete = true
	return destination, nil
}

func stagedSupervisorPath(control string) string {
	return filepath.Join(filepath.Dir(control), "supervisor")
}

func cleanupStagedSupervisorExecutable(control string) {
	directory := filepath.Dir(control)
	if filepath.Dir(directory) != "/tmp" || !strings.HasPrefix(filepath.Base(directory), fmt.Sprintf("sgrep-reranker-%d-", os.Getuid())) {
		return
	}
	info, err := os.Lstat(directory)
	if err != nil || !info.IsDir() || info.Mode().Perm()&0077 != 0 {
		return
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok || int(stat.Uid) != os.Getuid() {
		return
	}
	path := stagedSupervisorPath(control)
	if staged, err := os.Lstat(path); err == nil && staged.Mode().IsRegular() {
		_ = os.Remove(path)
	}
	_ = os.Remove(directory)
}

func newRerankerIdentitySockets() (*rerankerIdentitySockets, error) {
	descriptors, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		return nil, fmt.Errorf("create reranker identity socket: %w", err)
	}
	for _, descriptor := range descriptors {
		unix.CloseOnExec(descriptor)
	}
	guardianDescriptor, err := unix.Dup(descriptors[0])
	if err != nil {
		_ = unix.Close(descriptors[0])
		_ = unix.Close(descriptors[1])
		return nil, fmt.Errorf("duplicate reranker identity socket for guardian: %w", err)
	}
	unix.CloseOnExec(guardianDescriptor)
	return &rerankerIdentitySockets{
		supervisor: os.NewFile(uintptr(descriptors[0]), "reranker-supervisor-identity"),
		guardian:   os.NewFile(uintptr(guardianDescriptor), "reranker-guardian-identity"),
		root:       os.NewFile(uintptr(descriptors[1]), "reranker-root-identity"),
	}, nil
}

func (s *rerankerIdentitySockets) Close() {
	if s == nil {
		return
	}
	for _, file := range []*os.File{s.supervisor, s.guardian, s.root} {
		if file != nil {
			_ = file.Close()
		}
	}
}

func waitForRunnerReady(identity *os.File) error {
	if identity == nil {
		return fmt.Errorf("reranker identity socket is unavailable")
	}
	ready, err := waitForDescriptorReadable(identity, supervisorSetupTimeout)
	if err != nil {
		return fmt.Errorf("wait for constrained reranker: %w", err)
	}
	if !ready {
		return fmt.Errorf("constrained reranker did not initialize")
	}
	var message [1]byte
	if _, err := io.ReadFull(identity, message[:]); err != nil || message[0] != runnerReady {
		return fmt.Errorf("constrained reranker handshake failed: %w", err)
	}
	return nil
}

func authorizeRunnerExec(identity *os.File) error {
	if identity == nil {
		return fmt.Errorf("reranker identity socket is unavailable")
	}
	written, err := identity.Write([]byte{runnerContinue})
	if err != nil {
		return fmt.Errorf("authorize constrained reranker: %w", err)
	}
	if written != 1 {
		return fmt.Errorf("authorize constrained reranker: %w", io.ErrShortWrite)
	}
	return nil
}

func armRunnerExec(identity *os.File) error {
	if identity == nil {
		return fmt.Errorf("reranker identity socket is unavailable")
	}
	if _, err := identity.Write([]byte{runnerArmExec}); err != nil {
		return fmt.Errorf("arm constrained reranker exec trace: %w", err)
	}
	ready, err := waitForDescriptorReadable(identity, supervisorSetupTimeout)
	if err != nil {
		return fmt.Errorf("wait for constrained reranker exec trace: %w", err)
	}
	if !ready {
		return fmt.Errorf("constrained reranker did not arm its exec trace")
	}
	var message [1]byte
	if _, err := io.ReadFull(identity, message[:]); err != nil {
		return fmt.Errorf("read constrained reranker exec trace handshake: %w", err)
	}
	if message[0] != runnerExecArmed {
		return fmt.Errorf("invalid constrained reranker exec trace handshake")
	}
	return nil
}

func launchRerankerGuardian(output io.Writer, rootPID int, identity *os.File) (*rerankerGuardian, error) {
	if identity == nil {
		return nil, fmt.Errorf("reranker guardian identity socket is unavailable")
	}
	executable, err := os.Executable()
	if err != nil {
		return nil, fmt.Errorf("find reranker guardian executable: %w", err)
	}
	descriptors, err := unix.Socketpair(unix.AF_UNIX, unix.SOCK_STREAM, 0)
	if err != nil {
		return nil, fmt.Errorf("create reranker guardian lifetime socket: %w", err)
	}
	for _, descriptor := range descriptors {
		unix.CloseOnExec(descriptor)
	}
	lifetime := os.NewFile(uintptr(descriptors[0]), "reranker-supervisor-lifetime")
	guardianLifetime := os.NewFile(uintptr(descriptors[1]), "reranker-guardian-lifetime")
	defer func() { _ = guardianLifetime.Close() }()

	cmd := exec.Command(executable, rerankerGuardianCommand, strconv.Itoa(rootPID), "--")
	cmd.Env = environmentForGuardian(os.Environ())
	cmd.Stdout = output
	cmd.Stderr = output
	cmd.ExtraFiles = []*os.File{guardianLifetime, identity}
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true, Pgid: rootPID}
	if err := cmd.Start(); err != nil {
		_ = lifetime.Close()
		return nil, fmt.Errorf("start reranker lifetime guardian: %w", err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	return &rerankerGuardian{process: cmd.Process, lifetime: lifetime, done: done}, nil
}

func (g *rerankerGuardian) registerRoot(pid int, started string) error {
	if g == nil {
		return nil
	}
	if pid <= 1 || started == "" {
		return fmt.Errorf("invalid reranker root registration")
	}
	if g.registered {
		return fmt.Errorf("reranker guardian root is already registered")
	}
	if err := g.sendRegistration(guardianRegister, guardianRegistration{Protocol: supervisorProtocolVersion, PID: pid, Started: started}); err != nil {
		return err
	}
	g.registered = true
	return nil
}

func (g *rerankerGuardian) refreshRoot(pid int, started string, handle *rerankerProcessHandle) error {
	if g == nil {
		return nil
	}
	if !g.registered {
		return fmt.Errorf("reranker guardian root is not registered")
	}
	encoded, err := handle.encodeGuardianHandle()
	if err != nil {
		return fmt.Errorf("encode reranker guardian process handle: %w", err)
	}
	// Linux pidfds are reacquired rather than serialized. A non-empty marker
	// still distinguishes a post-exec refresh from initial registration.
	if encoded == "" {
		encoded = "reopen"
	}
	return g.sendRegistration(guardianRefresh, guardianRegistration{
		Protocol: supervisorProtocolVersion,
		PID:      pid,
		Started:  started,
		Handle:   encoded,
	})
}

func (g *rerankerGuardian) sendRegistration(command byte, registration guardianRegistration) error {
	if _, err := g.lifetime.Write([]byte{command}); err != nil {
		return fmt.Errorf("begin reranker guardian registration: %w", err)
	}
	if err := json.NewEncoder(g.lifetime).Encode(registration); err != nil {
		return fmt.Errorf("register reranker root with guardian: %w", err)
	}
	ready, err := waitForDescriptorReadable(g.lifetime, supervisorSetupTimeout)
	if err != nil {
		return fmt.Errorf("wait for reranker guardian registration: %w", err)
	}
	if !ready {
		return fmt.Errorf("reranker guardian did not register the root")
	}
	var message [1]byte
	if _, err := io.ReadFull(g.lifetime, message[:]); err != nil {
		return fmt.Errorf("read reranker guardian registration: %w", err)
	}
	if message[0] != guardianRegistered {
		return fmt.Errorf("invalid reranker guardian registration response")
	}
	return nil
}

func (g *rerankerGuardian) waitReady() error {
	if g == nil {
		return nil
	}
	ready, err := waitForDescriptorReadable(g.lifetime, supervisorSetupTimeout)
	if err != nil {
		return fmt.Errorf("wait for reranker guardian: %w", err)
	}
	if !ready {
		return fmt.Errorf("reranker guardian did not initialize")
	}
	var message [1]byte
	if _, err := io.ReadFull(g.lifetime, message[:]); err != nil {
		return fmt.Errorf("read reranker guardian handshake: %w", err)
	}
	if message[0] != guardianReady {
		return fmt.Errorf("invalid reranker guardian handshake")
	}
	return nil
}

func waitForDescriptorReadable(file *os.File, timeout time.Duration) (bool, error) {
	deadline := time.Now().Add(timeout)
	for {
		remaining := time.Until(deadline)
		if remaining <= 0 {
			return false, nil
		}
		milliseconds := int((remaining + time.Millisecond - 1) / time.Millisecond)
		poll := []unix.PollFd{{Fd: int32(file.Fd()), Events: unix.POLLIN}}
		count, err := unix.Poll(poll, milliseconds)
		if errors.Is(err, syscall.EINTR) {
			continue
		}
		if err != nil {
			return false, err
		}
		if count == 0 {
			return false, nil
		}
		return poll[0].Revents&(unix.POLLIN|unix.POLLHUP) != 0, nil
	}
}

func (g *rerankerGuardian) finishCleanly() error {
	if g == nil {
		return nil
	}
	_, writeErr := g.lifetime.Write([]byte{guardianCleanShutdown})
	closeErr := g.lifetime.Close()
	select {
	case waitErr := <-g.done:
		return errors.Join(writeErr, closeErr, waitErr)
	case <-time.After(2 * time.Second):
		killErr := g.process.Kill()
		select {
		case <-g.done:
		case <-time.After(2 * time.Second):
		}
		return errors.Join(writeErr, closeErr, fmt.Errorf("reranker guardian did not exit after clean shutdown"), killErr)
	}
}

func (g *rerankerGuardian) terminateRoot() error {
	if g == nil {
		return nil
	}
	closeErr := g.lifetime.Close()
	select {
	case waitErr := <-g.done:
		return errors.Join(closeErr, unexpectedGuardianWaitError(waitErr))
	case <-time.After(2 * time.Second):
		killErr := g.process.Kill()
		return errors.Join(closeErr, fmt.Errorf("reranker guardian did not terminate the root after lifetime closure"), killErr)
	}
}

func unexpectedGuardianWaitError(err error) error {
	var exitErr *exec.ExitError
	if errors.As(err, &exitErr) {
		if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() && status.Signal() == syscall.SIGKILL {
			return nil
		}
	}
	return err
}

func runRerankerSupervisor(cfg supervisorConfig) error {
	if cfg.Home == "" || cfg.Control == "" || cfg.Token == "" || cfg.Port <= 0 || cfg.Executable == "" {
		return fmt.Errorf("incomplete reranker supervisor configuration")
	}
	if cfg.GuardProcessLifetime && syscall.Getpgrp() != os.Getpid() {
		return fmt.Errorf("reranker supervisor is outside its private process group")
	}
	lifecycleLock := cfg.LifecycleLock
	if lifecycleLock != nil {
		defer func() {
			if lifecycleLock != nil {
				_ = lifecycleLock.Close()
			}
		}()
	}
	expectedDevice, expectedInode, err := configuredRerankerExecutableIdentity(cfg.Executable)
	if err != nil {
		return fmt.Errorf("identify configured reranker executable: %w", err)
	}

	listener, socketInfo, err := listenSupervisorSocket(cfg.Control)
	if err != nil {
		return err
	}
	defer cleanupSupervisorSocket(listener, cfg.Control, socketInfo)

	var signals chan os.Signal
	if cfg.HandleSignals {
		signals = make(chan os.Signal, 1)
		signal.Notify(signals, syscall.SIGINT, syscall.SIGTERM, syscall.SIGHUP)
		defer signal.Stop(signals)
	}

	identities, err := newRerankerIdentitySockets()
	if err != nil {
		return err
	}
	defer identities.Close()

	runnerExecutable, err := os.Executable()
	if err != nil {
		return fmt.Errorf("find constrained reranker runner: %w", err)
	}
	runnerArgs := []string{rerankerRunnerCommand, "--", cfg.Executable}
	runnerArgs = append(runnerArgs, cfg.Args...)
	cmd := exec.Command(runnerExecutable, runnerArgs...)
	cmd.Env = environmentForRunner(cfg.Environment, cfg.Token)
	cmd.Stdout = cfg.Stdout
	cmd.Stderr = cfg.Stderr
	cmd.ExtraFiles = []*os.File{identities.root}
	// The reranker is the leader of a private process group before it can
	// exec. A guardian joins that group while the runner is still blocked.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start reranker process: %w", err)
	}
	_ = identities.root.Close()
	identities.root = nil
	var guardian *rerankerGuardian
	var guardianDone <-chan error
	childReaped := false
	abortBeforeHandle := func(reason error) error {
		if !childReaped {
			_ = cmd.Process.Kill()
		}
		waitErr := cmd.Wait()
		if childReaped {
			waitErr = nil
		}
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			waitErr = nil
		}
		return errors.Join(reason, waitErr, guardian.finishCleanly())
	}

	if err := waitForRunnerReady(identities.supervisor); err != nil {
		return abortBeforeHandle(err)
	}
	if cfg.GuardProcessLifetime {
		guardian, err = launchRerankerGuardian(cfg.Stderr, cmd.Process.Pid, identities.guardian)
		if err != nil {
			return abortBeforeHandle(err)
		}
		_ = identities.guardian.Close()
		identities.guardian = nil
		guardianDone = guardian.done
		if err := guardian.waitReady(); err != nil {
			return abortBeforeHandle(err)
		}
		started := rerankerProcessStartIdentity(cmd.Process.Pid)
		if started == "" {
			return abortBeforeHandle(fmt.Errorf("identify constrained reranker generation before exec"))
		}
		if err := guardian.registerRoot(cmd.Process.Pid, started); err != nil {
			return abortBeforeHandle(err)
		}
	}
	if err := armRunnerExec(identities.supervisor); err != nil {
		return abortBeforeHandle(err)
	}
	if err := authorizeRunnerExec(identities.supervisor); err != nil {
		return abortBeforeHandle(err)
	}
	state, releaseExec, reaped, err := waitForStoppedRerankerExec(cmd.Process.Pid, expectedDevice, expectedInode)
	childReaped = reaped
	if err != nil {
		return abortBeforeHandle(err)
	}
	if state.PGID != state.PID {
		return abortBeforeHandle(fmt.Errorf("reranker escaped its protected process group"))
	}
	handle, err := openRerankerProcessHandle(state.PID, state.Started, identities.supervisor)
	if err != nil {
		return abortBeforeHandle(fmt.Errorf("acquire stable reranker process handle: %w", err))
	}
	defer func() { _ = handle.Close() }()
	if err := guardian.refreshRoot(state.PID, state.Started, handle); err != nil {
		_ = handle.Signal(syscall.SIGKILL)
		return abortBeforeHandle(fmt.Errorf("refresh reranker guardian process handle: %w", err))
	}
	if err := releaseExec(); err != nil {
		_ = handle.Signal(syscall.SIGKILL)
		return abortBeforeHandle(err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	stopAndFinish := func(reason error) error {
		stopErr := stopSupervisedProcess(handle, done)
		var guardianErr error
		if stopErr == nil {
			guardianErr = guardian.finishCleanly()
		} else {
			guardianErr = guardian.terminateRoot()
		}
		return errors.Join(reason, stopErr, guardianErr)
	}

	state.Instance = cfg.Token
	state.Control = cfg.Control
	state.Port = cfg.Port
	state.SupervisorPID = os.Getpid()
	state.SupervisorStarted = rerankerProcessStartIdentity(state.SupervisorPID)
	if guardian != nil {
		state.GuardianPID = guardian.process.Pid
	}
	if state.SupervisorStarted == "" {
		return stopAndFinish(fmt.Errorf("identify reranker supervisor process"))
	}
	manager := &RerankerManager{sgrepHome: cfg.Home, port: cfg.Port, host: DefaultHost}
	if cfg.BeforePublish != nil {
		cfg.BeforePublish()
	}
	if err := manager.writePIDState(state); err != nil {
		return stopAndFinish(fmt.Errorf("publish supervised reranker state: %w", err))
	}
	if lifecycleLock != nil {
		if err := lifecycleLock.Close(); err != nil {
			return stopAndFinish(fmt.Errorf("release inherited reranker lifecycle lock: %w", err))
		}
		lifecycleLock = nil
	}
	defer manager.removePIDFileIfOwned(state)

	commands, acceptErrors, stopServing := serveSupervisorConnections(listener, cfg.Token)
	defer stopServing()
	adoptionTimeout := cfg.AdoptionTimeout
	if adoptionTimeout <= 0 {
		adoptionTimeout = supervisorAdoptionTimeout
	}
	adoptionTimer := time.NewTimer(adoptionTimeout)
	defer adoptionTimer.Stop()
	adoptionDeadline := adoptionTimer.C
	adoptionLease := cfg.AdoptionLease
	if adoptionLease <= 0 {
		adoptionLease = RerankerStartupTimeout + 5*time.Second
	}

	for {
		select {
		case childErr := <-done:
			return errors.Join(childErr, guardian.finishCleanly())
		case <-signals:
			return stopAndFinish(nil)
		case guardianErr := <-guardianDone:
			_ = guardian.lifetime.Close()
			if guardianErr == nil {
				guardianErr = fmt.Errorf("reranker lifetime guardian exited unexpectedly")
			} else {
				guardianErr = fmt.Errorf("reranker lifetime guardian exited: %w", guardianErr)
			}
			return errors.Join(guardianErr, stopSupervisedProcess(handle, done))
		case err := <-acceptErrors:
			return stopAndFinish(fmt.Errorf("accept reranker control request: %w", err))
		case <-adoptionDeadline:
			return stopAndFinish(fmt.Errorf("reranker supervisor was not adopted"))
		case command := <-commands:
			response := supervisorResponse{Protocol: supervisorProtocolVersion, PID: state.PID, Running: true}
			switch command.action {
			case "adopt":
				if !adoptionTimer.Stop() {
					select {
					case <-adoptionTimer.C:
					default:
					}
				}
				adoptionTimer.Reset(adoptionLease)
				adoptionDeadline = adoptionTimer.C
			case "commit":
				if adoptionDeadline != nil {
					if !adoptionTimer.Stop() {
						select {
						case <-adoptionTimer.C:
						default:
						}
					}
					adoptionDeadline = nil
				}
			case "status":
			case "stop":
				response.Error = errorString(stopAndFinish(nil))
				response.Running = false
			default:
				response.Error = "unsupported control action"
			}
			command.response <- response
			if command.action == "stop" {
				select {
				case <-command.written:
				case <-time.After(time.Second):
				}
				var stopErr error
				if response.Error != "" {
					stopErr = errors.New(response.Error)
				}
				return stopErr
			}
		}
	}
}

func listenSupervisorSocket(control string) (*net.UnixListener, os.FileInfo, error) {
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: control, Net: "unix"})
	if err != nil {
		return nil, nil, fmt.Errorf("listen on reranker control socket: %w", err)
	}
	listener.SetUnlinkOnClose(false)
	if err := os.Chmod(control, 0600); err != nil {
		_ = listener.Close()
		_ = os.Remove(control)
		return nil, nil, fmt.Errorf("secure reranker control socket: %w", err)
	}
	info, err := os.Lstat(control)
	if err != nil {
		_ = listener.Close()
		_ = os.Remove(control)
		return nil, nil, fmt.Errorf("inspect reranker control socket: %w", err)
	}
	return listener, info, nil
}

func cleanupSupervisorSocket(listener *net.UnixListener, control string, original os.FileInfo) {
	_ = listener.Close()
	current, err := os.Lstat(control)
	if err == nil && current.Mode()&os.ModeSocket != 0 && os.SameFile(original, current) {
		_ = os.Remove(control)
	}
	_ = os.Remove(filepath.Dir(control))
}

func serveSupervisorConnections(listener *net.UnixListener, token string) (<-chan supervisorCommand, <-chan error, func()) {
	commands := make(chan supervisorCommand)
	errorsOut := make(chan error, 1)
	done := make(chan struct{})
	clients := make(chan struct{}, supervisorMaxClients)
	go func() {
		for {
			conn, err := listener.AcceptUnix()
			if err != nil {
				select {
				case errorsOut <- err:
				case <-done:
				}
				return
			}
			select {
			case clients <- struct{}{}:
				go func() {
					defer func() { <-clients }()
					handleSupervisorConnection(conn, token, commands, done)
				}()
			default:
				_ = conn.Close()
			}
		}
	}()
	var once sync.Once
	stop := func() {
		once.Do(func() {
			close(done)
			_ = listener.Close()
		})
	}
	return commands, errorsOut, stop
}

func handleSupervisorConnection(conn *net.UnixConn, token string, commands chan<- supervisorCommand, done <-chan struct{}) {
	defer func() { _ = conn.Close() }()
	_ = conn.SetReadDeadline(time.Now().Add(supervisorReadTimeout))
	var request supervisorRequest
	if err := json.NewDecoder(io.LimitReader(conn, supervisorRequestLimit)).Decode(&request); err != nil {
		return
	}
	wantProof := supervisorMAC(token, "client", strconv.Itoa(request.Protocol), request.Nonce, request.Action)
	if request.Protocol != supervisorProtocolVersion || subtle.ConstantTimeCompare([]byte(request.Proof), []byte(wantProof)) != 1 {
		return
	}
	_ = conn.SetDeadline(time.Now().Add(3 * time.Second))
	command := supervisorCommand{
		action:   request.Action,
		response: make(chan supervisorResponse, 1),
		written:  make(chan struct{}),
	}
	select {
	case commands <- command:
	case <-done:
		return
	}
	var response supervisorResponse
	select {
	case response = <-command.response:
	case <-done:
		return
	}
	response.Protocol = supervisorProtocolVersion
	response.Proof = supervisorMAC(token, "server", strconv.Itoa(response.Protocol), request.Nonce, request.Action, strconv.Itoa(response.PID), strconv.FormatBool(response.Running), response.Error)
	_ = json.NewEncoder(conn).Encode(response)
	close(command.written)
}

func requestRerankerSupervisor(state rerankerPIDState, action string) (supervisorResponse, error) {
	if state.Protocol != supervisorProtocolVersion {
		return supervisorResponse{}, fmt.Errorf("reranker supervisor protocol %d is incompatible with %d", state.Protocol, supervisorProtocolVersion)
	}
	conn, err := net.DialTimeout("unix", state.Control, 500*time.Millisecond)
	if err != nil {
		return supervisorResponse{}, err
	}
	unixConn, ok := conn.(*net.UnixConn)
	if !ok {
		_ = conn.Close()
		return supervisorResponse{}, fmt.Errorf("reranker control connection is not Unix")
	}
	defer func() { _ = unixConn.Close() }()
	peer, err := supervisorPeerPID(unixConn)
	if err != nil || peer != state.SupervisorPID || rerankerProcessStartIdentity(peer) != state.SupervisorStarted {
		return supervisorResponse{}, fmt.Errorf("reranker supervisor peer identity mismatch")
	}

	nonce, err := newSupervisorNonce()
	if err != nil {
		return supervisorResponse{}, err
	}
	request := supervisorRequest{
		Protocol: supervisorProtocolVersion,
		Nonce:    nonce,
		Action:   action,
		Proof:    supervisorMAC(state.Instance, "client", strconv.Itoa(supervisorProtocolVersion), nonce, action),
	}
	_ = unixConn.SetDeadline(time.Now().Add(3 * time.Second))
	if err := json.NewEncoder(unixConn).Encode(request); err != nil {
		return supervisorResponse{}, err
	}
	var response supervisorResponse
	if err := json.NewDecoder(io.LimitReader(unixConn, supervisorRequestLimit)).Decode(&response); err != nil {
		return supervisorResponse{}, err
	}
	wantProof := supervisorMAC(state.Instance, "server", strconv.Itoa(response.Protocol), nonce, action, strconv.Itoa(response.PID), strconv.FormatBool(response.Running), response.Error)
	if response.Protocol != supervisorProtocolVersion || subtle.ConstantTimeCompare([]byte(response.Proof), []byte(wantProof)) != 1 {
		return supervisorResponse{}, fmt.Errorf("reranker supervisor response authentication failed")
	}
	if response.Error != "" {
		return response, errors.New(response.Error)
	}
	return response, nil
}

func supervisorMAC(token string, fields ...string) string {
	mac := hmac.New(sha256.New, []byte(token))
	for _, field := range fields {
		_, _ = mac.Write([]byte(field))
		_, _ = mac.Write([]byte{0})
	}
	return hex.EncodeToString(mac.Sum(nil))
}

func newSupervisorNonce() (string, error) {
	var nonce [32]byte
	if _, err := rand.Read(nonce[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(nonce[:]), nil
}

func closeInheritedDescriptors(kept ...int) error {
	keep := make(map[int]struct{}, len(kept)+3)
	keep[0] = struct{}{}
	keep[1] = struct{}{}
	keep[2] = struct{}{}
	for _, descriptor := range kept {
		keep[descriptor] = struct{}{}
	}
	directory := "/dev/fd"
	if runtime.GOOS == "linux" {
		directory = "/proc/self/fd"
	}
	entries, err := os.ReadDir(directory)
	if err != nil {
		return fmt.Errorf("enumerate inherited descriptors: %w", err)
	}
	for _, entry := range entries {
		descriptor, err := strconv.Atoi(entry.Name())
		if err != nil || descriptor < 3 {
			continue
		}
		if _, ok := keep[descriptor]; ok {
			continue
		}
		flags, err := unix.FcntlInt(uintptr(descriptor), unix.F_GETFD, 0)
		if errors.Is(err, unix.EBADF) {
			continue
		}
		if err != nil {
			return fmt.Errorf("inspect inherited descriptor %d: %w", descriptor, err)
		}
		if flags&unix.FD_CLOEXEC != 0 {
			continue
		}
		if err := unix.Close(descriptor); err != nil && !errors.Is(err, unix.EBADF) {
			return fmt.Errorf("close inherited descriptor %d: %w", descriptor, err)
		}
	}
	return nil
}

func stopSupervisedProcess(process *rerankerProcessHandle, done <-chan error) error {
	if err := process.Signal(syscall.SIGTERM); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("terminate reranker process: %w", err)
	}
	timer := time.NewTimer(supervisorStopTimeout)
	defer timer.Stop()
	select {
	case <-done:
		return nil
	case <-timer.C:
	}
	if err := process.Signal(syscall.SIGKILL); err != nil && !errors.Is(err, os.ErrProcessDone) {
		return fmt.Errorf("kill reranker process: %w", err)
	}
	select {
	case <-done:
		return nil
	case <-time.After(2 * time.Second):
		return fmt.Errorf("reranker process did not exit after SIGKILL")
	}
}

func environmentForSupervisor(environment []string, home, control, token string, port int) []string {
	result := withoutSupervisorEnvironment(environment)
	return append(result,
		supervisorModeEnv+"=1",
		supervisorProtocolEnv+"="+strconv.Itoa(supervisorProtocolVersion),
		supervisorHomeEnv+"="+home,
		supervisorControlEnv+"="+control,
		supervisorTokenEnv+"="+token,
		supervisorPortEnv+"="+strconv.Itoa(port),
	)
}

func environmentForGuardian(environment []string) []string {
	result := withoutSupervisorEnvironment(environment)
	return append(result,
		supervisorModeEnv+"=1",
		supervisorProtocolEnv+"="+strconv.Itoa(supervisorProtocolVersion),
	)
}

func environmentForRunner(environment []string, instance string) []string {
	result := environmentForReranker(environment, instance)
	return append(result,
		supervisorModeEnv+"=1",
		supervisorProtocolEnv+"="+strconv.Itoa(supervisorProtocolVersion),
	)
}

func environmentForReranker(environment []string, instance string) []string {
	result := withoutSupervisorEnvironment(environment)
	prefix := rerankerInstanceEnv + "="
	filtered := result[:0]
	for _, entry := range result {
		if !strings.HasPrefix(entry, prefix) {
			filtered = append(filtered, entry)
		}
	}
	return append(filtered, prefix+instance)
}

func withoutSupervisorEnvironment(environment []string) []string {
	result := make([]string, 0, len(environment))
	prefix := "SGREP_RERANKER_SUPERVISOR_"
	for _, entry := range environment {
		if !strings.HasPrefix(entry, prefix) {
			result = append(result, entry)
		}
	}
	return result
}

func errorString(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}
