//go:build darwin

package rerank

import (
	"fmt"
	"os"
)

// The protected process group is the guardian's fallback when exec makes a
// previously captured Darwin audit token stale. posix_spawn is denied because
// SETEXEC can otherwise change the group atomically without forking.
const rerankerSandboxProfile = "(version 1) (allow default) (deny process-fork) (deny syscall-unix (syscall-number SYS_setpgid SYS_setsid SYS_posix_spawn))"

func preserveRerankerIdentityAfterExec() bool {
	// Darwin has no pidfd. Keeping the private socket open across exec lets
	// the supervisor obtain the audit token while the new image is ptrace-
	// stopped, then transfer that generation-bound handle to the guardian.
	return true
}

func prepareConstrainedReranker(executable string, args []string) (string, []string, error) {
	const sandboxExecutable = "/usr/bin/sandbox-exec"
	info, err := os.Stat(sandboxExecutable)
	if err != nil || !info.Mode().IsRegular() || info.Mode().Perm()&0111 == 0 {
		return "", nil, fmt.Errorf("reranker process confinement is unavailable: %w", err)
	}
	commandArgs := []string{sandboxExecutable, "-p", rerankerSandboxProfile, executable}
	commandArgs = append(commandArgs, args...)
	return sandboxExecutable, commandArgs, nil
}
