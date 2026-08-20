//go:build darwin

package rerank

import (
	"fmt"
	"os"
)

const rerankerSandboxProfile = "(version 1) (allow default) (deny process-fork)"

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
