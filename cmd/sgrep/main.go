package main

import (
	"fmt"
	"os"

	"github.com/XiaoConstantine/sgrep/internal/cli"
	"github.com/XiaoConstantine/sgrep/pkg/rerank"
)

func main() {
	if handled, err := rerank.RunSupervisorCommand(os.Args[1:]); handled {
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		return
	}
	if err := cli.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
