package cli

import "testing"

func TestConvViewAcceptsNoColorFlag(t *testing.T) {
	if convViewCmd.Flags().Lookup("no-color") == nil {
		t.Fatal("conv view is missing --no-color flag")
	}
}
