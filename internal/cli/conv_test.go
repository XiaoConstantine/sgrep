package cli

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/XiaoConstantine/sgrep/pkg/conv"
	"github.com/spf13/cobra"
)

func TestConvViewAcceptsNoColorFlag(t *testing.T) {
	if convViewCmd.Flags().Lookup("no-color") == nil {
		t.Fatal("conv view is missing --no-color flag")
	}
}

func TestConvRecallCommandContract(t *testing.T) {
	if convRecallCmd.Flags().Lookup("max-bytes") == nil || convRecallCmd.Flags().Lookup("cwd") == nil {
		t.Fatal("conv recall is missing agent-facing flags")
	}
	oldMaxBytes := convRecallMaxBytes
	convRecallMaxBytes = conv.DefaultRecallMaxBytes
	defer func() { convRecallMaxBytes = oldMaxBytes }()
	var output bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&output)
	if err := runConvRecall(cmd, nil); err != nil {
		t.Fatal(err)
	}
	var response conv.RecallResponse
	if err := json.Unmarshal(output.Bytes(), &response); err != nil {
		t.Fatalf("arity error was not JSON: %v", err)
	}
	if response.Status != conv.RecallInvalidRequest {
		t.Fatalf("arity status = %q, want invalid_request", response.Status)
	}
}

func TestEncodeRecallResponseWritesOneBudgetedJSONDocument(t *testing.T) {
	response := &conv.RecallResponse{
		Schema:        conv.RecallSchema,
		Status:        conv.RecallNotReady,
		Query:         "prior work",
		RetrievalMode: "hybrid",
		Budget:        conv.RecallBudget{MaxBytes: conv.DefaultRecallMaxBytes},
		Sessions:      []conv.RecallSession{},
		Warnings:      []conv.RecallWarning{},
	}
	var output bytes.Buffer
	cmd := &cobra.Command{}
	cmd.SetOut(&output)
	if err := encodeRecallResponse(cmd, response); err != nil {
		t.Fatal(err)
	}
	var decoded conv.RecallResponse
	if err := json.Unmarshal(output.Bytes(), &decoded); err != nil {
		t.Fatalf("recall output is not one JSON document: %v\n%s", err, output.String())
	}
	if decoded.Budget.UsedBytes != output.Len() {
		t.Fatalf("reported %d bytes, wrote %d", decoded.Budget.UsedBytes, output.Len())
	}
}
