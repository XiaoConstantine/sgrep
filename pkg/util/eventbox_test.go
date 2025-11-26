package util

import (
	"sync"
	"testing"
	"time"
)

func TestNewEventBox(t *testing.T) {
	eb := NewEventBox()

	if eb.events == nil {
		t.Error("events map should be initialized")
	}
	if eb.cond == nil {
		t.Error("cond should be initialized")
	}
	if eb.ignore == nil {
		t.Error("ignore map should be initialized")
	}
}

func TestEventBox_SetAndPeek(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerStarting, "starting")

	data, ok := eb.Peek(EvtServerStarting)
	if !ok {
		t.Error("expected event to be set")
	}
	if data != "starting" {
		t.Errorf("expected 'starting', got %v", data)
	}

	_, ok = eb.Peek(EvtServerReady)
	if ok {
		t.Error("expected EvtServerReady to not be set")
	}
}

func TestEventBox_Clear(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerStarting, nil)
	eb.Clear(EvtServerStarting)

	_, ok := eb.Peek(EvtServerStarting)
	if ok {
		t.Error("expected event to be cleared")
	}
}

func TestEventBox_Reset(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerStarting, nil)
	eb.Set(EvtServerReady, nil)
	eb.Set(EvtSearchStart, nil)

	eb.Reset()

	if len(eb.Events()) != 0 {
		t.Errorf("expected 0 events after reset, got %d", len(eb.Events()))
	}
}

func TestEventBox_Events(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerStarting, "data1")
	eb.Set(EvtSearchStart, "data2")

	events := eb.Events()

	if len(events) != 2 {
		t.Errorf("expected 2 events, got %d", len(events))
	}
	if events[EvtServerStarting] != "data1" {
		t.Errorf("expected 'data1' for EvtServerStarting")
	}
	if events[EvtSearchStart] != "data2" {
		t.Errorf("expected 'data2' for EvtSearchStart")
	}
}

func TestEventBox_Ignore(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerStarting, "before")
	eb.Ignore(EvtServerStarting)

	_, ok := eb.Peek(EvtServerStarting)
	if ok {
		t.Error("ignored event should be cleared")
	}

	eb.Set(EvtServerStarting, "after")
	_, ok = eb.Peek(EvtServerStarting)
	if ok {
		t.Error("ignored event should not be set")
	}
}

func TestEventBox_Unignore(t *testing.T) {
	eb := NewEventBox()

	eb.Ignore(EvtServerStarting)
	eb.Unignore(EvtServerStarting)

	eb.Set(EvtServerStarting, "data")
	_, ok := eb.Peek(EvtServerStarting)
	if !ok {
		t.Error("unignored event should be settable")
	}
}

func TestEventBox_Wait(t *testing.T) {
	eb := NewEventBox()
	done := make(chan bool)

	go func() {
		evtType, data := eb.Wait(EvtServerReady, EvtServerError)
		if evtType != EvtServerReady {
			t.Errorf("expected EvtServerReady, got %v", evtType)
		}
		if data != "ready" {
			t.Errorf("expected 'ready', got %v", data)
		}
		done <- true
	}()

	time.Sleep(10 * time.Millisecond)
	eb.Set(EvtServerReady, "ready")

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("Wait should have returned")
	}
}

func TestEventBox_Wait_AlreadySet(t *testing.T) {
	eb := NewEventBox()

	eb.Set(EvtServerReady, "ready")

	evtType, data := eb.Wait(EvtServerReady)
	if evtType != EvtServerReady {
		t.Errorf("expected EvtServerReady, got %v", evtType)
	}
	if data != "ready" {
		t.Errorf("expected 'ready', got %v", data)
	}
}

func TestEventBox_WaitFor(t *testing.T) {
	eb := NewEventBox()
	done := make(chan bool)

	go func() {
		data := eb.WaitFor(EvtSearchComplete)
		if data != 10 {
			t.Errorf("expected 10, got %v", data)
		}
		done <- true
	}()

	time.Sleep(10 * time.Millisecond)
	eb.Set(EvtSearchComplete, 10)

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("WaitFor should have returned")
	}
}

func TestEventBox_Channel(t *testing.T) {
	eb := NewEventBox()
	ch := eb.Channel(10)

	if cap(ch) != 10 {
		t.Errorf("expected channel capacity 10, got %d", cap(ch))
	}
}

func TestEventBox_Concurrent(t *testing.T) {
	eb := NewEventBox()
	var wg sync.WaitGroup

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			eb.Set(EvtSearchProgress, i)
		}(i)
	}

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			eb.Peek(EvtSearchProgress)
		}()
	}

	wg.Wait()
}

func TestEventBox_ConcurrentWait(t *testing.T) {
	eb := NewEventBox()
	var wg sync.WaitGroup

	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			eb.Wait(EvtServerReady, EvtServerError)
		}()
	}

	time.Sleep(10 * time.Millisecond)
	eb.Set(EvtServerReady, nil)

	done := make(chan bool)
	go func() {
		wg.Wait()
		done <- true
	}()

	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("all waiters should have been unblocked")
	}
}

func TestEventTypes(t *testing.T) {
	eventTypes := []EventType{
		EvtServerStarting,
		EvtServerReady,
		EvtServerStopped,
		EvtServerError,
		EvtSearchStart,
		EvtSearchProgress,
		EvtSearchComplete,
		EvtIndexStart,
		EvtIndexProgress,
		EvtIndexComplete,
	}

	for i, et := range eventTypes {
		if et != EventType(i) {
			t.Errorf("EventType %d has unexpected value %d", i, et)
		}
	}
}

// Benchmarks

func BenchmarkEventBox_Set(b *testing.B) {
	eb := NewEventBox()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eb.Set(EvtSearchProgress, i)
	}
}

func BenchmarkEventBox_Peek(b *testing.B) {
	eb := NewEventBox()
	eb.Set(EvtSearchProgress, 42)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eb.Peek(EvtSearchProgress)
	}
}

func BenchmarkEventBox_Concurrent(b *testing.B) {
	eb := NewEventBox()

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				eb.Set(EvtSearchProgress, i)
			} else {
				eb.Peek(EvtSearchProgress)
			}
			i++
		}
	})
}
