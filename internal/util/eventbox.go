package util

import (
	"sync"
)

// EventType represents different event types in the system.
type EventType int

const (
	EvtServerStarting EventType = iota
	EvtServerReady
	EvtServerStopped
	EvtServerError
	EvtSearchStart
	EvtSearchProgress
	EvtSearchComplete
	EvtIndexStart
	EvtIndexProgress
	EvtIndexComplete
)

// Event represents an event with optional data.
type Event struct {
	Type EventType
	Data interface{}
}

// EventBox is a thread-safe event coordination mechanism (inspired by fzf).
// Components communicate through events without tight coupling.
type EventBox struct {
	events map[EventType]interface{}
	cond   *sync.Cond
	ignore map[EventType]bool
}

// NewEventBox creates a new event box.
func NewEventBox() *EventBox {
	return &EventBox{
		events: make(map[EventType]interface{}),
		cond:   sync.NewCond(&sync.Mutex{}),
		ignore: make(map[EventType]bool),
	}
}

// Set sets an event with optional data.
func (b *EventBox) Set(event EventType, data interface{}) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()

	if b.ignore[event] {
		return
	}

	b.events[event] = data
	b.cond.Broadcast()
}

// Clear removes an event.
func (b *EventBox) Clear(event EventType) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()
	delete(b.events, event)
}

// Peek checks if an event is set without blocking.
func (b *EventBox) Peek(event EventType) (interface{}, bool) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()
	data, ok := b.events[event]
	return data, ok
}

// Wait blocks until any of the specified events is set.
// Returns the first event type that was set and its data.
func (b *EventBox) Wait(events ...EventType) (EventType, interface{}) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()

	for {
		for _, event := range events {
			if data, ok := b.events[event]; ok {
				return event, data
			}
		}
		b.cond.Wait()
	}
}

// WaitFor blocks until the specific event is set and returns its data.
func (b *EventBox) WaitFor(event EventType) interface{} {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()

	for {
		if data, ok := b.events[event]; ok {
			return data
		}
		b.cond.Wait()
	}
}

// Channel returns a channel that receives events.
// Useful for select-based event handling.
func (b *EventBox) Channel(bufSize int) chan Event {
	ch := make(chan Event, bufSize)
	return ch
}

// Ignore marks an event type to be ignored.
func (b *EventBox) Ignore(event EventType) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()
	b.ignore[event] = true
	delete(b.events, event)
}

// Unignore removes an event type from the ignore list.
func (b *EventBox) Unignore(event EventType) {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()
	delete(b.ignore, event)
}

// Reset clears all events.
func (b *EventBox) Reset() {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()
	b.events = make(map[EventType]interface{})
}

// Events returns a snapshot of all current events.
func (b *EventBox) Events() map[EventType]interface{} {
	b.cond.L.Lock()
	defer b.cond.L.Unlock()

	snapshot := make(map[EventType]interface{}, len(b.events))
	for k, v := range b.events {
		snapshot[k] = v
	}
	return snapshot
}
