package util

import (
	"container/list"
	"sync"
	"time"
)

// QueryCache is an LRU cache with TTL for search results.
// Avoids redundant LLM calls for repeated queries.
type QueryCache struct {
	mu       sync.RWMutex
	capacity int
	ttl      time.Duration
	items    map[string]*list.Element
	order    *list.List
}

type cacheEntry struct {
	key       string
	value     interface{}
	expiresAt time.Time
}

// NewQueryCache creates an LRU cache with given capacity and TTL.
func NewQueryCache(capacity int, ttl time.Duration) *QueryCache {
	return &QueryCache{
		capacity: capacity,
		ttl:      ttl,
		items:    make(map[string]*list.Element),
		order:    list.New(),
	}
}

// Get retrieves a value from cache, returns nil if not found or expired.
func (c *QueryCache) Get(key string) interface{} {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.items[key]
	if !ok {
		return nil
	}

	entry := elem.Value.(*cacheEntry)
	if time.Now().After(entry.expiresAt) {
		c.removeElement(elem)
		return nil
	}

	// Move to front (most recently used)
	c.order.MoveToFront(elem)

	return entry.value
}

// Set stores a value in cache.
func (c *QueryCache) Set(key string, value interface{}) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Update existing
	if elem, ok := c.items[key]; ok {
		entry := elem.Value.(*cacheEntry)
		entry.value = value
		entry.expiresAt = time.Now().Add(c.ttl)
		c.order.MoveToFront(elem)
		return
	}

	// Evict oldest if at capacity
	if c.order.Len() >= c.capacity {
		oldest := c.order.Back()
		if oldest != nil {
			c.removeElement(oldest)
		}
	}

	// Add new entry
	entry := &cacheEntry{
		key:       key,
		value:     value,
		expiresAt: time.Now().Add(c.ttl),
	}
	elem := c.order.PushFront(entry)
	c.items[key] = elem
}

// removeElement removes an element from both map and list.
func (c *QueryCache) removeElement(elem *list.Element) {
	entry := elem.Value.(*cacheEntry)
	delete(c.items, entry.key)
	c.order.Remove(elem)
}

// Clear removes all entries.
func (c *QueryCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*list.Element)
	c.order.Init()
}

// Len returns number of cached items.
func (c *QueryCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.items)
}

// Stats returns cache statistics.
type CacheStats struct {
	Size     int
	Capacity int
}

func (c *QueryCache) Stats() CacheStats {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return CacheStats{
		Size:     len(c.items),
		Capacity: c.capacity,
	}
}
