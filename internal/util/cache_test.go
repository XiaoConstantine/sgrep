package util

import (
	"sync"
	"testing"
	"time"
)

func TestNewQueryCache(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)

	if cache.capacity != 100 {
		t.Errorf("expected capacity 100, got %d", cache.capacity)
	}
	if cache.ttl != 5*time.Minute {
		t.Errorf("expected TTL 5m, got %v", cache.ttl)
	}
}

func TestQueryCache_SetAndGet(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)

	cache.Set("key1", "value1")
	cache.Set("key2", 42)

	v1 := cache.Get("key1")
	if v1 != "value1" {
		t.Errorf("expected 'value1', got %v", v1)
	}

	v2 := cache.Get("key2")
	if v2 != 42 {
		t.Errorf("expected 42, got %v", v2)
	}

	v3 := cache.Get("nonexistent")
	if v3 != nil {
		t.Errorf("expected nil for nonexistent key, got %v", v3)
	}
}

func TestQueryCache_Update(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)

	cache.Set("key1", "original")
	cache.Set("key1", "updated")

	v := cache.Get("key1")
	if v != "updated" {
		t.Errorf("expected 'updated', got %v", v)
	}

	if cache.Len() != 1 {
		t.Errorf("expected length 1 after update, got %d", cache.Len())
	}
}

func TestQueryCache_TTLExpiration(t *testing.T) {
	cache := NewQueryCache(100, 50*time.Millisecond)

	cache.Set("key1", "value1")

	v := cache.Get("key1")
	if v != "value1" {
		t.Errorf("expected 'value1' before expiration, got %v", v)
	}

	time.Sleep(60 * time.Millisecond)

	v = cache.Get("key1")
	if v != nil {
		t.Errorf("expected nil after expiration, got %v", v)
	}

	if cache.Len() != 0 {
		t.Errorf("expected length 0 after expiration, got %d", cache.Len())
	}
}

func TestQueryCache_LRUEviction(t *testing.T) {
	cache := NewQueryCache(3, 5*time.Minute)

	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")

	if cache.Len() != 3 {
		t.Errorf("expected length 3, got %d", cache.Len())
	}

	cache.Set("key4", "value4")

	if cache.Len() != 3 {
		t.Errorf("expected length 3 after eviction, got %d", cache.Len())
	}

	if cache.Get("key1") != nil {
		t.Error("expected key1 to be evicted (LRU)")
	}

	if cache.Get("key4") != "value4" {
		t.Error("expected key4 to be present")
	}
}

func TestQueryCache_LRUReordering(t *testing.T) {
	cache := NewQueryCache(3, 5*time.Minute)

	cache.Set("key1", "value1")
	cache.Set("key2", "value2")
	cache.Set("key3", "value3")

	cache.Get("key1")

	cache.Set("key4", "value4")

	if cache.Get("key1") == nil {
		t.Error("key1 should still exist after being accessed")
	}

	if cache.Get("key2") != nil {
		t.Error("key2 should be evicted (least recently used)")
	}
}

func TestQueryCache_Clear(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)

	cache.Set("key1", "value1")
	cache.Set("key2", "value2")

	cache.Clear()

	if cache.Len() != 0 {
		t.Errorf("expected length 0 after clear, got %d", cache.Len())
	}

	if cache.Get("key1") != nil {
		t.Error("expected nil after clear")
	}
}

func TestQueryCache_Stats(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)

	cache.Set("key1", "value1")
	cache.Set("key2", "value2")

	stats := cache.Stats()
	if stats.Size != 2 {
		t.Errorf("expected size 2, got %d", stats.Size)
	}
	if stats.Capacity != 100 {
		t.Errorf("expected capacity 100, got %d", stats.Capacity)
	}
}

func TestQueryCache_Concurrent(t *testing.T) {
	cache := NewQueryCache(100, 5*time.Minute)
	var wg sync.WaitGroup

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := "key" + string(rune('0'+i%10))
			cache.Set(key, i)
		}(i)
	}

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			key := "key" + string(rune('0'+i%10))
			cache.Get(key)
		}(i)
	}

	wg.Wait()
}

func TestQueryCache_ConcurrentReadWrite(t *testing.T) {
	cache := NewQueryCache(1000, 5*time.Minute)
	done := make(chan bool)

	go func() {
		for i := 0; i < 1000; i++ {
			cache.Set("key", i)
		}
		done <- true
	}()

	go func() {
		for i := 0; i < 1000; i++ {
			cache.Get("key")
		}
		done <- true
	}()

	<-done
	<-done
}

// Benchmarks

func BenchmarkQueryCache_Set(b *testing.B) {
	cache := NewQueryCache(10000, 5*time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Set("key", "value")
	}
}

func BenchmarkQueryCache_Get_Hit(b *testing.B) {
	cache := NewQueryCache(10000, 5*time.Minute)
	cache.Set("key", "value")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get("key")
	}
}

func BenchmarkQueryCache_Get_Miss(b *testing.B) {
	cache := NewQueryCache(10000, 5*time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get("nonexistent")
	}
}

func BenchmarkQueryCache_Concurrent(b *testing.B) {
	cache := NewQueryCache(10000, 5*time.Minute)

	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%2 == 0 {
				cache.Set("key", i)
			} else {
				cache.Get("key")
			}
			i++
		}
	})
}
