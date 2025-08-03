"""In-memory cache backend implementation."""

import time
import threading
from typing import Any, Dict, Optional
from dataclasses import dataclass
import logging

from .manager import CacheBackend


logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with value and expiration."""
    value: bytes
    expires_at: Optional[float] = None
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


class MemoryCache(CacheBackend):
    """In-memory cache backend with TTL support."""
    
    def __init__(self, max_size: int = 10000, cleanup_interval: int = 60):
        self._store: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self.max_size = max_size
        self.cleanup_interval = cleanup_interval
        self._stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'evictions': 0
        }
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        
        logger.info(f"MemoryCache initialized with max_size={max_size}")
    
    def _cleanup_loop(self) -> None:
        """Background cleanup loop to remove expired entries."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries."""
        removed_count = 0
        
        with self._lock:
            expired_keys = [
                key for key, entry in self._store.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                del self._store[key]
                removed_count += 1
        
        if removed_count > 0:
            logger.debug(f"Cleaned up {removed_count} expired cache entries")
        
        return removed_count
    
    def _evict_if_needed(self) -> None:
        """Evict entries if cache is at capacity."""
        if len(self._store) >= self.max_size:
            # Simple LRU: remove oldest entries
            entries_to_remove = len(self._store) - self.max_size + 1
            
            # Sort by creation time and remove oldest
            sorted_keys = sorted(
                self._store.keys(),
                key=lambda k: self._store[k].created_at
            )
            
            for key in sorted_keys[:entries_to_remove]:
                del self._store[key]
                self._stats['evictions'] += 1
            
            logger.debug(f"Evicted {entries_to_remove} cache entries due to size limit")
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from memory cache."""
        with self._lock:
            entry = self._store.get(key)
            
            if entry is None:
                self._stats['misses'] += 1
                return None
            
            if entry.is_expired():
                del self._store[key]
                self._stats['misses'] += 1
                return None
            
            self._stats['hits'] += 1
            return entry.value
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in memory cache."""
        try:
            with self._lock:
                expires_at = None
                if ttl is not None:
                    expires_at = time.time() + ttl
                
                entry = CacheEntry(value=value, expires_at=expires_at)
                
                # Check if we need to evict
                if key not in self._store:
                    self._evict_if_needed()
                
                self._store[key] = entry
                self._stats['sets'] += 1
                
            return True
            
        except Exception as e:
            logger.error(f"Memory cache set failed for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from memory cache."""
        try:
            with self._lock:
                if key in self._store:
                    del self._store[key]
                    self._stats['deletes'] += 1
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Memory cache delete failed for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        with self._lock:
            entry = self._store.get(key)
            
            if entry is None:
                return False
            
            if entry.is_expired():
                del self._store[key]
                return False
            
            return True
    
    def clear(self) -> bool:
        """Clear all entries from memory cache."""
        try:
            with self._lock:
                self._store.clear()
                logger.info("Memory cache cleared")
            return True
            
        except Exception as e:
            logger.error(f"Memory cache clear failed: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern (simplified glob matching)."""
        try:
            import fnmatch
            
            removed_count = 0
            with self._lock:
                keys_to_remove = [
                    key for key in self._store.keys()
                    if fnmatch.fnmatch(key, pattern)
                ]
                
                for key in keys_to_remove:
                    del self._store[key]
                    removed_count += 1
                    
                self._stats['deletes'] += removed_count
            
            return removed_count
            
        except Exception as e:
            logger.error(f"Memory cache pattern delete failed for pattern {pattern}: {e}")
            return 0
    
    def get_multiple(self, keys: list) -> Dict[str, Optional[bytes]]:
        """Get multiple values from memory cache."""
        result = {}
        
        with self._lock:
            for key in keys:
                entry = self._store.get(key)
                
                if entry is None or entry.is_expired():
                    if entry is not None and entry.is_expired():
                        del self._store[key]
                    result[key] = None
                    self._stats['misses'] += 1
                else:
                    result[key] = entry.value
                    self._stats['hits'] += 1
        
        return result
    
    def set_multiple(self, mapping: Dict[str, bytes], ttl: Optional[int] = None) -> bool:
        """Set multiple values in memory cache."""
        try:
            with self._lock:
                expires_at = None
                if ttl is not None:
                    expires_at = time.time() + ttl
                
                for key, value in mapping.items():
                    entry = CacheEntry(value=value, expires_at=expires_at)
                    
                    # Check if we need to evict
                    if key not in self._store:
                        self._evict_if_needed()
                    
                    self._store[key] = entry
                    self._stats['sets'] += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Memory cache mset failed: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in memory cache."""
        try:
            with self._lock:
                entry = self._store.get(key)
                
                if entry is None or entry.is_expired():
                    # Initialize counter
                    new_value = amount
                    if entry is not None and entry.is_expired():
                        del self._store[key]
                else:
                    # Increment existing value
                    try:
                        current_value = int(entry.value.decode('utf-8'))
                        new_value = current_value + amount
                    except (ValueError, UnicodeDecodeError):
                        # Not a valid integer, start from amount
                        new_value = amount
                
                # Store new value
                new_entry = CacheEntry(value=str(new_value).encode('utf-8'))
                self._store[key] = new_entry
                self._stats['sets'] += 1
                
                return new_value
                
        except Exception as e:
            logger.error(f"Memory cache increment failed for key {key}: {e}")
            return None
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            with self._lock:
                entry = self._store.get(key)
                
                if entry is None or entry.is_expired():
                    return False
                
                # Update expiration time
                entry.expires_at = time.time() + ttl
                return True
                
        except Exception as e:
            logger.error(f"Memory cache expire failed for key {key}: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get TTL for key."""
        try:
            with self._lock:
                entry = self._store.get(key)
                
                if entry is None:
                    return -2  # Key doesn't exist
                
                if entry.is_expired():
                    del self._store[key]
                    return -2
                
                if entry.expires_at is None:
                    return -1  # No expiration
                
                remaining = int(entry.expires_at - time.time())
                return max(0, remaining)
                
        except Exception as e:
            logger.error(f"Memory cache TTL check failed for key {key}: {e}")
            return -1
    
    def health_check(self) -> Dict[str, Any]:
        """Perform memory cache health check."""
        try:
            with self._lock:
                # Clean up expired entries for accurate stats
                self._cleanup_expired()
                
                total_entries = len(self._store)
                memory_usage = sum(
                    len(entry.value) + len(key.encode('utf-8'))
                    for key, entry in self._store.items()
                )
                
                # Calculate hit rate
                total_requests = self._stats['hits'] + self._stats['misses']
                hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
                
                return {
                    'status': 'healthy',
                    'backend_type': 'memory',
                    'total_entries': total_entries,
                    'max_size': self.max_size,
                    'memory_usage_bytes': memory_usage,
                    'memory_usage_mb': round(memory_usage / (1024 * 1024), 2),
                    'hit_rate': round(hit_rate, 4),
                    'stats': self._stats.copy(),
                    'utilization': round(total_entries / self.max_size, 4) if self.max_size > 0 else 0
                }
                
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'backend_type': 'memory'
            }
    
    def get_size(self) -> int:
        """Get current cache size."""
        with self._lock:
            return len(self._store)
    
    def get_memory_usage(self) -> int:
        """Get approximate memory usage in bytes."""
        with self._lock:
            return sum(
                len(entry.value) + len(key.encode('utf-8'))
                for key, entry in self._store.items()
            )