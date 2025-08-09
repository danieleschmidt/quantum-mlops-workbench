"""Redis cache backend implementation."""

import os
import logging
from typing import Any, Dict, Optional
from urllib.parse import urlparse

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value by key."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set key-value pair with optional TTL."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached items."""
        pass


class RedisCache(CacheBackend):
    """Redis-based cache backend."""
    
    def __init__(self, redis_url: Optional[str] = None, **kwargs):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package not available. Install with: pip install redis")
        
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.client = self._create_client(**kwargs)
        self._test_connection()
    
    def _create_client(self, **kwargs) -> redis.Redis:
        """Create Redis client."""
        try:
            # Parse Redis URL
            parsed_url = urlparse(self.redis_url)
            
            # Connection parameters
            connection_params = {
                'host': parsed_url.hostname or 'localhost',
                'port': parsed_url.port or 6379,
                'db': int(parsed_url.path.lstrip('/')) if parsed_url.path else 0,
                'decode_responses': False,  # We handle serialization ourselves
                'socket_timeout': kwargs.get('socket_timeout', 5),
                'socket_connect_timeout': kwargs.get('socket_connect_timeout', 5),
                'retry_on_timeout': kwargs.get('retry_on_timeout', True),
                'health_check_interval': kwargs.get('health_check_interval', 30)
            }
            
            # Add authentication if provided
            if parsed_url.password:
                connection_params['password'] = parsed_url.password
            if parsed_url.username:
                connection_params['username'] = parsed_url.username
            
            # Connection pooling
            connection_pool = redis.ConnectionPool(**connection_params)
            
            return redis.Redis(connection_pool=connection_pool)
            
        except Exception as e:
            logger.error(f"Failed to create Redis client: {e}")
            raise
    
    def _test_connection(self) -> None:
        """Test Redis connection."""
        try:
            self.client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection test failed: {e}")
            raise
    
    def get(self, key: str) -> Optional[bytes]:
        """Get value from Redis."""
        try:
            return self.client.get(key)
        except redis.RedisError as e:
            logger.error(f"Redis get failed for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Redis get for key {key}: {e}")
            return None
    
    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """Set value in Redis."""
        try:
            if ttl:
                return self.client.setex(key, ttl, value)
            else:
                return self.client.set(key, value)
        except redis.RedisError as e:
            logger.error(f"Redis set failed for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis set for key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete key from Redis."""
        try:
            return bool(self.client.delete(key))
        except redis.RedisError as e:
            logger.error(f"Redis delete failed for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis delete for key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            return bool(self.client.exists(key))
        except redis.RedisError as e:
            logger.error(f"Redis exists check failed for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis exists for key {key}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all keys from current Redis database."""
        try:
            self.client.flushdb()
            return True
        except redis.RedisError as e:
            logger.error(f"Redis clear failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis clear: {e}")
            return False
    
    def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = self.client.keys(pattern)
            if keys:
                return self.client.delete(*keys)
            return 0
        except redis.RedisError as e:
            logger.error(f"Redis pattern delete failed for pattern {pattern}: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error in Redis pattern delete: {e}")
            return 0
    
    def get_multiple(self, keys: list) -> Dict[str, Optional[bytes]]:
        """Get multiple values from Redis."""
        try:
            if not keys:
                return {}
            
            values = self.client.mget(keys)
            return dict(zip(keys, values))
        except redis.RedisError as e:
            logger.error(f"Redis mget failed: {e}")
            return {key: None for key in keys}
        except Exception as e:
            logger.error(f"Unexpected error in Redis mget: {e}")
            return {key: None for key in keys}
    
    def set_multiple(self, mapping: Dict[str, bytes], ttl: Optional[int] = None) -> bool:
        """Set multiple values in Redis."""
        try:
            if not mapping:
                return True
            
            # Use pipeline for efficiency
            pipe = self.client.pipeline()
            
            for key, value in mapping.items():
                if ttl:
                    pipe.setex(key, ttl, value)
                else:
                    pipe.set(key, value)
            
            results = pipe.execute()
            return all(results)
            
        except redis.RedisError as e:
            logger.error(f"Redis mset failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis mset: {e}")
            return False
    
    def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a counter in Redis."""
        try:
            return self.client.incrby(key, amount)
        except redis.RedisError as e:
            logger.error(f"Redis increment failed for key {key}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Redis increment: {e}")
            return None
    
    def expire(self, key: str, ttl: int) -> bool:
        """Set TTL for existing key."""
        try:
            return bool(self.client.expire(key, ttl))
        except redis.RedisError as e:
            logger.error(f"Redis expire failed for key {key}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in Redis expire: {e}")
            return False
    
    def ttl(self, key: str) -> int:
        """Get TTL for key."""
        try:
            return self.client.ttl(key)
        except redis.RedisError as e:
            logger.error(f"Redis TTL check failed for key {key}: {e}")
            return -1
        except Exception as e:
            logger.error(f"Unexpected error in Redis TTL: {e}")
            return -1
    
    def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check."""
        try:
            # Basic ping test
            ping_result = self.client.ping()
            
            # Get Redis info
            info = self.client.info()
            
            # Memory usage
            memory_info = {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'maxmemory': info.get('maxmemory', 0),
                'maxmemory_human': info.get('maxmemory_human', '0B'),
            }
            
            # Connection info
            connection_info = {
                'connected_clients': info.get('connected_clients', 0),
                'blocked_clients': info.get('blocked_clients', 0),
                'total_connections_received': info.get('total_connections_received', 0),
            }
            
            # Performance info
            performance_info = {
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'ops_per_sec': info.get('instantaneous_ops_per_sec', 0),
            }
            
            # Calculate hit rate
            hits = performance_info['keyspace_hits']
            misses = performance_info['keyspace_misses']
            hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0
            
            return {
                'status': 'healthy' if ping_result else 'unhealthy',
                'ping': ping_result,
                'redis_version': info.get('redis_version', 'unknown'),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0),
                'memory': memory_info,
                'connections': connection_info,
                'performance': performance_info,
                'hit_rate': hit_rate,
                'database_keys': self._get_database_keys(),
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'ping': False
            }
    
    def _get_database_keys(self) -> Dict[str, int]:
        """Get key count for each database."""
        try:
            info = self.client.info('keyspace')
            db_keys = {}
            
            for key, value in info.items():
                if key.startswith('db'):
                    # Parse "keys=123,expires=45,avg_ttl=67890"
                    parts = value.split(',')
                    for part in parts:
                        if part.startswith('keys='):
                            db_keys[key] = int(part.split('=')[1])
                            break
            
            return db_keys
            
        except Exception as e:
            logger.error(f"Failed to get database keys: {e}")
            return {}
    
    def close(self) -> None:
        """Close Redis connection."""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")