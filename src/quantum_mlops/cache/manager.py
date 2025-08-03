"""Cache management for quantum MLOps workbench."""

import os
import pickle
import json
import hashlib
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Dict
import logging

from .redis_backend import RedisCache
from .memory_backend import MemoryCache


logger = logging.getLogger(__name__)


class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete key from cache."""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cache entries."""
        pass
    
    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on cache backend."""
        pass


class CacheManager:
    """Manager for quantum MLOps caching operations."""
    
    def __init__(self, backend: Optional[CacheBackend] = None):
        self.backend = backend or self._create_default_backend()
        self.key_prefix = "qmlops:"
        self.default_ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour
    
    def _create_default_backend(self) -> CacheBackend:
        """Create default cache backend based on environment."""
        redis_url = os.getenv("REDIS_URL")
        
        if redis_url:
            try:
                return RedisCache(redis_url)
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}. Falling back to memory cache.")
        
        return MemoryCache()
    
    def _make_key(self, key: str, namespace: str = "") -> str:
        """Create cache key with prefix and namespace."""
        if namespace:
            return f"{self.key_prefix}{namespace}:{key}"
        return f"{self.key_prefix}{key}"
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                return json.dumps(value).encode('utf-8')
            else:
                # Use pickle for complex objects
                return pickle.dumps(value)
        except Exception as e:
            logger.error(f"Failed to serialize value: {e}")
            raise
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            try:
                return json.loads(data.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Fall back to pickle
                return pickle.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize value: {e}")
            raise
    
    def get(self, key: str, namespace: str = "") -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._make_key(key, namespace)
        
        try:
            data = self.backend.get(cache_key)
            if data is not None:
                return self._deserialize_value(data)
            return None
        except Exception as e:
            logger.error(f"Cache get failed for key {cache_key}: {e}")
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        namespace: str = ""
    ) -> bool:
        """Set value in cache."""
        cache_key = self._make_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_value = self._serialize_value(value)
            return self.backend.set(cache_key, serialized_value, ttl)
        except Exception as e:
            logger.error(f"Cache set failed for key {cache_key}: {e}")
            return False
    
    def delete(self, key: str, namespace: str = "") -> bool:
        """Delete key from cache."""
        cache_key = self._make_key(key, namespace)
        
        try:
            return self.backend.delete(cache_key)
        except Exception as e:
            logger.error(f"Cache delete failed for key {cache_key}: {e}")
            return False
    
    def exists(self, key: str, namespace: str = "") -> bool:
        """Check if key exists in cache."""
        cache_key = self._make_key(key, namespace)
        
        try:
            return self.backend.exists(cache_key)
        except Exception as e:
            logger.error(f"Cache exists check failed for key {cache_key}: {e}")
            return False
    
    def get_or_set(
        self,
        key: str,
        default_fn: callable,
        ttl: Optional[int] = None,
        namespace: str = ""
    ) -> Any:
        """Get value from cache or set it using default function."""
        value = self.get(key, namespace)
        
        if value is not None:
            return value
        
        # Compute value and cache it
        try:
            value = default_fn()
            self.set(key, value, ttl, namespace)
            return value
        except Exception as e:
            logger.error(f"Failed to compute default value for key {key}: {e}")
            raise
    
    def cache_circuit_result(
        self,
        circuit_hash: str,
        parameters: Union[list, tuple],
        backend: str,
        result: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache quantum circuit execution result."""
        # Create unique key for circuit + parameters + backend
        param_hash = hashlib.md5(str(parameters).encode()).hexdigest()
        key = f"circuit:{circuit_hash}:{param_hash}:{backend}"
        
        return self.set(key, result, ttl, namespace="circuits")
    
    def get_cached_circuit_result(
        self,
        circuit_hash: str,
        parameters: Union[list, tuple],
        backend: str
    ) -> Optional[Any]:
        """Get cached quantum circuit execution result."""
        param_hash = hashlib.md5(str(parameters).encode()).hexdigest()
        key = f"circuit:{circuit_hash}:{param_hash}:{backend}"
        
        return self.get(key, namespace="circuits")
    
    def cache_model_prediction(
        self,
        model_id: str,
        input_hash: str,
        prediction: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache model prediction result."""
        key = f"model:{model_id}:input:{input_hash}"
        return self.set(key, prediction, ttl, namespace="predictions")
    
    def get_cached_model_prediction(
        self,
        model_id: str,
        input_hash: str
    ) -> Optional[Any]:
        """Get cached model prediction result."""
        key = f"model:{model_id}:input:{input_hash}"
        return self.get(key, namespace="predictions")
    
    def cache_experiment_metrics(
        self,
        experiment_id: str,
        run_id: str,
        metrics: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache experiment metrics."""
        key = f"experiment:{experiment_id}:run:{run_id}:metrics"
        return self.set(key, metrics, ttl, namespace="experiments")
    
    def get_cached_experiment_metrics(
        self,
        experiment_id: str,
        run_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached experiment metrics."""
        key = f"experiment:{experiment_id}:run:{run_id}:metrics"
        return self.get(key, namespace="experiments")
    
    def cache_backend_status(
        self,
        backend_name: str,
        status_info: Dict[str, Any],
        ttl: int = 300  # 5 minutes
    ) -> bool:
        """Cache quantum backend status information."""
        key = f"backend:{backend_name}:status"
        return self.set(key, status_info, ttl, namespace="backends")
    
    def get_cached_backend_status(self, backend_name: str) -> Optional[Dict[str, Any]]:
        """Get cached quantum backend status."""
        key = f"backend:{backend_name}:status"
        return self.get(key, namespace="backends")
    
    def invalidate_namespace(self, namespace: str) -> int:
        """Invalidate all cache entries in a namespace."""
        # This is a simplified implementation
        # In production, would need pattern-based deletion
        try:
            pattern = self._make_key("*", namespace)
            return self.backend.delete_pattern(pattern) if hasattr(self.backend, 'delete_pattern') else 0
        except Exception as e:
            logger.error(f"Failed to invalidate namespace {namespace}: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries."""
        try:
            return self.backend.clear()
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            health = self.backend.health_check()
            return {
                "backend_type": self.backend.__class__.__name__,
                "health": health,
                "default_ttl": self.default_ttl,
                "key_prefix": self.key_prefix
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def initialize_cache(backend: Optional[CacheBackend] = None) -> CacheManager:
    """Initialize the global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(backend)
    return _cache_manager