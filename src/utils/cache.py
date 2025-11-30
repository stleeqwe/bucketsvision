"""
Caching utilities for API responses and computed data.

디스크 기반 캐싱으로 중복 API 호출 방지 및 데이터 지속성 제공.
"""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Callable, TypeVar
from functools import wraps

from src.utils.logger import logger

T = TypeVar('T')


@dataclass
class CacheEntry:
    """캐시 엔트리"""
    data: Any
    created_at: float
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """만료 여부 확인"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class DiskCache:
    """
    디스크 기반 JSON 캐시.

    API 응답을 JSON 파일로 저장하여 재시작 후에도 캐시 유지.
    """

    def __init__(self, cache_dir: Path, default_ttl: Optional[float] = None):
        """
        Args:
            cache_dir: 캐시 저장 디렉토리
            default_ttl: 기본 TTL (초). None이면 무기한.
        """
        self.cache_dir = cache_dir
        self.default_ttl = default_ttl
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, key: str) -> Path:
        """캐시 키에 대한 파일 경로"""
        # 키를 해시하여 파일명으로 사용
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def _make_key(self, *args, **kwargs) -> str:
        """인자들로부터 캐시 키 생성"""
        key_parts = [str(a) for a in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        return ":".join(key_parts)

    def get(self, key: str) -> Optional[Any]:
        """
        캐시에서 값 조회.

        Args:
            key: 캐시 키

        Returns:
            캐시된 값 또는 None (만료/없음)
        """
        cache_path = self._get_cache_path(key)

        if not cache_path.exists():
            return None

        try:
            with open(cache_path, 'r') as f:
                entry_data = json.load(f)

            entry = CacheEntry(
                data=entry_data['data'],
                created_at=entry_data['created_at'],
                ttl=entry_data.get('ttl')
            )

            if entry.is_expired():
                cache_path.unlink()  # 만료된 캐시 삭제
                return None

            return entry.data

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Cache read error for key {key}: {e}")
            cache_path.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """
        캐시에 값 저장.

        Args:
            key: 캐시 키
            value: 저장할 값 (JSON 직렬화 가능해야 함)
            ttl: TTL (초). None이면 default_ttl 사용.
        """
        cache_path = self._get_cache_path(key)
        effective_ttl = ttl if ttl is not None else self.default_ttl

        entry_data = {
            'data': value,
            'created_at': time.time(),
            'ttl': effective_ttl
        }

        try:
            with open(cache_path, 'w') as f:
                json.dump(entry_data, f)
        except (TypeError, IOError) as e:
            logger.warning(f"Cache write error for key {key}: {e}")

    def delete(self, key: str) -> bool:
        """캐시 항목 삭제"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear(self) -> int:
        """전체 캐시 삭제"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count

    def cleanup_expired(self) -> int:
        """만료된 캐시 정리"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    entry_data = json.load(f)

                entry = CacheEntry(
                    data=None,
                    created_at=entry_data['created_at'],
                    ttl=entry_data.get('ttl')
                )

                if entry.is_expired():
                    cache_file.unlink()
                    count += 1

            except (json.JSONDecodeError, KeyError):
                cache_file.unlink()
                count += 1

        return count


class MemoryCache:
    """
    메모리 기반 캐시.

    빠른 접근이 필요한 데이터를 위한 인메모리 캐시.
    """

    def __init__(self, default_ttl: Optional[float] = None, max_size: int = 1000):
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: dict[str, CacheEntry] = {}

    def get(self, key: str) -> Optional[Any]:
        """캐시 조회"""
        entry = self._cache.get(key)
        if entry is None:
            return None

        if entry.is_expired():
            del self._cache[key]
            return None

        return entry.data

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """캐시 저장"""
        # 크기 제한 초과 시 오래된 항목 제거
        if len(self._cache) >= self.max_size:
            self._evict_oldest()

        effective_ttl = ttl if ttl is not None else self.default_ttl
        self._cache[key] = CacheEntry(
            data=value,
            created_at=time.time(),
            ttl=effective_ttl
        )

    def _evict_oldest(self) -> None:
        """가장 오래된 항목 제거"""
        if not self._cache:
            return

        oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k].created_at)
        del self._cache[oldest_key]

    def delete(self, key: str) -> bool:
        """항목 삭제"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    def clear(self) -> int:
        """전체 삭제"""
        count = len(self._cache)
        self._cache.clear()
        return count


def cached(cache: DiskCache, ttl: Optional[float] = None):
    """
    캐싱 데코레이터 (동기 함수용).

    Args:
        cache: 캐시 인스턴스
        ttl: TTL (초)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            key = cache._make_key(func.__name__, *args, **kwargs)
            cached_value = cache.get(key)

            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator


def async_cached(cache: DiskCache, ttl: Optional[float] = None):
    """
    캐싱 데코레이터 (비동기 함수용).

    Args:
        cache: 캐시 인스턴스
        ttl: TTL (초)
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            key = cache._make_key(func.__name__, *args, **kwargs)
            cached_value = cache.get(key)

            if cached_value is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return cached_value

            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result

        return wrapper
    return decorator
