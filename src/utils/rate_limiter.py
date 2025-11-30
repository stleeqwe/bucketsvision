"""
Rate limiter implementation using token bucket algorithm.

정밀한 API 호출 제어를 위한 토큰 버킷 알고리즘 구현.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenBucket:
    """
    토큰 버킷 알고리즘 기반 Rate Limiter.

    Args:
        rate: 초당 토큰 생성 속도 (예: 1.5 = 분당 90개)
        capacity: 버킷 최대 용량
    """
    rate: float  # tokens per second
    capacity: int
    tokens: float = field(init=False)
    last_update: float = field(init=False)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self):
        self.tokens = float(self.capacity)
        self.last_update = time.monotonic()

    def _refill(self) -> None:
        """토큰 리필"""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    async def acquire(self, tokens: int = 1) -> float:
        """
        토큰 획득 (필요시 대기).

        Args:
            tokens: 필요한 토큰 수

        Returns:
            대기 시간 (초)
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return 0.0

            # 토큰 부족 시 대기 시간 계산
            tokens_needed = tokens - self.tokens
            wait_time = tokens_needed / self.rate

            await asyncio.sleep(wait_time)

            self._refill()
            self.tokens -= tokens
            return wait_time

    async def try_acquire(self, tokens: int = 1) -> bool:
        """
        토큰 획득 시도 (대기 없음).

        Args:
            tokens: 필요한 토큰 수

        Returns:
            획득 성공 여부
        """
        async with self._lock:
            self._refill()

            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    @property
    def available_tokens(self) -> float:
        """현재 사용 가능한 토큰 수"""
        self._refill()
        return self.tokens


class RateLimiter:
    """
    다중 rate limit 관리자.

    서로 다른 rate limit이 필요한 엔드포인트를 위한 관리자.
    """

    def __init__(self):
        self._limiters: dict[str, TokenBucket] = {}

    def register(self, name: str, rate_per_minute: int, burst: Optional[int] = None) -> None:
        """
        새 rate limiter 등록.

        Args:
            name: limiter 이름 (예: "default", "season")
            rate_per_minute: 분당 허용 요청 수
            burst: 버스트 허용량 (기본값: rate_per_minute // 6)
        """
        rate_per_second = rate_per_minute / 60.0
        capacity = burst or max(1, rate_per_minute // 6)
        self._limiters[name] = TokenBucket(rate=rate_per_second, capacity=capacity)

    async def acquire(self, name: str = "default", tokens: int = 1) -> float:
        """토큰 획득"""
        if name not in self._limiters:
            raise ValueError(f"Rate limiter '{name}' not registered")
        return await self._limiters[name].acquire(tokens)

    def get_limiter(self, name: str) -> TokenBucket:
        """limiter 인스턴스 조회"""
        if name not in self._limiters:
            raise ValueError(f"Rate limiter '{name}' not registered")
        return self._limiters[name]


# 전역 rate limiter 인스턴스
rate_limiter = RateLimiter()
