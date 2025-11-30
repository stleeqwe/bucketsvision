"""
Dunks and Threes API Client.

비동기 HTTP 클라이언트로 D&T API와 통신합니다.
Rate limiting, 재시도, 캐싱을 포함한 robust한 구현.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

import aiohttp
from aiohttp import ClientTimeout, ClientError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from src.utils.logger import logger
from src.utils.rate_limiter import RateLimiter, TokenBucket
from src.utils.cache import DiskCache
from src.utils.helpers import format_date, parse_date


class DNTAPIError(Exception):
    """D&T API 에러 기본 클래스"""
    pass


class DNTRateLimitError(DNTAPIError):
    """Rate limit 초과 에러"""
    pass


class DNTAuthenticationError(DNTAPIError):
    """인증 실패 에러"""
    pass


class DNTNotFoundError(DNTAPIError):
    """데이터 없음 에러"""
    pass


@dataclass
class DNTResponse:
    """API 응답 래퍼"""
    data: Any
    status_code: int
    cached: bool = False


@dataclass
class DNTClientConfig:
    """클라이언트 설정"""
    api_key: str
    base_url: str = "https://dunksandthrees.com/api/v1"
    rate_limit: int = 90  # requests per minute
    season_rate_limit: int = 3  # for season queries
    timeout: int = 30
    max_retries: int = 3
    cache_ttl: Optional[int] = 3600  # 1시간 기본 캐시


class DNTClient:
    """
    Dunks and Threes API 비동기 클라이언트.

    Features:
    - 토큰 버킷 기반 rate limiting
    - 지수 백오프 재시도
    - 디스크 캐싱
    - 배치 요청 지원

    Usage:
        async with DNTClient(config) as client:
            team_epm = await client.get_team_epm(date="2024-01-15")
    """

    def __init__(
        self,
        config: DNTClientConfig,
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            config: 클라이언트 설정
            cache_dir: 캐시 디렉토리 (None이면 캐싱 비활성화)
        """
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

        # Rate limiters 설정
        self._rate_limiter = RateLimiter()
        self._rate_limiter.register(
            "default",
            rate_per_minute=config.rate_limit,
            burst=15
        )
        self._rate_limiter.register(
            "season",
            rate_per_minute=config.season_rate_limit,
            burst=1
        )

        # 캐시 설정
        self._cache: Optional[DiskCache] = None
        if cache_dir:
            self._cache = DiskCache(cache_dir, default_ttl=config.cache_ttl)

        # 세마포어로 동시 요청 제한
        self._semaphore = asyncio.Semaphore(10)

    async def __aenter__(self) -> "DNTClient":
        """컨텍스트 매니저 진입"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """컨텍스트 매니저 종료"""
        await self.close()

    async def _create_session(self) -> None:
        """HTTP 세션 생성"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(total=self.config.timeout)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": self.config.api_key,
                    "Accept": "application/json",
                    "User-Agent": "NBA-Score-Predictor/1.0"
                }
            )

    async def close(self) -> None:
        """리소스 정리"""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_cache_key(self, endpoint: str, params: Dict[str, Any]) -> str:
        """캐시 키 생성"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
        return f"dnt:{endpoint}:{param_str}"

    async def _make_request(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        rate_limiter_name: str = "default",
        use_cache: bool = True
    ) -> DNTResponse:
        """
        API 요청 실행.

        Args:
            endpoint: API 엔드포인트 (예: "/team-epm")
            params: 쿼리 파라미터
            rate_limiter_name: 사용할 rate limiter
            use_cache: 캐시 사용 여부

        Returns:
            DNTResponse 객체
        """
        params = params or {}
        cache_key = self._get_cache_key(endpoint, params)

        # 캐시 확인
        if use_cache and self._cache:
            cached_data = self._cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {endpoint}")
                return DNTResponse(data=cached_data, status_code=200, cached=True)

        # Rate limiting
        wait_time = await self._rate_limiter.acquire(rate_limiter_name)
        if wait_time > 0:
            logger.debug(f"Rate limited, waited {wait_time:.2f}s")

        # 세마포어로 동시 요청 제한
        async with self._semaphore:
            response = await self._execute_request(endpoint, params)

        # 캐시 저장
        if use_cache and self._cache and response.status_code == 200:
            self._cache.set(cache_key, response.data)

        return response

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((ClientError, asyncio.TimeoutError, DNTRateLimitError)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _execute_request(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> DNTResponse:
        """
        실제 HTTP 요청 실행 (재시도 로직 포함).

        Args:
            endpoint: API 엔드포인트
            params: 쿼리 파라미터

        Returns:
            DNTResponse 객체
        """
        await self._create_session()

        url = f"{self.config.base_url}{endpoint}"

        try:
            async with self._session.get(url, params=params) as response:
                # 상태 코드별 처리
                if response.status == 200:
                    data = await response.json()
                    return DNTResponse(data=data, status_code=200)

                elif response.status == 429:
                    # Rate limit - 재시도 트리거
                    logger.warning(f"Rate limit hit for {endpoint}")
                    await asyncio.sleep(5)  # 5초 대기
                    raise DNTRateLimitError("Rate limit exceeded")

                elif response.status == 401:
                    raise DNTAuthenticationError("Invalid API key")

                elif response.status == 404:
                    raise DNTNotFoundError(f"No data found for {endpoint}")

                else:
                    text = await response.text()
                    raise DNTAPIError(f"API error {response.status}: {text}")

        except aiohttp.ClientError as e:
            logger.error(f"Client error for {endpoint}: {e}")
            raise

    # ===================
    # Public API Methods
    # ===================

    async def get_team_epm(
        self,
        date: Optional[str] = None,
        season: Optional[int] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        팀 EPM 데이터 조회.

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            season: 시즌 연도 (전체 시즌 조회)
            use_cache: 캐시 사용 여부

        Returns:
            팀 EPM 데이터 리스트
        """
        params = {}
        rate_limiter = "default"

        if date:
            params["date"] = date
        if season:
            params["season"] = season
            rate_limiter = "season"  # 시즌 쿼리는 더 엄격한 rate limit

        response = await self._make_request(
            "/team-epm",
            params=params,
            rate_limiter_name=rate_limiter,
            use_cache=use_cache
        )

        return response.data

    async def get_player_epm(
        self,
        date: str,
        game_optimized: int = 0,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        선수 EPM 데이터 조회 (경기일 기준).

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            game_optimized: game optimized 여부 (0 or 1)
            use_cache: 캐시 사용 여부

        Returns:
            선수 EPM 데이터 리스트
        """
        params = {
            "date": date,
            "game_optimized": game_optimized
        }

        response = await self._make_request(
            "/epm",
            params=params,
            use_cache=use_cache
        )

        return response.data

    async def get_all_player_epm(
        self,
        date: str,
        game_optimized: int = 0,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        모든 선수 EPM 데이터 조회.

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            game_optimized: game optimized 여부 (0 or 1)
            use_cache: 캐시 사용 여부

        Returns:
            전체 선수 EPM 데이터 리스트
        """
        params = {
            "date": date,
            "game_optimized": game_optimized
        }

        response = await self._make_request(
            "/epm-all",
            params=params,
            use_cache=use_cache
        )

        return response.data

    async def get_season_epm(
        self,
        season: int,
        seasontype: int = 2,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        시즌 EPM 데이터 조회.

        Args:
            season: 시즌 연도 (예: 2024 for 23-24)
            seasontype: 시즌 타입 (2: 정규시즌, 4: 플레이오프)
            use_cache: 캐시 사용 여부

        Returns:
            시즌 EPM 데이터 리스트
        """
        params = {
            "season": season,
            "seasontype": seasontype
        }

        response = await self._make_request(
            "/season-epm",
            params=params,
            rate_limiter_name="season",
            use_cache=use_cache
        )

        return response.data

    async def get_game_predictions(
        self,
        date: Optional[str] = None,
        game_id: Optional[int] = None,
        days: int = 1,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        경기 예측 데이터 조회.

        Args:
            date: 조회 날짜 (YYYY-MM-DD)
            game_id: 특정 경기 ID
            days: 조회 기간 (일)
            use_cache: 캐시 사용 여부

        Returns:
            경기 예측 데이터 리스트
        """
        params = {"days": days}
        if date:
            params["date"] = date
        if game_id:
            params["game_id"] = game_id

        response = await self._make_request(
            "/game-predictions",
            params=params,
            use_cache=use_cache
        )

        return response.data

    # ===================
    # Batch Collection Methods
    # ===================

    async def collect_season_team_epm(
        self,
        season: int,
        progress_callback: Optional[callable] = None
    ) -> List[Dict[str, Any]]:
        """
        시즌 전체 팀 EPM 수집.

        Args:
            season: 시즌 연도
            progress_callback: 진행률 콜백 함수

        Returns:
            시즌 전체 팀 EPM 데이터
        """
        logger.info(f"Collecting team EPM for season {season}")

        data = await self.get_team_epm(season=season)

        if progress_callback:
            progress_callback(1.0)

        logger.info(f"Collected {len(data)} team EPM records for season {season}")
        return data

    async def collect_date_range_player_epm(
        self,
        start_date: str,
        end_date: str,
        game_optimized: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        날짜 범위의 선수 EPM 수집.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            game_optimized: game optimized 여부
            progress_callback: 진행률 콜백

        Returns:
            날짜별 선수 EPM 딕셔너리
        """
        start = parse_date(start_date)
        end = parse_date(end_date)
        total_days = (end - start).days + 1

        results = {}
        current_date = start

        for i in range(total_days):
            date_str = format_date(current_date)

            try:
                data = await self.get_player_epm(date_str, game_optimized)
                if data:
                    results[date_str] = data
            except DNTNotFoundError:
                logger.debug(f"No player EPM data for {date_str}")
            except Exception as e:
                logger.error(f"Error collecting player EPM for {date_str}: {e}")

            current_date += timedelta(days=1)

            if progress_callback:
                progress_callback((i + 1) / total_days)

        logger.info(f"Collected player EPM for {len(results)} dates")
        return results

    async def collect_multiple_seasons(
        self,
        seasons: List[int],
        data_type: str = "team_epm"
    ) -> Dict[int, Any]:
        """
        다중 시즌 데이터 수집.

        Args:
            seasons: 수집할 시즌 리스트
            data_type: 데이터 타입 ("team_epm", "season_epm")

        Returns:
            시즌별 데이터 딕셔너리
        """
        results = {}

        for season in seasons:
            logger.info(f"Collecting {data_type} for season {season}")

            try:
                if data_type == "team_epm":
                    data = await self.get_team_epm(season=season)
                elif data_type == "season_epm":
                    data = await self.get_season_epm(season=season)
                else:
                    raise ValueError(f"Unknown data type: {data_type}")

                results[season] = data
                logger.info(f"Collected {len(data)} records for season {season}")

                # 시즌 쿼리 간 추가 대기 (rate limit 보호)
                await asyncio.sleep(2)

            except Exception as e:
                logger.error(f"Failed to collect {data_type} for season {season}: {e}")
                results[season] = []

        return results


# ===================
# Factory Function
# ===================

def create_dnt_client(
    api_key: str,
    cache_dir: Optional[Path] = None,
    **kwargs
) -> DNTClient:
    """
    DNT 클라이언트 팩토리 함수.

    Args:
        api_key: API 키
        cache_dir: 캐시 디렉토리
        **kwargs: 추가 설정

    Returns:
        DNTClient 인스턴스
    """
    config = DNTClientConfig(api_key=api_key, **kwargs)
    return DNTClient(config, cache_dir)
