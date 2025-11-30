"""
Dunks and Threes API 동기식 클라이언트.

Streamlit 앱에서 사용하기 위한 requests 기반 동기 클라이언트입니다.
"""

import time
from typing import Any, Dict, List, Optional

import requests

from config.settings import settings


class DNTApiClient:
    """Dunks and Threes API 동기식 클라이언트"""

    BASE_URL = "https://dunksandthrees.com/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: API 키 (없으면 settings에서 로드)
        """
        self.api_key = api_key or settings.dnt_api_key
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": self.api_key,
            "Accept": "application/json",
        })
        self._last_request_time = 0.0
        self._min_interval = 0.7  # 초당 약 1.5 요청

    def _rate_limit(self) -> None:
        """Rate limiting"""
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: Optional[Dict] = None) -> Any:
        """
        API 요청.

        Args:
            endpoint: API 엔드포인트
            params: 쿼리 파라미터

        Returns:
            JSON 응답
        """
        self._rate_limit()

        url = f"{self.BASE_URL}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"DNT API error: {e}")

    def get_team_epm(
        self,
        date: Optional[str] = None
    ) -> List[Dict]:
        """
        팀 EPM 조회.

        Args:
            date: 날짜 (YYYY-MM-DD), 없으면 최신

        Returns:
            팀 EPM 리스트
        """
        params = {}
        if date:
            params["date"] = date

        return self._request("team-epm", params)

    def get_player_epm(
        self,
        date: Optional[str] = None,
        season: int = 2026
    ) -> List[Dict]:
        """
        선수 EPM 조회.

        Args:
            date: 날짜 (YYYY-MM-DD)
            season: 시즌

        Returns:
            선수 EPM 리스트
        """
        params = {"season": season}
        if date:
            params["date"] = date

        return self._request("epm", params)

    def get_game_predictions(
        self,
        date: Optional[str] = None
    ) -> List[Dict]:
        """
        경기 예측 조회.

        Args:
            date: 날짜 (YYYY-MM-DD)

        Returns:
            경기 예측 리스트
        """
        params = {}
        if date:
            params["date"] = date

        return self._request("game-predictions", params)
