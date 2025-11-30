"""
The Odds API Client.

NBA 경기의 베팅 라인(스프레드, 머니라인, 토탈)을 가져옵니다.
https://the-odds-api.com/
"""

import os
from dataclasses import dataclass
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv

from src.utils.logger import logger
from config.constants import TEAM_INFO, ABBR_TO_ID

load_dotenv()


@dataclass
class GameOdds:
    """경기별 배당 정보"""
    game_id: str  # Odds API game ID
    home_team: str
    away_team: str
    commence_time: datetime

    # 스프레드 (핸디캡)
    spread_home: Optional[float] = None  # 홈팀 핸디캡 (예: -11.5)
    spread_home_odds: Optional[float] = None  # 배당률
    spread_away: Optional[float] = None
    spread_away_odds: Optional[float] = None

    # 머니라인 (승패)
    moneyline_home: Optional[float] = None  # 배당률
    moneyline_away: Optional[float] = None

    # 토탈 (오버/언더)
    total_line: Optional[float] = None  # 기준 점수 (예: 220.5)
    over_odds: Optional[float] = None
    under_odds: Optional[float] = None

    # 북메이커 정보
    bookmaker: str = ""


class OddsAPIClient:
    """The Odds API 클라이언트"""

    BASE_URL = "https://api.the-odds-api.com/v4"
    SPORT = "basketball_nba"

    # NBA 팀명 매핑 (Odds API -> 내부 약어)
    TEAM_NAME_MAP = {
        "Atlanta Hawks": "ATL",
        "Boston Celtics": "BOS",
        "Brooklyn Nets": "BKN",
        "Charlotte Hornets": "CHA",
        "Chicago Bulls": "CHI",
        "Cleveland Cavaliers": "CLE",
        "Dallas Mavericks": "DAL",
        "Denver Nuggets": "DEN",
        "Detroit Pistons": "DET",
        "Golden State Warriors": "GSW",
        "Houston Rockets": "HOU",
        "Indiana Pacers": "IND",
        "Los Angeles Clippers": "LAC",
        "Los Angeles Lakers": "LAL",
        "Memphis Grizzlies": "MEM",
        "Miami Heat": "MIA",
        "Milwaukee Bucks": "MIL",
        "Minnesota Timberwolves": "MIN",
        "New Orleans Pelicans": "NOP",
        "New York Knicks": "NYK",
        "Oklahoma City Thunder": "OKC",
        "Orlando Magic": "ORL",
        "Philadelphia 76ers": "PHI",
        "Phoenix Suns": "PHX",
        "Portland Trail Blazers": "POR",
        "Sacramento Kings": "SAC",
        "San Antonio Spurs": "SAS",
        "Toronto Raptors": "TOR",
        "Utah Jazz": "UTA",
        "Washington Wizards": "WAS",
    }

    def __init__(self, api_key: Optional[str] = None):
        """
        Args:
            api_key: Odds API 키 (없으면 환경변수에서 로드)
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("ODDS_API_KEY not found in environment")

        self._session = requests.Session()
        self._remaining_requests: Optional[int] = None
        self._used_requests: Optional[int] = None

    def get_nba_odds(
        self,
        markets: List[str] = ["spreads", "h2h", "totals"],
        bookmakers: List[str] = ["pinnacle", "draftkings", "fanduel"],
        odds_format: str = "decimal"
    ) -> List[GameOdds]:
        """
        NBA 경기 배당 조회.

        Args:
            markets: 조회할 마켓 ["spreads", "h2h", "totals"]
            bookmakers: 북메이커 필터 (빈 리스트면 전체)
            odds_format: 배당 형식 ("decimal", "american")

        Returns:
            GameOdds 리스트
        """
        if not self.api_key:
            logger.error("Odds API key not configured")
            return []

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": odds_format,
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            url = f"{self.BASE_URL}/sports/{self.SPORT}/odds"
            response = self._session.get(url, params=params, timeout=15)

            # API 사용량 추적
            self._remaining_requests = response.headers.get("x-requests-remaining")
            self._used_requests = response.headers.get("x-requests-used")
            logger.info(f"Odds API: {self._remaining_requests} requests remaining")

            response.raise_for_status()
            data = response.json()

            return self._parse_odds_response(data)

        except requests.RequestException as e:
            logger.error(f"Failed to fetch odds: {e}")
            return []

    def get_odds_by_teams(
        self,
        home_abbr: str,
        away_abbr: str,
        markets: List[str] = ["spreads", "h2h"]
    ) -> Optional[GameOdds]:
        """
        특정 경기의 배당 조회.

        Args:
            home_abbr: 홈팀 약어 (예: "HOU")
            away_abbr: 원정팀 약어 (예: "UTA")
            markets: 조회할 마켓

        Returns:
            GameOdds 또는 None
        """
        all_odds = self.get_nba_odds(markets=markets)

        for odds in all_odds:
            home_match = self.TEAM_NAME_MAP.get(odds.home_team) == home_abbr
            away_match = self.TEAM_NAME_MAP.get(odds.away_team) == away_abbr
            if home_match and away_match:
                return odds

        return None

    def get_all_games_odds(self) -> Dict[Tuple[str, str], GameOdds]:
        """
        모든 경기의 배당을 딕셔너리로 반환.

        Returns:
            {(home_abbr, away_abbr): GameOdds} 형태의 딕셔너리
        """
        all_odds = self.get_nba_odds()
        result = {}

        for odds in all_odds:
            home_abbr = self.TEAM_NAME_MAP.get(odds.home_team)
            away_abbr = self.TEAM_NAME_MAP.get(odds.away_team)
            if home_abbr and away_abbr:
                result[(home_abbr, away_abbr)] = odds

        return result

    def _parse_odds_response(self, data: List[Dict]) -> List[GameOdds]:
        """API 응답 파싱"""
        results = []

        for game in data:
            try:
                game_odds = GameOdds(
                    game_id=game["id"],
                    home_team=game["home_team"],
                    away_team=game["away_team"],
                    commence_time=datetime.fromisoformat(
                        game["commence_time"].replace("Z", "+00:00")
                    )
                )

                # 북메이커별 배당 파싱 (Pinnacle 우선, 없으면 첫 번째 사용)
                if game.get("bookmakers"):
                    # Pinnacle 우선 선택 (가장 정확한 라인)
                    bookmaker = None
                    for bm in game["bookmakers"]:
                        if bm["key"] == "pinnacle":
                            bookmaker = bm
                            break
                    if not bookmaker:
                        bookmaker = game["bookmakers"][0]

                    game_odds.bookmaker = bookmaker["key"]

                    for market in bookmaker.get("markets", []):
                        self._parse_market(game_odds, market)

                results.append(game_odds)

            except Exception as e:
                logger.warning(f"Failed to parse game odds: {e}")
                continue

        return results

    def _parse_market(self, game_odds: GameOdds, market: Dict) -> None:
        """마켓별 배당 파싱"""
        market_key = market["key"]
        outcomes = market.get("outcomes", [])

        if market_key == "spreads":
            for outcome in outcomes:
                if outcome["name"] == game_odds.home_team:
                    game_odds.spread_home = outcome.get("point")
                    game_odds.spread_home_odds = outcome.get("price")
                elif outcome["name"] == game_odds.away_team:
                    game_odds.spread_away = outcome.get("point")
                    game_odds.spread_away_odds = outcome.get("price")

        elif market_key == "h2h":
            for outcome in outcomes:
                if outcome["name"] == game_odds.home_team:
                    game_odds.moneyline_home = outcome.get("price")
                elif outcome["name"] == game_odds.away_team:
                    game_odds.moneyline_away = outcome.get("price")

        elif market_key == "totals":
            for outcome in outcomes:
                if outcome["name"] == "Over":
                    game_odds.total_line = outcome.get("point")
                    game_odds.over_odds = outcome.get("price")
                elif outcome["name"] == "Under":
                    game_odds.under_odds = outcome.get("price")

    @property
    def remaining_requests(self) -> Optional[int]:
        """남은 API 요청 횟수"""
        return self._remaining_requests

    @staticmethod
    def calculate_implied_probability(decimal_odds: float) -> float:
        """
        배당률을 암묵적 확률로 변환.

        Args:
            decimal_odds: 소수점 배당 (예: 1.91)

        Returns:
            확률 (0~1)
        """
        return 1 / decimal_odds if decimal_odds > 0 else 0

    @staticmethod
    def calculate_edge(
        model_prob: float,
        market_odds: float
    ) -> float:
        """
        Edge 계산 (모델 확률 vs 시장 배당).

        Args:
            model_prob: 모델 예측 확률 (0~1)
            market_odds: 시장 배당률 (decimal)

        Returns:
            Edge (양수면 가치 배팅)
        """
        implied_prob = OddsAPIClient.calculate_implied_probability(market_odds)
        return model_prob - implied_prob

    @staticmethod
    def calculate_ev(
        model_prob: float,
        market_odds: float
    ) -> float:
        """
        기대값(EV) 계산.

        Args:
            model_prob: 모델 예측 확률 (0~1)
            market_odds: 시장 배당률 (decimal)

        Returns:
            EV (1단위 베팅 기준)
        """
        return model_prob * (market_odds - 1) - (1 - model_prob)
