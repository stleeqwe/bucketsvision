"""
ESPN Injuries API Client.

ESPN 비공식 API를 사용하여 부상/결장 정보를 수집합니다.
gtd-calculator 프로젝트의 espn_api.py를 참고하여 구현.

API Endpoint:
https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from difflib import SequenceMatcher

import requests

from src.utils.logger import logger


@dataclass
class ESPNInjury:
    """ESPN 부상 정보"""
    espn_id: Optional[str]
    player_name: str
    team_abbr: str
    position: Optional[str]
    status: str  # Out, Day-To-Day, etc.
    detail: Optional[str]  # 부상 상세 (e.g., "Knee - Soreness")
    injury_type: Optional[str]
    fantasy_status: Optional[str]  # O, GTD, etc.


class ESPNInjuryClient:
    """
    ESPN 부상 정보 클라이언트.

    비공식 ESPN API를 사용하여 전체 NBA 팀의 부상 정보를 수집합니다.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

    # ESPN 팀 약어 -> 표준 약어 매핑
    TEAM_ABBR_MAP = {
        "GS": "GSW",
        "NY": "NYK",
        "NO": "NOP",
        "SA": "SAS",
        "UTAH": "UTA",
        "WSH": "WAS",
        "PHX": "PHX",
        "BKN": "BKN",
    }

    def __init__(self):
        self._cache: Dict[str, List[ESPNInjury]] = {}

    def _normalize_team_abbr(self, abbr: str) -> str:
        """ESPN 팀 약어를 표준 약어로 변환"""
        abbr = abbr.upper()
        return self.TEAM_ABBR_MAP.get(abbr, abbr)

    def fetch_all_injuries(self, force_refresh: bool = False) -> Dict[str, List[ESPNInjury]]:
        """
        전체 NBA 팀 부상 정보 조회.

        Args:
            force_refresh: 캐시 무시하고 새로 조회

        Returns:
            팀 약어 -> 부상 리스트 딕셔너리
        """
        if self._cache and not force_refresh:
            return self._cache

        url = f"{self.BASE_URL}/injuries"

        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()

            self._cache = {}

            for team_data in data.get("injuries", []):
                team_injuries = team_data.get("injuries", [])

                for injury in team_injuries:
                    athlete = injury.get("athlete", {})
                    team = athlete.get("team", {})
                    team_abbr = self._normalize_team_abbr(
                        team.get("abbreviation", "")
                    )

                    if not team_abbr:
                        continue

                    # ESPN player ID 추출
                    espn_id = None
                    for link in athlete.get("links", []):
                        href = link.get("href", "")
                        if "/player/_/id/" in href:
                            parts = href.split("/id/")
                            if len(parts) > 1:
                                espn_id = parts[1].split("/")[0]
                                break

                    injury_record = ESPNInjury(
                        espn_id=espn_id,
                        player_name=athlete.get("displayName", ""),
                        team_abbr=team_abbr,
                        position=athlete.get("position", {}).get("abbreviation"),
                        status=injury.get("status", ""),
                        detail=injury.get("details", {}).get("detail"),
                        injury_type=injury.get("details", {}).get("type"),
                        fantasy_status=injury.get("details", {}).get(
                            "fantasyStatus", {}
                        ).get("abbreviation"),
                    )

                    if team_abbr not in self._cache:
                        self._cache[team_abbr] = []
                    self._cache[team_abbr].append(injury_record)

            total_injuries = sum(len(v) for v in self._cache.values())
            logger.info(f"ESPN: Loaded {total_injuries} injuries for {len(self._cache)} teams")

        except Exception as e:
            logger.error(f"ESPN API error: {e}")

        return self._cache

    def get_team_injuries(
        self,
        team_abbr: str,
        status_filter: Optional[List[str]] = None
    ) -> List[ESPNInjury]:
        """
        특정 팀 부상 정보 조회.

        Args:
            team_abbr: 팀 약어 (e.g., "LAL")
            status_filter: 필터링할 상태 리스트 (e.g., ["Out"])

        Returns:
            부상 리스트
        """
        self.fetch_all_injuries()

        injuries = self._cache.get(team_abbr.upper(), [])

        if status_filter:
            injuries = [
                inj for inj in injuries
                if inj.status in status_filter
            ]

        return injuries

    def get_out_players(self, team_abbr: str) -> List[ESPNInjury]:
        """Out 상태 선수만 조회"""
        return self.get_team_injuries(team_abbr, status_filter=["Out"])

    def get_gtd_players(self, team_abbr: str) -> List[ESPNInjury]:
        """
        GTD (Day-To-Day) 선수 조회.

        Out 상태 선수는 제외하고 순수 GTD 선수만 반환합니다.

        Args:
            team_abbr: 팀 약어

        Returns:
            GTD 선수 리스트
        """
        self.fetch_all_injuries()
        injuries = self._cache.get(team_abbr.upper(), [])

        gtd_players = []
        for injury in injuries:
            # Out 상태는 제외 (Out은 별도로 처리됨)
            if injury.status == "Out":
                continue

            is_gtd = (
                injury.status == "Day-To-Day"
                or injury.fantasy_status == "GTD"
                or (injury.detail and "day-to-day" in injury.detail.lower())
                or (injury.detail and "game time decision" in injury.detail.lower())
            )

            if is_gtd:
                gtd_players.append(injury)

        return gtd_players

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache = {}


def fuzzy_match_name(name1: str, name2: str, threshold: float = 0.8) -> bool:
    """
    이름 퍼지 매칭.

    Args:
        name1: 첫 번째 이름
        name2: 두 번째 이름
        threshold: 매칭 임계값 (0-1)

    Returns:
        매칭 여부
    """
    if not name1 or not name2:
        return False

    n1 = name1.lower().strip()
    n2 = name2.lower().strip()

    # 정확한 매칭
    if n1 == n2:
        return True

    # 퍼지 매칭
    ratio = SequenceMatcher(None, n1, n2).ratio()
    return ratio >= threshold
