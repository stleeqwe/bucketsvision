"""
NBA Official Injury Report Client.

NBA 공식 Injury Report PDF를 파싱하여 부상 정보를 수집합니다.
ESPN API 대비 Probable/Questionable/Doubtful 세부 상태를 구분할 수 있습니다.

PDF URL 패턴:
https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{HH}PM.pdf

업데이트 시간 (ET):
- 1PM, 5PM, 7PM, 9PM (경기일 기준)
"""

import io
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum

import requests

from src.utils.logger import logger

try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed. NBA PDF parsing disabled.")


class InjuryStatus(Enum):
    """부상 상태 (출전 가능성 순)"""
    AVAILABLE = "Available"      # 출전 가능
    PROBABLE = "Probable"        # 출전 가능성 75%
    QUESTIONABLE = "Questionable"  # 출전 가능성 50%
    DOUBTFUL = "Doubtful"        # 출전 가능성 25%
    OUT = "Out"                  # 결장 확정

    @property
    def play_probability(self) -> float:
        """출전 확률 반환"""
        return {
            InjuryStatus.AVAILABLE: 1.0,
            InjuryStatus.PROBABLE: 0.75,
            InjuryStatus.QUESTIONABLE: 0.50,
            InjuryStatus.DOUBTFUL: 0.25,
            InjuryStatus.OUT: 0.0,
        }[self]


@dataclass
class NBAInjury:
    """NBA 공식 부상 정보"""
    player_name: str
    team_name: str
    team_abbr: str
    status: InjuryStatus
    reason: str
    game_date: date
    game_time: str
    matchup: str


class NBAInjuryClient:
    """
    NBA 공식 Injury Report 클라이언트.

    PDF를 파싱하여 Probable/Questionable/Doubtful/Out 세부 상태를 구분합니다.
    """

    BASE_URL = "https://ak-static.cms.nba.com/referee/injury"

    # PDF 업데이트 시간 (ET 기준)
    UPDATE_TIMES = ["01PM", "05PM", "07PM", "09PM"]

    # 팀명 → 약어 매핑
    TEAM_NAME_TO_ABBR = {
        "AtlantaHawks": "ATL",
        "BostonCeltics": "BOS",
        "BrooklynNets": "BKN",
        "CharlotteHornets": "CHA",
        "ChicagoBulls": "CHI",
        "ClevelandCavaliers": "CLE",
        "DallasMavericks": "DAL",
        "DenverNuggets": "DEN",
        "DetroitPistons": "DET",
        "GoldenStateWarriors": "GSW",
        "HoustonRockets": "HOU",
        "IndianaPacers": "IND",
        "LAClippers": "LAC",
        "LosAngelesClippers": "LAC",
        "LosAngelesLakers": "LAL",
        "LALakers": "LAL",
        "MemphisGrizzlies": "MEM",
        "MiamiHeat": "MIA",
        "MilwaukeeBucks": "MIL",
        "MinnesotaTimberwolves": "MIN",
        "NewOrleansPelicans": "NOP",
        "NewYorkKnicks": "NYK",
        "OklahomaCityThunder": "OKC",
        "OrlandoMagic": "ORL",
        "Philadelphia76ers": "PHI",
        "PhoenixSuns": "PHX",
        "PortlandTrailBlazers": "POR",
        "SacramentoKings": "SAC",
        "SanAntonioSpurs": "SAS",
        "TorontoRaptors": "TOR",
        "UtahJazz": "UTA",
        "WashingtonWizards": "WAS",
    }

    def __init__(self):
        self._cache: Dict[str, List[NBAInjury]] = {}
        self._last_update: Optional[datetime] = None
        # 날짜별 마지막으로 조회한 PDF 시간 (예: "01PM", "05PM")
        self._last_pdf_time: Dict[str, str] = {}

    def _get_latest_pdf_url(self, target_date: date) -> Tuple[Optional[str], Optional[str]]:
        """
        가장 최신 PDF URL 찾기.

        Args:
            target_date: 조회할 날짜

        Returns:
            (PDF URL, PDF 시간) 또는 (None, None)
        """
        date_str = target_date.strftime("%Y-%m-%d")

        # 최신 시간부터 역순으로 확인
        for time in reversed(self.UPDATE_TIMES):
            url = f"{self.BASE_URL}/Injury-Report_{date_str}_{time}.pdf"
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Found NBA injury report: {date_str}_{time}")
                    return url, time
            except Exception:
                continue

        return None, None

    def _has_newer_pdf(self, target_date: date) -> bool:
        """
        더 최신 PDF가 있는지 확인.

        Args:
            target_date: 조회할 날짜

        Returns:
            True if 더 최신 PDF가 존재함
        """
        cache_key = target_date.isoformat()
        last_time = self._last_pdf_time.get(cache_key)

        if not last_time:
            return True  # 캐시 없으면 조회 필요

        # 마지막 조회 시간 이후의 PDF만 확인
        last_time_idx = self.UPDATE_TIMES.index(last_time) if last_time in self.UPDATE_TIMES else -1
        newer_times = self.UPDATE_TIMES[last_time_idx + 1:]

        date_str = target_date.strftime("%Y-%m-%d")

        for time in reversed(newer_times):
            url = f"{self.BASE_URL}/Injury-Report_{date_str}_{time}.pdf"
            try:
                response = requests.head(url, timeout=5)
                if response.status_code == 200:
                    logger.info(f"Newer NBA PDF available: {date_str}_{time} (was: {last_time})")
                    return True
            except Exception:
                continue

        return False

    def _normalize_player_name(self, name: str) -> str:
        """
        선수 이름 정규화.

        PDF 형식: "LastName,FirstName" → "FirstName LastName"
        """
        if "," in name:
            parts = name.split(",")
            if len(parts) == 2:
                return f"{parts[1].strip()} {parts[0].strip()}"
        return name

    def _get_team_abbr(self, team_name: str) -> str:
        """팀명에서 약어 추출"""
        # 공백 제거한 팀명으로 매칭
        clean_name = team_name.replace(" ", "")
        return self.TEAM_NAME_TO_ABBR.get(clean_name, "UNK")

    def _parse_status(self, status_str: str) -> Optional[InjuryStatus]:
        """상태 문자열 파싱"""
        status_str = status_str.strip()
        for status in InjuryStatus:
            if status.value.lower() == status_str.lower():
                return status
        return None

    def _parse_pdf(self, pdf_content: bytes, target_date: date) -> List[NBAInjury]:
        """
        PDF 파싱.

        Args:
            pdf_content: PDF 바이너리
            target_date: 경기 날짜

        Returns:
            부상 정보 리스트
        """
        if not PDF_AVAILABLE:
            logger.warning("pdfplumber not available")
            return []

        injuries = []

        try:
            pdf = pdfplumber.open(io.BytesIO(pdf_content))

            current_matchup = ""
            current_game_time = ""
            current_team = ""
            current_team_abbr = ""

            for page in pdf.pages:
                text = page.extract_text() or ""
                lines = text.split('\n')

                for line in lines:
                    # 헤더나 푸터 스킵
                    if "Injury Report:" in line or "GameDate" in line or "Page" in line:
                        continue

                    # 매치업 라인 감지 (예: "07:00(ET) WAS@PHI PortlandTrailBlazers")
                    matchup_match = re.search(r'(\d{2}:\d{2})\(ET\)\s+(\w+@\w+)', line)
                    if matchup_match:
                        current_game_time = matchup_match.group(1)
                        current_matchup = matchup_match.group(2)
                        # 매치업 정보 제거
                        line = re.sub(r'\d{2}:\d{2}\(ET\)\s+\w+@\w+\s*', '', line)

                    # 팀명 감지 (라인에서 팀명 찾기)
                    found_team = False
                    for team_name in sorted(self.TEAM_NAME_TO_ABBR.keys(), key=len, reverse=True):
                        if team_name in line:
                            current_team = team_name
                            current_team_abbr = self.TEAM_NAME_TO_ABBR[team_name]
                            # 팀명 제거
                            line = line.replace(team_name, "").strip()
                            found_team = True
                            break

                    # 선수 상태 파싱 (라인에 상태가 있으면)
                    if not line:
                        continue

                    for status in InjuryStatus:
                        if status.value in line:
                            # 선수명 추출 (상태 앞부분)
                            parts = line.split(status.value)
                            if len(parts) >= 1:
                                player_name_raw = parts[0].strip()
                                reason = parts[1].strip() if len(parts) > 1 else ""

                                # 선수명이 비어있거나 날짜 형식이면 스킵
                                if not player_name_raw or re.match(r'\d{2}/\d{2}', player_name_raw):
                                    continue

                                player_name = self._normalize_player_name(player_name_raw)

                                injury = NBAInjury(
                                    player_name=player_name,
                                    team_name=current_team,
                                    team_abbr=current_team_abbr,
                                    status=status,
                                    reason=reason,
                                    game_date=target_date,
                                    game_time=current_game_time,
                                    matchup=current_matchup,
                                )
                                injuries.append(injury)
                            break

            pdf.close()

        except Exception as e:
            logger.error(f"Error parsing NBA injury PDF: {e}")

        return injuries

    def fetch_injuries(
        self,
        target_date: Optional[date] = None,
        force_refresh: bool = False
    ) -> Dict[str, List[NBAInjury]]:
        """
        부상 정보 조회.

        같은 날 더 최신 PDF가 있으면 자동으로 재조회합니다.

        Args:
            target_date: 조회할 날짜 (None이면 오늘)
            force_refresh: 캐시 무시

        Returns:
            팀 약어 → 부상 리스트 딕셔너리
        """
        if not PDF_AVAILABLE:
            return {}

        if target_date is None:
            target_date = date.today()

        cache_key = target_date.isoformat()

        # 캐시 존재 시 더 최신 PDF가 있는지 확인
        if not force_refresh and cache_key in self._cache:
            # 같은 날이면 더 최신 PDF 확인
            if target_date == date.today() and self._has_newer_pdf(target_date):
                logger.info(f"Auto-refreshing: newer NBA PDF available for {target_date}")
                force_refresh = True
            else:
                # 팀별로 그룹화하여 반환
                result: Dict[str, List[NBAInjury]] = {}
                for injury in self._cache[cache_key]:
                    if injury.team_abbr not in result:
                        result[injury.team_abbr] = []
                    result[injury.team_abbr].append(injury)
                return result

        # PDF URL 찾기
        pdf_url, pdf_time = self._get_latest_pdf_url(target_date)

        if pdf_url is None:
            logger.info(f"No NBA injury report found for {target_date}")
            return {}

        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            injuries = self._parse_pdf(response.content, target_date)

            # 팀별로 그룹화
            result: Dict[str, List[NBAInjury]] = {}
            for injury in injuries:
                if injury.team_abbr not in result:
                    result[injury.team_abbr] = []
                result[injury.team_abbr].append(injury)

            self._cache[cache_key] = injuries
            self._last_update = datetime.now()
            if pdf_time:
                self._last_pdf_time[cache_key] = pdf_time

            total = len(injuries)
            teams = len(result)
            logger.info(f"NBA PDF: Loaded {total} injuries for {teams} teams ({target_date} {pdf_time})")

            return result

        except Exception as e:
            logger.error(f"Error fetching NBA injury report: {e}")
            return {}

    def get_team_injuries(
        self,
        team_abbr: str,
        target_date: Optional[date] = None
    ) -> List[NBAInjury]:
        """특정 팀 부상 정보 조회"""
        injuries = self.fetch_injuries(target_date)
        return injuries.get(team_abbr.upper(), [])

    def get_player_status(
        self,
        player_name: str,
        team_abbr: str,
        target_date: Optional[date] = None
    ) -> Optional[InjuryStatus]:
        """
        선수 상태 조회.

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어
            target_date: 조회 날짜

        Returns:
            InjuryStatus 또는 None (정보 없음)
        """
        injuries = self.get_team_injuries(team_abbr, target_date)

        normalized_query = player_name.lower().strip()

        for injury in injuries:
            normalized_player = injury.player_name.lower().strip()
            # 정확한 매칭 또는 부분 매칭
            if normalized_query == normalized_player:
                return injury.status
            if normalized_query in normalized_player or normalized_player in normalized_query:
                return injury.status

        return None

    def clear_cache(self) -> None:
        """캐시 초기화"""
        self._cache = {}
        self._last_update = None
        self._last_pdf_time = {}


def merge_espn_nba_injuries(
    espn_injuries: List,  # List[ESPNInjury]
    nba_injuries: Dict[str, List[NBAInjury]],
) -> Dict[str, Dict]:
    """
    ESPN과 NBA 부상 정보 병합.

    ESPN을 기본으로 하고, NBA PDF로 Day-To-Day를
    Probable/Questionable/Doubtful로 세분화합니다.

    Args:
        espn_injuries: ESPN 부상 리스트
        nba_injuries: NBA 부상 딕셔너리 (팀별)

    Returns:
        선수별 병합된 상태 {player_name: {status, source, play_prob}}
    """
    merged = {}

    for espn_injury in espn_injuries:
        player_name = espn_injury.player_name
        team_abbr = espn_injury.team_abbr

        result = {
            "espn_status": espn_injury.status,
            "nba_status": None,
            "final_status": espn_injury.status,
            "play_probability": 0.0 if espn_injury.status == "Out" else 0.5,  # GTD 기본 50%
            "source": "ESPN",
        }

        # NBA PDF에서 추가 정보 확인
        nba_team_injuries = nba_injuries.get(team_abbr, [])

        for nba_injury in nba_team_injuries:
            # 이름 매칭
            espn_name = player_name.lower().strip()
            nba_name = nba_injury.player_name.lower().strip()

            if espn_name == nba_name or espn_name in nba_name or nba_name in espn_name:
                result["nba_status"] = nba_injury.status.value
                result["final_status"] = nba_injury.status.value
                result["play_probability"] = nba_injury.status.play_probability
                result["source"] = "NBA_PDF"
                break

        merged[f"{player_name}_{team_abbr}"] = result

    return merged
