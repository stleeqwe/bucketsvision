"""
NBA Injury Report Client.

NBA 공식 Injury Report PDF를 다운로드하고 파싱하여 부상/결장 정보를 수집합니다.

PDF URL 형식:
https://ak-static.cms.nba.com/referee/injury/Injury-Report_{DATE}_{TIME}.pdf
예: https://ak-static.cms.nba.com/referee/injury/Injury-Report_2025-11-25_07PM.pdf
"""

import re
import io
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

import requests
import pdfplumber
import pandas as pd

from src.utils.logger import logger


@dataclass
class InjuryRecord:
    """부상/결장 기록"""
    game_date: str
    game_time: str
    matchup: str
    team: str
    player_name: str
    status: str  # Out, Questionable, Available, Probable
    reason: str


class NBAInjuryClient:
    """
    NBA 공식 Injury Report 클라이언트.

    PDF를 다운로드하고 파싱하여 부상/결장 정보를 추출합니다.
    """

    BASE_URL = "https://ak-static.cms.nba.com/referee/injury"

    # 팀명 매핑 (PDF 형식 -> 표준 형식)
    TEAM_MAPPING = {
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

    # 팀 ID 매핑
    TEAM_ID_MAPPING = {
        "ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751,
        "CHA": 1610612766, "CHI": 1610612741, "CLE": 1610612739,
        "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765,
        "GSW": 1610612744, "HOU": 1610612745, "IND": 1610612754,
        "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
        "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750,
        "NOP": 1610612740, "NYK": 1610612752, "OKC": 1610612760,
        "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756,
        "POR": 1610612757, "SAC": 1610612758, "SAS": 1610612759,
        "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764,
    }

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Args:
            cache_dir: PDF 캐시 디렉토리
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _build_pdf_url(self, report_date: date, time_str: str = "07PM") -> str:
        """PDF URL 생성"""
        date_str = report_date.strftime("%Y-%m-%d")
        return f"{self.BASE_URL}/Injury-Report_{date_str}_{time_str}.pdf"

    def _download_pdf(self, url: str) -> Optional[bytes]:
        """PDF 다운로드"""
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                return response.content
            else:
                logger.debug(f"PDF not found: {url} (status: {response.status_code})")
                return None
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None

    def _parse_pdf(self, pdf_content: bytes) -> List[InjuryRecord]:
        """PDF에서 부상 정보 추출"""
        records = []

        try:
            with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                all_text = ""
                for page in pdf.pages:
                    all_text += page.extract_text() + "\n"

                records = self._parse_text(all_text)
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")

        return records

    def _parse_text(self, text: str) -> List[InjuryRecord]:
        """텍스트에서 부상 정보 파싱"""
        records = []
        lines = text.split('\n')

        current_game_date = None
        current_game_time = None
        current_matchup = None
        current_team = None

        for line in lines:
            line = line.strip()
            if not line or line.startswith("Page") or line.startswith("Injury Report:"):
                continue

            # 헤더 라인 스킵
            if line.startswith("GameDate"):
                continue

            # 날짜 패턴: MM/DD/YYYY
            date_match = re.match(r'^(\d{2}/\d{2}/\d{4})\s+(\d{2}:\d{2}\(ET\))\s+(\S+@\S+)\s+(.+)', line)
            if date_match:
                current_game_date = date_match.group(1)
                current_game_time = date_match.group(2)
                current_matchup = date_match.group(3)
                rest = date_match.group(4)

                # 팀명과 선수 정보 파싱
                team_player = self._parse_team_player_line(rest)
                if team_player:
                    current_team, player_name, status, reason = team_player
                    records.append(InjuryRecord(
                        game_date=current_game_date,
                        game_time=current_game_time,
                        matchup=current_matchup,
                        team=current_team,
                        player_name=player_name,
                        status=status,
                        reason=reason
                    ))
                continue

            # 시간만 있는 라인 (같은 날짜의 다른 경기)
            time_match = re.match(r'^(\d{2}:\d{2}\(ET\))\s+(\S+@\S+)\s+(.+)', line)
            if time_match:
                current_game_time = time_match.group(1)
                current_matchup = time_match.group(2)
                rest = time_match.group(3)

                team_player = self._parse_team_player_line(rest)
                if team_player:
                    current_team, player_name, status, reason = team_player
                    records.append(InjuryRecord(
                        game_date=current_game_date,
                        game_time=current_game_time,
                        matchup=current_matchup,
                        team=current_team,
                        player_name=player_name,
                        status=status,
                        reason=reason
                    ))
                continue

            # 팀명으로 시작하는 라인
            for team_name, team_abbr in self.TEAM_MAPPING.items():
                if line.startswith(team_name):
                    current_team = team_abbr
                    rest = line[len(team_name):].strip()

                    player_info = self._parse_player_info(rest)
                    if player_info:
                        player_name, status, reason = player_info
                        records.append(InjuryRecord(
                            game_date=current_game_date,
                            game_time=current_game_time,
                            matchup=current_matchup,
                            team=current_team,
                            player_name=player_name,
                            status=status,
                            reason=reason
                        ))
                    break
            else:
                # 선수 정보만 있는 라인 (이전 팀 계속)
                if current_team and current_game_date:
                    player_info = self._parse_player_info(line)
                    if player_info:
                        player_name, status, reason = player_info
                        records.append(InjuryRecord(
                            game_date=current_game_date,
                            game_time=current_game_time,
                            matchup=current_matchup,
                            team=current_team,
                            player_name=player_name,
                            status=status,
                            reason=reason
                        ))

        return records

    def _parse_team_player_line(self, text: str) -> Optional[Tuple[str, str, str, str]]:
        """팀명+선수 정보 파싱"""
        for team_name, team_abbr in self.TEAM_MAPPING.items():
            if text.startswith(team_name):
                rest = text[len(team_name):].strip()
                player_info = self._parse_player_info(rest)
                if player_info:
                    return (team_abbr, *player_info)
        return None

    def _parse_player_info(self, text: str) -> Optional[Tuple[str, str, str]]:
        """선수 정보 파싱: 이름, 상태, 사유"""
        # 패턴: LastName,FirstName Status Reason
        # 예: Young,Trae Out Injury/Illness-RightKnee;MCLSprain

        status_keywords = ["Out", "Questionable", "Available", "Probable", "Doubtful"]

        for status in status_keywords:
            if f" {status} " in text or text.endswith(f" {status}"):
                parts = text.split(f" {status}")
                if len(parts) >= 1:
                    player_name = parts[0].strip()
                    reason = parts[1].strip() if len(parts) > 1 else ""

                    # 이름 정리: "Last,First" -> "First Last"
                    player_name = self._normalize_name(player_name)

                    if player_name:
                        return (player_name, status, reason)

        return None

    def _normalize_name(self, name: str) -> str:
        """이름 정규화: 'Last,First' -> 'First Last'"""
        name = name.strip()
        if "," in name:
            parts = name.split(",")
            if len(parts) == 2:
                return f"{parts[1].strip()} {parts[0].strip()}"
        return name

    def get_injury_report(
        self,
        report_date: date,
        time_str: str = "07PM"
    ) -> List[InjuryRecord]:
        """
        특정 날짜의 부상 리포트 조회.

        Args:
            report_date: 리포트 날짜
            time_str: 리포트 시간 (예: "07PM", "05PM")

        Returns:
            부상 기록 리스트
        """
        url = self._build_pdf_url(report_date, time_str)

        # 캐시 확인
        if self.cache_dir:
            cache_path = self.cache_dir / f"injury_{report_date}_{time_str}.pdf"
            if cache_path.exists():
                pdf_content = cache_path.read_bytes()
                return self._parse_pdf(pdf_content)

        # 다운로드
        pdf_content = self._download_pdf(url)
        if not pdf_content:
            return []

        # 캐시 저장
        if self.cache_dir:
            cache_path.write_bytes(pdf_content)

        return self._parse_pdf(pdf_content)

    def get_game_day_injuries(
        self,
        game_date: date,
        times: List[str] = ["07PM", "05PM", "01PM"]
    ) -> List[InjuryRecord]:
        """
        경기일 부상 정보 조회 (여러 시간대 시도).

        Args:
            game_date: 경기 날짜
            times: 시도할 시간대 리스트

        Returns:
            부상 기록 리스트
        """
        for time_str in times:
            records = self.get_injury_report(game_date, time_str)
            if records:
                logger.info(f"Found {len(records)} injury records for {game_date} at {time_str}")
                return records

        logger.warning(f"No injury report found for {game_date}")
        return []

    def get_team_injuries(
        self,
        game_date: date,
        team_abbr: str,
        status_filter: List[str] = ["Out"]
    ) -> List[InjuryRecord]:
        """
        특정 팀의 결장 선수 조회.

        Args:
            game_date: 경기 날짜
            team_abbr: 팀 약어 (예: "GSW")
            status_filter: 필터링할 상태 (기본: Out만)

        Returns:
            해당 팀의 결장 기록 리스트
        """
        records = self.get_game_day_injuries(game_date)

        return [
            r for r in records
            if r.team == team_abbr and r.status in status_filter
        ]

    def to_dataframe(self, records: List[InjuryRecord]) -> pd.DataFrame:
        """InjuryRecord 리스트를 DataFrame으로 변환"""
        if not records:
            return pd.DataFrame()

        data = [
            {
                "game_date": r.game_date,
                "game_time": r.game_time,
                "matchup": r.matchup,
                "team": r.team,
                "team_id": self.TEAM_ID_MAPPING.get(r.team),
                "player_name": r.player_name,
                "status": r.status,
                "reason": r.reason
            }
            for r in records
        ]

        return pd.DataFrame(data)

    def collect_date_range(
        self,
        start_date: date,
        end_date: date,
        status_filter: List[str] = ["Out"]
    ) -> pd.DataFrame:
        """
        날짜 범위의 부상 정보 수집.

        Args:
            start_date: 시작 날짜
            end_date: 종료 날짜
            status_filter: 필터링할 상태

        Returns:
            부상 정보 DataFrame
        """
        all_records = []
        current = start_date

        while current <= end_date:
            records = self.get_game_day_injuries(current)
            filtered = [r for r in records if r.status in status_filter]
            all_records.extend(filtered)

            logger.info(f"{current}: {len(filtered)} out players")
            current += timedelta(days=1)

        return self.to_dataframe(all_records)
