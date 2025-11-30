"""
Injury Impact Calculator.

결장 선수의 영향도를 Player EPM을 기반으로 계산합니다.

공식:
    injury_impact = (결장 선수 EPM - 대체 선수 EPM) × (32/48)

    - 결장 선수 EPM: 해당 선수의 시즌 EPM
    - 대체 선수 EPM: 해당 팀 벤치 선수(MPG < 20) 평균 EPM
    - 32/48: 평균 출장 시간 비율
"""

from datetime import date
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd
import numpy as np

from src.utils.logger import logger


class InjuryImpactCalculator:
    """
    결장 영향도 계산기.

    Player EPM 데이터를 사용하여 결장 선수의 팀 성적 영향도를 계산합니다.
    """

    # 출장 시간 비율 (32분 / 48분)
    MINUTES_RATIO = 32 / 48

    # 주전 기준 MPG
    STARTER_MPG_THRESHOLD = 20

    def __init__(self, player_epm_df: pd.DataFrame):
        """
        Args:
            player_epm_df: Player EPM DataFrame (season_epm 데이터)
        """
        self.player_epm = player_epm_df.copy()

        # 이름 정규화
        self.player_epm['player_name_normalized'] = self.player_epm['player_name'].apply(
            self._normalize_name
        )

        # 팀별 벤치 평균 EPM 계산
        self.bench_avg_epm = self._calculate_bench_avg()

        logger.info(f"Loaded {len(self.player_epm)} players, {len(self.bench_avg_epm)} teams")

    def _normalize_name(self, name: str) -> str:
        """이름 정규화 (소문자, 공백 제거)"""
        if pd.isna(name):
            return ""
        return name.lower().strip().replace(".", "").replace("'", "").replace("-", " ")

    def _calculate_bench_avg(self) -> Dict[str, float]:
        """팀별 벤치 평균 EPM 계산"""
        bench = self.player_epm[self.player_epm['mpg'] < self.STARTER_MPG_THRESHOLD]
        return bench.groupby('team_alias')['tot'].mean().to_dict()

    def find_player(
        self,
        player_name: str,
        team_abbr: Optional[str] = None
    ) -> Optional[pd.Series]:
        """
        선수 찾기 (퍼지 매칭 지원).

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어 (선택)

        Returns:
            선수 정보 Series 또는 None
        """
        normalized = self._normalize_name(player_name)

        # 정확한 매칭 시도
        matches = self.player_epm[
            self.player_epm['player_name_normalized'] == normalized
        ]

        if team_abbr and len(matches) > 1:
            matches = matches[matches['team_alias'] == team_abbr]

        if len(matches) == 1:
            return matches.iloc[0]

        if len(matches) > 1:
            return matches.iloc[0]  # 첫 번째 반환

        # 퍼지 매칭 시도
        if team_abbr:
            team_players = self.player_epm[self.player_epm['team_alias'] == team_abbr]
        else:
            team_players = self.player_epm

        best_match = None
        best_ratio = 0.0

        for _, player in team_players.iterrows():
            ratio = SequenceMatcher(
                None, normalized, player['player_name_normalized']
            ).ratio()

            if ratio > best_ratio and ratio > 0.7:  # 70% 이상 일치
                best_ratio = ratio
                best_match = player

        if best_match is not None:
            logger.debug(f"Fuzzy matched '{player_name}' -> '{best_match['player_name']}' ({best_ratio:.2%})")

        return best_match

    def calculate_player_impact(
        self,
        player_name: str,
        team_abbr: str
    ) -> float:
        """
        단일 선수 결장 영향도 계산.

        공식: (선수 EPM - 벤치 평균 EPM) × (32/48)

        Args:
            player_name: 선수 이름
            team_abbr: 팀 약어

        Returns:
            결장 영향도 (양수 = 팀에 불리)
        """
        player = self.find_player(player_name, team_abbr)

        if player is None:
            logger.debug(f"Player not found: {player_name} ({team_abbr})")
            return 0.0

        player_epm = player['tot']
        bench_epm = self.bench_avg_epm.get(team_abbr, -2.0)  # 기본값: -2.0

        # 주전급 선수만 의미있는 영향도 계산 (벤치 선수 결장은 영향 미미)
        if player['mpg'] < self.STARTER_MPG_THRESHOLD:
            return 0.0

        impact = (player_epm - bench_epm) * self.MINUTES_RATIO

        logger.debug(
            f"{player_name}: EPM={player_epm:.2f}, Bench={bench_epm:.2f}, "
            f"Impact={impact:.2f}"
        )

        return impact

    def calculate_team_injury_impact(
        self,
        out_players: List[Tuple[str, str]],  # [(player_name, team_abbr), ...]
    ) -> Dict[str, float]:
        """
        팀별 결장 영향도 합계 계산.

        Args:
            out_players: 결장 선수 리스트 [(이름, 팀약어), ...]

        Returns:
            팀별 결장 영향도 딕셔너리
        """
        team_impacts = {}

        for player_name, team_abbr in out_players:
            impact = self.calculate_player_impact(player_name, team_abbr)

            if team_abbr not in team_impacts:
                team_impacts[team_abbr] = 0.0

            team_impacts[team_abbr] += impact

        return team_impacts

    def get_game_injury_impact(
        self,
        home_team: str,
        away_team: str,
        injury_records: List,  # List of InjuryRecord
    ) -> Tuple[float, float]:
        """
        경기별 결장 영향도 계산.

        Args:
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            injury_records: InjuryRecord 리스트

        Returns:
            (home_injury_impact, away_injury_impact)
            양수 = 해당 팀에 불리
        """
        home_impact = 0.0
        away_impact = 0.0

        for record in injury_records:
            if record.status != "Out":
                continue

            # G-League, Two-Way 선수는 제외
            if "GLeague" in record.reason or "Two-Way" in record.reason:
                continue

            impact = self.calculate_player_impact(record.player_name, record.team)

            if record.team == home_team:
                home_impact += impact
            elif record.team == away_team:
                away_impact += impact

        return home_impact, away_impact


def load_player_epm(data_dir: Path, season: int) -> pd.DataFrame:
    """Player EPM 데이터 로드"""
    path = data_dir / "raw" / "dnt" / "season_epm" / f"season_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Player EPM not found: {path}")

    return pd.read_parquet(path)


def create_injury_calculator(data_dir: Path, season: int) -> InjuryImpactCalculator:
    """InjuryImpactCalculator 팩토리 함수"""
    player_epm = load_player_epm(data_dir, season)
    return InjuryImpactCalculator(player_epm)
