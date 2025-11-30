"""
Player On/Off Analyzer.

선수별 출전/미출전 시 팀 성적을 분석하여
데이터 기반 영향도를 계산합니다.

핵심 지표:
- Raw On/Off: 단순 마진 차이
- Adjusted On/Off: 상대팀 강도 보정된 마진 차이
- Statistical Significance: t-test, 신뢰구간
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats

from src.utils.logger import logger


@dataclass
class OnOffResult:
    """On/Off 분석 결과"""
    player_id: int
    player_name: str
    team_id: int
    team_abbr: str

    # 경기 수
    games_on: int
    games_off: int

    # Raw On/Off
    margin_on: float  # 출전 시 평균 마진
    margin_off: float  # 미출전 시 평균 마진
    raw_impact: float  # margin_on - margin_off

    # Adjusted On/Off
    adjusted_margin_on: float  # 보정된 출전 시 마진
    adjusted_margin_off: float  # 보정된 미출전 시 마진
    adjusted_impact: float  # 조정된 영향도

    # 통계적 유의성
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float  # Cohen's d

    # 신뢰도
    is_significant: bool  # p < 0.05
    sample_quality: str  # "high", "medium", "low"

    def __repr__(self):
        sig = "*" if self.is_significant else ""
        return (
            f"OnOffResult({self.player_name} [{self.team_abbr}]: "
            f"Impact={self.adjusted_impact:+.2f}{sig}, "
            f"On={self.games_on}, Off={self.games_off})"
        )


@dataclass
class TeamGameRecord:
    """팀 경기 기록 (분석용)"""
    game_id: str
    game_date: str
    team_id: int
    opponent_id: int
    is_home: bool
    margin: int
    team_epm: float
    opponent_epm: float
    expected_margin: float  # 예상 마진


class PlayerOnOffAnalyzer:
    """
    선수별 On/Off 성적 분석기.

    출전/미출전 시 팀 성적 차이를 계산하고
    통계적 유의성을 검증합니다.
    """

    # 홈 어드밴티지 (점)
    HOME_ADVANTAGE = 3.0

    # 최소 경기 수 기준
    MIN_GAMES_ON = 5
    MIN_GAMES_OFF = 3

    def __init__(
        self,
        games_df: pd.DataFrame,
        player_games_df: pd.DataFrame,
        team_epm_df: Optional[pd.DataFrame] = None,
        min_games_on: int = 10,
        min_games_off: int = 3
    ):
        """
        Args:
            games_df: 경기 결과 DataFrame (game_id, margin, home_team_id, away_team_id)
            player_games_df: 선수별 경기 출전 DataFrame (player_id, game_id, played)
            team_epm_df: 팀 EPM DataFrame (상대팀 강도 보정용)
            min_games_on: 분석 대상 최소 출전 경기 수
            min_games_off: 분석 대상 최소 미출전 경기 수
        """
        self.games_df = games_df.copy()
        self.player_games_df = player_games_df.copy()
        self.team_epm_df = team_epm_df.copy() if team_epm_df is not None else None
        self.min_games_on = min_games_on
        self.min_games_off = min_games_off

        # 데이터 전처리
        self._preprocess_data()

        # 결과 캐시
        self._impact_cache: Dict[int, OnOffResult] = {}

    def _preprocess_data(self):
        """데이터 전처리"""
        # 경기 데이터 정규화
        if "game_date" in self.games_df.columns:
            self.games_df["game_date"] = pd.to_datetime(
                self.games_df["game_date"]
            ).dt.strftime("%Y-%m-%d")

        # 선수 경기 데이터 정규화
        if "game_date" in self.player_games_df.columns:
            self.player_games_df["game_date"] = pd.to_datetime(
                self.player_games_df["game_date"]
            ).dt.strftime("%Y-%m-%d")

        # 팀 EPM 데이터 준비
        if self.team_epm_df is not None:
            self._prepare_team_epm()

        # 팀별 경기 매핑 구축
        self._build_team_game_mapping()

        logger.info(f"Preprocessed data:")
        logger.info(f"  Games: {len(self.games_df)}")
        logger.info(f"  Player game records: {len(self.player_games_df)}")
        logger.info(f"  Unique players: {self.player_games_df['player_id'].nunique()}")

    def _prepare_team_epm(self):
        """팀 EPM 데이터 준비 (가장 최근 값 사용)"""
        if self.team_epm_df is None or self.team_epm_df.empty:
            return

        # 날짜 정규화
        if "game_dt" in self.team_epm_df.columns:
            self.team_epm_df["game_date"] = pd.to_datetime(
                self.team_epm_df["game_dt"]
            ).dt.strftime("%Y-%m-%d")

        # 팀별 가장 최근 EPM 값
        if "team_epm" in self.team_epm_df.columns:
            latest_epm = (
                self.team_epm_df.sort_values("game_date")
                .groupby("team_id")
                .last()
                .reset_index()
            )
            self._team_epm_lookup = latest_epm.set_index("team_id")["team_epm"].to_dict()
        else:
            self._team_epm_lookup = {}

    def _build_team_game_mapping(self):
        """팀별 경기 매핑 구축"""
        # 홈팀 관점
        home_games = self.games_df[["game_id", "home_team_id", "away_team_id", "margin"]].copy()
        home_games = home_games.rename(columns={
            "home_team_id": "team_id",
            "away_team_id": "opponent_id"
        })
        home_games["is_home"] = True
        home_games["team_margin"] = home_games["margin"]

        # 원정팀 관점
        away_games = self.games_df[["game_id", "home_team_id", "away_team_id", "margin"]].copy()
        away_games = away_games.rename(columns={
            "away_team_id": "team_id",
            "home_team_id": "opponent_id"
        })
        away_games["is_home"] = False
        away_games["team_margin"] = -away_games["margin"]

        # 병합
        self._team_games = pd.concat([home_games, away_games], ignore_index=True)

    def get_player_on_off_games(
        self,
        player_id: int,
        team_id: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        선수의 출전/미출전 경기 분리.

        Args:
            player_id: 선수 ID
            team_id: 팀 ID (선수가 여러 팀 소속인 경우)

        Returns:
            (출전 경기 DataFrame, 미출전 경기 DataFrame)
        """
        # 선수의 출전 경기 목록
        player_played = self.player_games_df[
            (self.player_games_df["player_id"] == player_id) &
            (self.player_games_df["played"] == True)
        ]

        if team_id is None and not player_played.empty:
            team_id = player_played["team_id"].mode().iloc[0]

        played_game_ids = set(player_played["game_id"].unique())

        # 팀의 전체 경기
        team_games = self._team_games[self._team_games["team_id"] == team_id].copy()

        # 출전/미출전 분리
        on_games = team_games[team_games["game_id"].isin(played_game_ids)]
        off_games = team_games[~team_games["game_id"].isin(played_game_ids)]

        return on_games, off_games

    def calculate_raw_on_off(
        self,
        player_id: int,
        team_id: Optional[int] = None
    ) -> Dict:
        """
        Raw On/Off 계산 (보정 없음).

        Args:
            player_id: 선수 ID
            team_id: 팀 ID

        Returns:
            {
                "margin_on": 출전 시 평균 마진,
                "margin_off": 미출전 시 평균 마진,
                "raw_impact": 차이,
                "games_on": 출전 경기 수,
                "games_off": 미출전 경기 수
            }
        """
        on_games, off_games = self.get_player_on_off_games(player_id, team_id)

        margin_on = on_games["team_margin"].mean() if len(on_games) > 0 else np.nan
        margin_off = off_games["team_margin"].mean() if len(off_games) > 0 else np.nan

        raw_impact = margin_on - margin_off if not np.isnan(margin_on) and not np.isnan(margin_off) else np.nan

        return {
            "margin_on": margin_on,
            "margin_off": margin_off,
            "raw_impact": raw_impact,
            "games_on": len(on_games),
            "games_off": len(off_games),
            "margins_on": on_games["team_margin"].tolist(),
            "margins_off": off_games["team_margin"].tolist(),
        }

    def calculate_expected_margin(
        self,
        team_id: int,
        opponent_id: int,
        is_home: bool
    ) -> float:
        """
        예상 마진 계산 (상대팀 강도 보정용).

        Args:
            team_id: 팀 ID
            opponent_id: 상대팀 ID
            is_home: 홈 경기 여부

        Returns:
            예상 마진
        """
        team_epm = self._team_epm_lookup.get(team_id, 0)
        opp_epm = self._team_epm_lookup.get(opponent_id, 0)

        expected = (team_epm - opp_epm) + (self.HOME_ADVANTAGE if is_home else -self.HOME_ADVANTAGE)

        return expected

    def calculate_adjusted_on_off(
        self,
        player_id: int,
        team_id: Optional[int] = None
    ) -> Dict:
        """
        Adjusted On/Off 계산 (상대팀 강도 보정).

        각 경기에서 (실제 마진 - 예상 마진)을 계산하고 평균.

        Args:
            player_id: 선수 ID
            team_id: 팀 ID

        Returns:
            조정된 On/Off 결과
        """
        on_games, off_games = self.get_player_on_off_games(player_id, team_id)

        # 예상 마진 대비 실제 성적 계산
        def get_residuals(games_df):
            residuals = []
            for _, game in games_df.iterrows():
                expected = self.calculate_expected_margin(
                    game["team_id"],
                    game["opponent_id"],
                    game["is_home"]
                )
                residual = game["team_margin"] - expected
                residuals.append(residual)
            return residuals

        if self._team_epm_lookup:
            residuals_on = get_residuals(on_games) if len(on_games) > 0 else []
            residuals_off = get_residuals(off_games) if len(off_games) > 0 else []
        else:
            # EPM 데이터 없으면 raw 마진 사용
            residuals_on = on_games["team_margin"].tolist() if len(on_games) > 0 else []
            residuals_off = off_games["team_margin"].tolist() if len(off_games) > 0 else []

        adjusted_on = np.mean(residuals_on) if residuals_on else np.nan
        adjusted_off = np.mean(residuals_off) if residuals_off else np.nan
        adjusted_impact = adjusted_on - adjusted_off if not np.isnan(adjusted_on) and not np.isnan(adjusted_off) else np.nan

        return {
            "adjusted_margin_on": adjusted_on,
            "adjusted_margin_off": adjusted_off,
            "adjusted_impact": adjusted_impact,
            "residuals_on": residuals_on,
            "residuals_off": residuals_off,
        }

    def calculate_statistical_significance(
        self,
        margins_on: List[float],
        margins_off: List[float],
        confidence_level: float = 0.95
    ) -> Dict:
        """
        통계적 유의성 검증.

        Args:
            margins_on: 출전 경기 마진 리스트
            margins_off: 미출전 경기 마진 리스트
            confidence_level: 신뢰 수준

        Returns:
            {
                "t_statistic": t 통계량,
                "p_value": p 값,
                "confidence_interval": 신뢰구간,
                "effect_size": Cohen's d
            }
        """
        if len(margins_on) < 2 or len(margins_off) < 2:
            return {
                "t_statistic": np.nan,
                "p_value": np.nan,
                "confidence_interval": (np.nan, np.nan),
                "effect_size": np.nan,
                "is_significant": False,
            }

        # Independent t-test (Welch's t-test)
        t_stat, p_value = stats.ttest_ind(margins_on, margins_off, equal_var=False)

        # Effect size (Cohen's d)
        mean_diff = np.mean(margins_on) - np.mean(margins_off)
        pooled_std = np.sqrt(
            ((len(margins_on) - 1) * np.var(margins_on, ddof=1) +
             (len(margins_off) - 1) * np.var(margins_off, ddof=1)) /
            (len(margins_on) + len(margins_off) - 2)
        )
        effect_size = mean_diff / pooled_std if pooled_std > 0 else np.nan

        # Bootstrap confidence interval
        ci = self._bootstrap_ci(margins_on, margins_off, confidence_level)

        return {
            "t_statistic": t_stat,
            "p_value": p_value,
            "confidence_interval": ci,
            "effect_size": effect_size,
            "is_significant": p_value < (1 - confidence_level),
        }

    def _bootstrap_ci(
        self,
        margins_on: List[float],
        margins_off: List[float],
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Bootstrap 신뢰구간 계산"""
        np.random.seed(42)

        boot_diffs = []
        for _ in range(n_bootstrap):
            boot_on = np.random.choice(margins_on, size=len(margins_on), replace=True)
            boot_off = np.random.choice(margins_off, size=len(margins_off), replace=True)
            boot_diffs.append(np.mean(boot_on) - np.mean(boot_off))

        alpha = 1 - confidence_level
        lower = np.percentile(boot_diffs, 100 * alpha / 2)
        upper = np.percentile(boot_diffs, 100 * (1 - alpha / 2))

        return (lower, upper)

    def analyze_player(
        self,
        player_id: int,
        team_id: Optional[int] = None
    ) -> Optional[OnOffResult]:
        """
        선수 종합 분석.

        Args:
            player_id: 선수 ID
            team_id: 팀 ID

        Returns:
            OnOffResult 또는 None (기준 미달 시)
        """
        # 캐시 확인
        cache_key = (player_id, team_id)
        if cache_key in self._impact_cache:
            return self._impact_cache[cache_key]

        # Raw On/Off 계산
        raw = self.calculate_raw_on_off(player_id, team_id)

        # 최소 경기 수 확인
        if raw["games_on"] < self.min_games_on:
            return None
        if raw["games_off"] < self.min_games_off:
            return None

        # Adjusted On/Off 계산
        adjusted = self.calculate_adjusted_on_off(player_id, team_id)

        # 통계적 유의성 검증
        stats_result = self.calculate_statistical_significance(
            adjusted.get("residuals_on", raw["margins_on"]),
            adjusted.get("residuals_off", raw["margins_off"])
        )

        # 선수 정보 조회
        player_info = self.player_games_df[
            self.player_games_df["player_id"] == player_id
        ].iloc[0]

        player_name = player_info.get("player_name", f"Player {player_id}")
        if team_id is None:
            team_id = player_info.get("team_id", 0)
        team_abbr = player_info.get("team_abbr", "")

        # 표본 품질 평가
        total_games = raw["games_on"] + raw["games_off"]
        if total_games >= 50 and raw["games_off"] >= 10:
            sample_quality = "high"
        elif total_games >= 30 and raw["games_off"] >= 5:
            sample_quality = "medium"
        else:
            sample_quality = "low"

        result = OnOffResult(
            player_id=player_id,
            player_name=player_name,
            team_id=team_id,
            team_abbr=team_abbr,
            games_on=raw["games_on"],
            games_off=raw["games_off"],
            margin_on=raw["margin_on"],
            margin_off=raw["margin_off"],
            raw_impact=raw["raw_impact"],
            adjusted_margin_on=adjusted["adjusted_margin_on"],
            adjusted_margin_off=adjusted["adjusted_margin_off"],
            adjusted_impact=adjusted["adjusted_impact"],
            t_statistic=stats_result["t_statistic"],
            p_value=stats_result["p_value"],
            confidence_interval=stats_result["confidence_interval"],
            effect_size=stats_result["effect_size"],
            is_significant=stats_result["is_significant"],
            sample_quality=sample_quality,
        )

        # 캐시 저장
        self._impact_cache[cache_key] = result

        return result

    def get_all_players_impact(
        self,
        min_games: Optional[int] = None
    ) -> pd.DataFrame:
        """
        전체 선수 영향도 분석.

        Args:
            min_games: 최소 출전 경기 수 (기본: 초기화 시 설정값)

        Returns:
            선수별 영향도 DataFrame
        """
        if min_games is not None:
            self.min_games_on = min_games

        # 분석 대상 선수 목록
        player_game_counts = (
            self.player_games_df[self.player_games_df["played"] == True]
            .groupby(["player_id", "team_id"])
            .size()
            .reset_index(name="games_played")
        )

        candidates = player_game_counts[
            player_game_counts["games_played"] >= self.min_games_on
        ]

        logger.info(f"Analyzing {len(candidates)} player-team combinations...")

        results = []
        for _, row in candidates.iterrows():
            result = self.analyze_player(row["player_id"], row["team_id"])
            if result is not None:
                results.append(result)

        logger.info(f"Completed analysis for {len(results)} players")

        # DataFrame 변환
        if not results:
            return pd.DataFrame()

        df = pd.DataFrame([
            {
                "player_id": r.player_id,
                "player_name": r.player_name,
                "team_id": r.team_id,
                "team_abbr": r.team_abbr,
                "games_on": r.games_on,
                "games_off": r.games_off,
                "margin_on": r.margin_on,
                "margin_off": r.margin_off,
                "raw_impact": r.raw_impact,
                "adjusted_impact": r.adjusted_impact,
                "t_statistic": r.t_statistic,
                "p_value": r.p_value,
                "ci_lower": r.confidence_interval[0],
                "ci_upper": r.confidence_interval[1],
                "effect_size": r.effect_size,
                "is_significant": r.is_significant,
                "sample_quality": r.sample_quality,
            }
            for r in results
        ])

        # 영향도 기준 정렬
        df = df.sort_values("adjusted_impact", ascending=False)

        return df

    def get_top_impact_players(
        self,
        n: int = 20,
        significant_only: bool = False
    ) -> pd.DataFrame:
        """
        영향도 상위 선수 조회.

        Args:
            n: 상위 N명
            significant_only: 통계적으로 유의한 선수만

        Returns:
            상위 선수 DataFrame
        """
        df = self.get_all_players_impact()

        if df.empty:
            return df

        if significant_only:
            df = df[df["is_significant"] == True]

        return df.head(n)

    def save_results(self, output_path: Path):
        """분석 결과 저장"""
        df = self.get_all_players_impact()

        if df.empty:
            logger.warning("No results to save")
            return

        df.to_parquet(output_path, index=False)
        logger.info(f"Saved {len(df)} player impact results to {output_path}")

    @classmethod
    def load_results(cls, path: Path) -> pd.DataFrame:
        """저장된 분석 결과 로드"""
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path)


def create_analyzer_from_data(
    data_dir: Path,
    season: int,
    min_games: int = 10
) -> PlayerOnOffAnalyzer:
    """
    데이터 디렉토리에서 분석기 생성.

    Args:
        data_dir: 데이터 디렉토리
        season: 시즌 연도
        min_games: 최소 출전 경기 수

    Returns:
        PlayerOnOffAnalyzer 인스턴스
    """
    # 경기 데이터 로드
    games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"
    games_df = pd.read_parquet(games_path) if games_path.exists() else pd.DataFrame()

    # 선수 경기 데이터 로드
    player_games_path = data_dir / "raw" / "nba_stats" / "player_games" / f"season_{season}.parquet"
    player_games_df = pd.read_parquet(player_games_path) if player_games_path.exists() else pd.DataFrame()

    # 팀 EPM 데이터 로드
    team_epm_path = data_dir / "raw" / "dnt" / "team_epm" / f"season_{season}.parquet"
    team_epm_df = pd.read_parquet(team_epm_path) if team_epm_path.exists() else None

    return PlayerOnOffAnalyzer(
        games_df=games_df,
        player_games_df=player_games_df,
        team_epm_df=team_epm_df,
        min_games_on=min_games,
    )
