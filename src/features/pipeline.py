"""
Feature Pipeline.

전체 피처 생성 파이프라인을 관리합니다.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm

from src.features.base import (
    BaseFeature,
    CompositeFeature,
    FeatureContext,
    FeatureResult
)
from src.features.team_strength import TeamStrengthFeature, AdvancedTeamStrengthFeature
from src.features.player_epm import PlayerEPMFeature, DepthFeature
from src.features.rest_fatigue import RestFatigueFeature, ScheduleDensityFeature
from src.features.four_factors import FourFactorsFeature, AdvancedFourFactorsFeature
from src.features.lineup import LineupFeature

from src.utils.logger import logger
from src.utils.helpers import (
    save_dataframe,
    load_dataframe,
    get_season_from_date,
    ensure_dir
)
from config.settings import settings


@dataclass
class FeaturePipelineConfig:
    """파이프라인 설정"""
    min_games_threshold: int = 10
    rolling_windows: List[int] = None
    include_advanced: bool = False
    parallel: bool = False

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10]


class FeaturePipeline:
    """
    피처 생성 파이프라인.

    경기 데이터와 컨텍스트를 받아 전체 피처셋을 생성합니다.
    """

    def __init__(
        self,
        config: Optional[FeaturePipelineConfig] = None,
        features: Optional[List[BaseFeature]] = None
    ):
        """
        Args:
            config: 파이프라인 설정
            features: 피처 모듈 리스트 (None이면 기본 피처 사용)
        """
        self.config = config or FeaturePipelineConfig()
        self.features = features or self._create_default_features()

        # 전체 피처 이름 목록
        self._feature_names = []
        for f in self.features:
            self._feature_names.extend(f.feature_names)

        logger.info(f"Pipeline initialized with {len(self.features)} feature modules")
        logger.info(f"Total features: {len(self._feature_names)}")

    def _create_default_features(self) -> List[BaseFeature]:
        """기본 피처 모듈 생성"""
        features = [
            TeamStrengthFeature(),
            PlayerEPMFeature(),
            RestFatigueFeature(),
            FourFactorsFeature(windows=self.config.rolling_windows),
            LineupFeature(),
        ]

        if self.config.include_advanced:
            features.extend([
                AdvancedTeamStrengthFeature(),
                DepthFeature(),
                AdvancedFourFactorsFeature(),
            ])

        return features

    @property
    def feature_names(self) -> List[str]:
        """전체 피처 이름 목록"""
        return self._feature_names

    def create_context(
        self,
        game: Dict[str, Any],
        team_epm: pd.DataFrame,
        player_epm: Optional[pd.DataFrame] = None,
        games_history: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> FeatureContext:
        """
        피처 계산용 컨텍스트 생성.

        Args:
            game: 경기 정보 딕셔너리
            team_epm: 팀 EPM DataFrame
            player_epm: 선수 EPM DataFrame
            games_history: 경기 이력 DataFrame
            **kwargs: 추가 컨텍스트 데이터

        Returns:
            FeatureContext 객체
        """
        return FeatureContext(
            game_id=str(game.get("game_id", "")),
            game_date=str(game.get("game_date", "")),
            home_team_id=int(game.get("home_team_id", 0)),
            away_team_id=int(game.get("away_team_id", 0)),
            season=int(game.get("season", get_season_from_date(game.get("game_date", "2024-01-01")))),
            team_epm=team_epm,
            player_epm=player_epm,
            games_history=games_history,
            **kwargs
        )

    def compute_features(self, context: FeatureContext) -> Dict[str, float]:
        """
        단일 경기 피처 계산.

        Args:
            context: 피처 컨텍스트

        Returns:
            피처 딕셔너리
        """
        all_features = {}

        for feature_module in self.features:
            result = feature_module.compute_safe(context)
            all_features.update(result.features)

            if result.warnings:
                logger.debug(f"[{feature_module.name}] Warnings: {result.warnings}")

        return all_features

    def build_dataset(
        self,
        games: pd.DataFrame,
        team_epm: pd.DataFrame,
        player_epm: Optional[pd.DataFrame] = None,
        games_history: Optional[pd.DataFrame] = None,
        progress_callback: Optional[Callable] = None
    ) -> pd.DataFrame:
        """
        전체 경기에 대한 피처 데이터셋 생성.

        Args:
            games: 경기 DataFrame (game_id, game_date, home_team_id, away_team_id 필수)
            team_epm: 팀 EPM DataFrame
            player_epm: 선수 EPM DataFrame
            games_history: 경기 이력 DataFrame
            progress_callback: 진행률 콜백

        Returns:
            피처 DataFrame
        """
        logger.info(f"Building feature dataset for {len(games)} games...")

        # 유효 경기 필터링
        valid_games = self._filter_valid_games(games, games_history)
        logger.info(f"Valid games after filtering: {len(valid_games)}")

        results = []
        total = len(valid_games)

        for i, (_, game) in enumerate(tqdm(valid_games.iterrows(), total=total, desc="Computing features")):
            # 컨텍스트 생성
            context = self.create_context(
                game.to_dict(),
                team_epm=team_epm,
                player_epm=player_epm,
                games_history=games_history
            )

            # 피처 계산
            features = self.compute_features(context)

            # 메타데이터 추가
            features["game_id"] = context.game_id
            features["game_date"] = context.game_date
            features["home_team_id"] = context.home_team_id
            features["away_team_id"] = context.away_team_id
            features["season"] = context.season

            results.append(features)

            if progress_callback and i % 100 == 0:
                progress_callback((i + 1) / total)

        df = pd.DataFrame(results)

        logger.info(f"Feature dataset built: {df.shape}")
        return df

    def _filter_valid_games(
        self,
        games: pd.DataFrame,
        games_history: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        유효 경기 필터링 (양팀 10경기 이상).

        Args:
            games: 경기 DataFrame
            games_history: 경기 이력 DataFrame

        Returns:
            필터링된 경기 DataFrame
        """
        if games_history is None or games_history.empty:
            logger.warning("No games history provided, skipping filtering")
            return games

        # games_history에 team_id 컬럼이 없으면 게임 기반 필터링
        if "team_id" not in games_history.columns:
            # 게임 데이터만 있는 경우 - 홈/어웨이 양쪽으로 카운트
            valid_game_ids = []
            games_sorted = games.sort_values("game_date")

            # 팀별 경기 수 추적
            team_game_counts = {}

            for _, game in games_sorted.iterrows():
                game_date = game.get("game_date", "")
                home_id = game.get("home_team_id")
                away_id = game.get("away_team_id")
                game_id = game.get("game_id")

                home_count = team_game_counts.get(home_id, 0)
                away_count = team_game_counts.get(away_id, 0)

                if home_count >= self.config.min_games_threshold and \
                   away_count >= self.config.min_games_threshold:
                    valid_game_ids.append(game_id)

                # 경기 카운트 업데이트
                team_game_counts[home_id] = home_count + 1
                team_game_counts[away_id] = away_count + 1

            logger.info(f"Filtered {len(games)} -> {len(valid_game_ids)} valid games")
            return games[games["game_id"].isin(valid_game_ids)]

        # 기존 team_id 기반 필터링
        valid_game_ids = []

        for _, game in games.iterrows():
            game_date = game.get("game_date", "")
            home_id = game.get("home_team_id")
            away_id = game.get("away_team_id")

            # 홈팀 경기 수
            home_games = games_history[
                (games_history["team_id"] == home_id) &
                (games_history["game_date"] < game_date)
            ]

            # 어웨이팀 경기 수
            away_games = games_history[
                (games_history["team_id"] == away_id) &
                (games_history["game_date"] < game_date)
            ]

            if len(home_games) >= self.config.min_games_threshold and \
               len(away_games) >= self.config.min_games_threshold:
                valid_game_ids.append(game.get("game_id"))

        logger.info(f"Filtered {len(games)} -> {len(valid_game_ids)} valid games")
        return games[games["game_id"].isin(valid_game_ids)]

    def add_target(
        self,
        feature_df: pd.DataFrame,
        games: pd.DataFrame,
        target_col: str = "margin"
    ) -> pd.DataFrame:
        """
        타겟 변수(점수차) 추가.

        Args:
            feature_df: 피처 DataFrame
            games: 경기 결과가 포함된 DataFrame
            target_col: 타겟 컬럼명

        Returns:
            타겟이 추가된 DataFrame
        """
        # 게임별 점수차 계산
        if "home_score" in games.columns and "away_score" in games.columns:
            games = games.copy()
            games[target_col] = games["home_score"] - games["away_score"]

            # 병합
            result = feature_df.merge(
                games[["game_id", target_col]],
                on="game_id",
                how="left"
            )

            logger.info(f"Added target '{target_col}': {result[target_col].notna().sum()} games")
            return result

        logger.warning("Could not add target: missing score columns")
        return feature_df


class SeasonFeatureBuilder:
    """
    시즌별 피처 데이터셋 빌더.

    저장된 raw 데이터를 읽어 피처 데이터셋을 생성하고 저장합니다.
    """

    def __init__(
        self,
        data_dir: Path,
        pipeline: Optional[FeaturePipeline] = None
    ):
        """
        Args:
            data_dir: 데이터 디렉토리
            pipeline: 피처 파이프라인 (None이면 기본 생성)
        """
        self.data_dir = data_dir
        self.pipeline = pipeline or FeaturePipeline()

        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"
        ensure_dir(self.processed_dir / "features")

    def load_season_data(self, season: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        시즌 데이터 로드.

        Returns:
            (team_epm, player_epm, games_history)
        """
        team_epm_path = self.raw_dir / "dnt" / "team_epm" / f"season_{season}.parquet"
        season_epm_path = self.raw_dir / "dnt" / "season_epm" / f"season_{season}.parquet"
        games_path = self.raw_dir / "nba_stats" / "games" / f"season_{season}.parquet"

        team_epm = pd.DataFrame()
        player_epm = pd.DataFrame()
        games = pd.DataFrame()

        if team_epm_path.exists():
            team_epm = pd.read_parquet(team_epm_path)
            logger.info(f"Loaded team EPM: {len(team_epm)} records")

        if season_epm_path.exists():
            player_epm = pd.read_parquet(season_epm_path)
            logger.info(f"Loaded player EPM: {len(player_epm)} records")

        if games_path.exists():
            games = pd.read_parquet(games_path)
            logger.info(f"Loaded games: {len(games)} records")

        return team_epm, player_epm, games

    def build_season_features(self, season: int) -> pd.DataFrame:
        """
        단일 시즌 피처 데이터셋 생성.

        Args:
            season: 시즌 연도

        Returns:
            피처 DataFrame
        """
        logger.info(f"Building features for season {season}...")

        # 데이터 로드
        team_epm, player_epm, games = self.load_season_data(season)

        if team_epm.empty:
            logger.error(f"No team EPM data for season {season}")
            return pd.DataFrame()

        if games.empty:
            logger.warning(f"No games data for season {season}, using team EPM dates")
            # team_epm에서 고유 경기 추출
            games = self._extract_games_from_team_epm(team_epm)

        # 피처 생성
        feature_df = self.pipeline.build_dataset(
            games=games,
            team_epm=team_epm,
            player_epm=player_epm if not player_epm.empty else None,
            games_history=games
        )

        # 저장
        output_path = self.processed_dir / "features" / f"season_{season}.parquet"
        feature_df.to_parquet(output_path, index=False)
        logger.info(f"Saved features to {output_path}")

        return feature_df

    def build_all_seasons(self, seasons: List[int]) -> pd.DataFrame:
        """
        다중 시즌 피처 데이터셋 생성 및 병합.

        Args:
            seasons: 시즌 리스트

        Returns:
            병합된 피처 DataFrame
        """
        all_features = []

        for season in seasons:
            df = self.build_season_features(season)
            if not df.empty:
                all_features.append(df)

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)

        # 저장
        output_path = self.processed_dir / "features" / "all_seasons.parquet"
        combined.to_parquet(output_path, index=False)
        logger.info(f"Saved combined features: {combined.shape}")

        return combined

    def _extract_games_from_team_epm(self, team_epm: pd.DataFrame) -> pd.DataFrame:
        """team_epm에서 경기 정보 추출 (임시)"""
        # 날짜별 유니크 팀 쌍을 경기로 추정
        # 실제로는 NBA Stats에서 경기 정보를 가져와야 함

        games = []
        for date in team_epm["game_dt"].unique():
            day_data = team_epm[team_epm["game_dt"] == date]
            teams = day_data["team_id"].tolist()

            # 2팀씩 경기로 가정 (실제로는 정확하지 않음)
            for i in range(0, len(teams) - 1, 2):
                games.append({
                    "game_id": f"{date}_{teams[i]}_{teams[i+1]}",
                    "game_date": date,
                    "home_team_id": teams[i],
                    "away_team_id": teams[i+1],
                    "season": get_season_from_date(date)
                })

        return pd.DataFrame(games)


# ===================
# Convenience Functions
# ===================

def create_pipeline(
    include_advanced: bool = False,
    rolling_windows: List[int] = None
) -> FeaturePipeline:
    """
    피처 파이프라인 팩토리 함수.

    Args:
        include_advanced: 고급 피처 포함 여부
        rolling_windows: 롤링 윈도우 크기

    Returns:
        FeaturePipeline 인스턴스
    """
    config = FeaturePipelineConfig(
        include_advanced=include_advanced,
        rolling_windows=rolling_windows or [5, 10]
    )
    return FeaturePipeline(config)


def build_features_for_season(
    season: int,
    data_dir: Path = None
) -> pd.DataFrame:
    """
    시즌 피처 빌드 헬퍼 함수.

    Args:
        season: 시즌 연도
        data_dir: 데이터 디렉토리 (None이면 설정 사용)

    Returns:
        피처 DataFrame
    """
    data_dir = data_dir or settings.data_dir
    builder = SeasonFeatureBuilder(data_dir)
    return builder.build_season_features(season)
