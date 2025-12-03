"""
예측 파이프라인 서비스.

리팩토링 Phase 3: Facade 패턴 적용.

main.py의 복잡한 예측 로직을 단순화합니다.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from app.services.data_loader import DataLoader
from app.services.predictor_v5 import V5PredictionService
from app.services.injury_adjuster import InjuryAdjuster
from app.models.data_types import (
    TeamInfo, InjurySummary, PlayerInjuryDetail, PredictionResult
)
from config.constants import TEAM_INFO


@dataclass
class GamePrediction:
    """단일 경기 예측 결과"""
    game_id: str
    game_time: str
    game_status: int  # 1=예정, 2=진행중, 3=종료

    # 팀 정보
    home_team_id: int
    away_team_id: int
    home_abbr: str
    away_abbr: str
    home_name: str
    away_name: str
    home_color: str
    away_color: str

    # 점수 (종료/진행중 경기)
    home_score: Optional[int] = None
    away_score: Optional[int] = None

    # B2B 정보
    home_b2b: bool = False
    away_b2b: bool = False

    # 예측 결과
    base_prob: float = 0.5
    adjusted_prob: float = 0.5
    predicted_margin: float = 0.0
    features: Dict[str, float] = field(default_factory=dict)

    # 부상 정보
    home_injury_summary: Optional[Dict] = None
    away_injury_summary: Optional[Dict] = None
    home_prob_shift: float = 0.0
    away_prob_shift: float = 0.0

    # 배당 정보
    odds_info: Optional[Dict] = None

    @property
    def predicted_winner(self) -> str:
        """예측 승자 ('home' 또는 'away')"""
        return 'home' if self.adjusted_prob >= 0.5 else 'away'

    @property
    def confidence(self) -> float:
        """신뢰도 (0~1)"""
        return abs(self.adjusted_prob - 0.5) * 2

    @property
    def is_high_confidence(self) -> bool:
        """고신뢰도 여부 (70% 이상)"""
        return self.adjusted_prob >= 0.7 or self.adjusted_prob <= 0.3


class PredictionPipeline:
    """
    예측 파이프라인.

    Facade 패턴으로 복잡한 예측 로직을 캡슐화합니다.

    사용 예시:
        pipeline = PredictionPipeline(data_dir, model_dir)
        predictions = pipeline.predict_games(game_date)
    """

    def __init__(self, data_dir: Path, model_dir: Path):
        """
        Args:
            data_dir: 데이터 디렉토리
            model_dir: 모델 디렉토리
        """
        self.data_dir = data_dir
        self.model_dir = model_dir
        self._loader: Optional[DataLoader] = None
        self._predictor: Optional[V5PredictionService] = None

    @property
    def loader(self) -> DataLoader:
        """데이터 로더 (지연 로딩)"""
        if self._loader is None:
            self._loader = DataLoader(self.data_dir)
        return self._loader

    @property
    def predictor(self) -> V5PredictionService:
        """예측 서비스 (지연 로딩)"""
        if self._predictor is None:
            self._predictor = V5PredictionService(self.model_dir)
        return self._predictor

    def predict_games(
        self,
        game_date: date,
        include_injuries: bool = True,
        include_odds: bool = True
    ) -> List[GamePrediction]:
        """
        날짜별 전체 경기 예측.

        Args:
            game_date: 경기 날짜
            include_injuries: 부상 정보 포함 여부
            include_odds: 배당 정보 포함 여부

        Returns:
            GamePrediction 리스트
        """
        # 팀 EPM 로드
        team_epm = self.loader.load_team_epm(game_date)
        if not team_epm:
            return []

        # 경기 목록 조회
        games = self.loader.get_games(game_date)
        if not games:
            return []

        predictions = []
        for game in games:
            prediction = self.predict_single_game(
                game=game,
                game_date=game_date,
                team_epm=team_epm,
                include_injuries=include_injuries,
                include_odds=include_odds,
            )
            predictions.append(prediction)

        return predictions

    def predict_single_game(
        self,
        game: Dict,
        game_date: date,
        team_epm: Dict[int, Dict],
        include_injuries: bool = True,
        include_odds: bool = True,
    ) -> GamePrediction:
        """
        단일 경기 예측.

        Args:
            game: 경기 정보 딕셔너리
            game_date: 경기 날짜
            team_epm: 팀 EPM 데이터
            include_injuries: 부상 정보 포함 여부
            include_odds: 배당 정보 포함 여부

        Returns:
            GamePrediction
        """
        home_id = game["home_team_id"]
        away_id = game["away_team_id"]

        # 팀 정보
        home_info = TEAM_INFO.get(home_id, {})
        away_info = TEAM_INFO.get(away_id, {})

        home_abbr = home_info.get("abbr", "UNK")
        away_abbr = away_info.get("abbr", "UNK")

        # 피처 생성
        features = self.loader.build_v5_4_features(
            home_id, away_id, team_epm, game_date
        )

        # 기본 예측
        base_prob = self.predictor.predict_proba(features)

        # 경기 상태
        game_status = game.get("game_status", 1)

        # 부상 분석 (예정된 경기만)
        home_injury_summary = None
        away_injury_summary = None
        home_prob_shift = 0.0
        away_prob_shift = 0.0

        if include_injuries and game_status == 1:
            try:
                home_injury_summary = self.loader.get_injury_summary(
                    home_abbr, game_date, team_epm
                )
                away_injury_summary = self.loader.get_injury_summary(
                    away_abbr, game_date, team_epm
                )
                home_prob_shift = home_injury_summary.get("total_prob_shift", 0.0)
                away_prob_shift = away_injury_summary.get("total_prob_shift", 0.0)
            except Exception:
                pass  # 부상 분석 실패 시 무시

        # 부상 보정 적용
        adjusted_prob = self.predictor.apply_injury_adjustment(
            base_prob, home_prob_shift, away_prob_shift
        )

        # 마진 계산
        from scipy.stats import norm
        raw_margin = norm.ppf(adjusted_prob) * 12.0
        # 가비지 타임 압축
        if abs(adjusted_prob - 0.5) > 0.25:
            predicted_margin = raw_margin * 0.85
        else:
            predicted_margin = raw_margin

        # 배당 정보 (예정된 경기만)
        odds_info = None
        if include_odds and game_status == 1:
            odds_info = self.loader.get_game_odds(home_abbr, away_abbr)

        return GamePrediction(
            game_id=game.get("game_id", f"{home_abbr}_{away_abbr}"),
            game_time=game.get("game_time", ""),
            game_status=game_status,
            home_team_id=home_id,
            away_team_id=away_id,
            home_abbr=home_abbr,
            away_abbr=away_abbr,
            home_name=home_info.get("name", "Unknown"),
            away_name=away_info.get("name", "Unknown"),
            home_color=home_info.get("color", "#666666"),
            away_color=away_info.get("color", "#666666"),
            home_score=game.get("home_score"),
            away_score=game.get("away_score"),
            home_b2b=game.get("home_b2b", False),
            away_b2b=game.get("away_b2b", False),
            base_prob=base_prob,
            adjusted_prob=adjusted_prob,
            predicted_margin=round(predicted_margin, 1),
            features=features,
            home_injury_summary=home_injury_summary,
            away_injury_summary=away_injury_summary,
            home_prob_shift=home_prob_shift,
            away_prob_shift=away_prob_shift,
            odds_info=odds_info,
        )

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return self.predictor.get_model_info()

    def clear_cache(self) -> None:
        """캐시 초기화"""
        if self._loader:
            self._loader.clear_cache()
