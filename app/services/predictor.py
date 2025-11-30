"""
예측 서비스 모듈.

학습된 Ridge 모델을 로드하고 경기 예측을 수행합니다.
"""

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class GamePrediction:
    """경기 예측 결과"""
    game_id: str
    game_date: str
    game_time: str
    home_team: str
    away_team: str
    home_team_id: int
    away_team_id: int
    predicted_margin: float  # 홈팀 기준 점수차
    home_win_prob: float
    features: Dict[str, float]


@dataclass
class AdjustedGamePrediction(GamePrediction):
    """부상 조정된 예측 결과"""
    original_margin: float
    adjusted_margin: float
    home_injuries: List[Dict]
    away_injuries: List[Dict]
    home_injury_impact: float
    away_injury_impact: float


class PredictionService:
    """예측 서비스"""

    MARGIN_STD = 12.0  # 승률 변환용 표준편차

    def __init__(self, model_dir: Path):
        """
        Args:
            model_dir: 모델 디렉토리 (data/models/final)
        """
        self.model_dir = model_dir
        self.model = None
        self.feature_names = None
        self.metadata = None

        self._load_model()

    def _load_model(self) -> None:
        """모델 및 메타데이터 로드"""
        # Ridge 모델 로드
        model_path = self.model_dir / "ridge" / "model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)

        # 피처명 로드
        feature_path = self.model_dir / "feature_names.json"
        if feature_path.exists():
            with open(feature_path) as f:
                self.feature_names = json.load(f)

        # 메타데이터 로드
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)

    def predict_margin(self, features: Dict[str, float]) -> float:
        """
        점수차 예측.

        Args:
            features: 피처 딕셔너리

        Returns:
            예측 홈팀 점수차
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # 피처 순서 맞추기 (DataFrame으로 전달)
        X = pd.DataFrame([[features.get(f, 0.0) for f in self.feature_names]], columns=self.feature_names)
        return float(self.model.predict(X)[0])

    def margin_to_win_prob(self, margin: float) -> float:
        """
        점수차를 승률로 변환.

        Args:
            margin: 예측 홈팀 점수차

        Returns:
            홈팀 승리 확률 (0-1)
        """
        return float(norm.cdf(margin / self.MARGIN_STD))

    def predict_game(
        self,
        game_id: str,
        game_date: str,
        game_time: str,
        home_team: str,
        away_team: str,
        home_team_id: int,
        away_team_id: int,
        features: Dict[str, float]
    ) -> GamePrediction:
        """
        단일 경기 예측.

        Args:
            game_id: 경기 ID
            game_date: 경기 날짜
            game_time: 경기 시간
            home_team: 홈팀 약어
            away_team: 원정팀 약어
            home_team_id: 홈팀 ID
            away_team_id: 원정팀 ID
            features: 피처 딕셔너리

        Returns:
            GamePrediction
        """
        margin = self.predict_margin(features)
        win_prob = self.margin_to_win_prob(margin)

        return GamePrediction(
            game_id=game_id,
            game_date=game_date,
            game_time=game_time,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            predicted_margin=round(margin, 1),
            home_win_prob=round(win_prob, 3),
            features=features
        )

    def get_model_info(self) -> Dict:
        """모델 정보 반환"""
        return {
            "model_type": "Ridge Regression",
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "training_date": self.metadata.get("training_date") if self.metadata else None,
            "n_samples": self.metadata.get("n_samples") if self.metadata else None,
            "cv_rmse": self.metadata.get("best_cv_rmse") if self.metadata else None,
        }
