"""
Application settings and configuration.

이 모듈은 프로젝트 전역 설정을 관리합니다.
Pydantic v2를 사용하여 타입 안전성과 환경변수 로딩을 처리합니다.
"""

from pathlib import Path
from typing import List, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    애플리케이션 전역 설정.

    환경변수 또는 .env 파일에서 설정을 로드합니다.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # ===================
    # API Keys
    # ===================
    dnt_api_key: str = Field(
        ...,
        description="Dunks and Threes API key"
    )

    # ===================
    # API Settings
    # ===================
    dnt_base_url: str = Field(
        default="https://dunksandthrees.com/api/v1",
        description="D&T API base URL"
    )
    dnt_rate_limit: int = Field(
        default=90,
        description="D&T API rate limit (requests per minute)"
    )
    dnt_season_rate_limit: int = Field(
        default=3,
        description="D&T API rate limit for season queries"
    )
    dnt_timeout: int = Field(
        default=30,
        description="API request timeout in seconds"
    )
    dnt_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests"
    )

    # ===================
    # Model Settings
    # ===================
    random_seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    min_games_threshold: int = Field(
        default=10,
        description="Minimum games played for valid predictions"
    )

    # ===================
    # Seasons Configuration
    # ===================
    training_seasons: List[int] = Field(
        default=[2023, 2024],
        description="Seasons for training (2023=22-23, 2024=23-24)"
    )
    validation_season: int = Field(
        default=2025,
        description="Season for validation (2025=24-25)"
    )

    # ===================
    # Feature Settings
    # ===================
    rolling_windows: List[int] = Field(
        default=[5, 10],
        description="Rolling window sizes for feature calculation"
    )

    # ===================
    # Paths (computed)
    # ===================
    @property
    def project_root(self) -> Path:
        """프로젝트 루트 디렉토리"""
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        """데이터 디렉토리"""
        return self.project_root / "data"

    @property
    def raw_data_dir(self) -> Path:
        """원본 데이터 디렉토리"""
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        """처리된 데이터 디렉토리"""
        return self.data_dir / "processed"

    @property
    def external_data_dir(self) -> Path:
        """외부 데이터 디렉토리"""
        return self.data_dir / "external"

    @property
    def model_dir(self) -> Path:
        """모델 저장 디렉토리"""
        return self.project_root / "models"

    @property
    def dnt_data_dir(self) -> Path:
        """D&T API 데이터 디렉토리"""
        return self.raw_data_dir / "dnt"

    @property
    def nba_stats_data_dir(self) -> Path:
        """NBA Stats 데이터 디렉토리"""
        return self.raw_data_dir / "nba_stats"

    # ===================
    # Validation
    # ===================
    @field_validator('dnt_rate_limit')
    @classmethod
    def validate_rate_limit(cls, v: int) -> int:
        if v <= 0 or v > 100:
            raise ValueError("Rate limit must be between 1 and 100")
        return v

    @field_validator('min_games_threshold')
    @classmethod
    def validate_min_games(cls, v: int) -> int:
        if v < 1 or v > 20:
            raise ValueError("Min games threshold must be between 1 and 20")
        return v


class ModelConfig:
    """
    모델 학습 관련 설정.

    하이퍼파라미터 탐색 범위 및 기본값을 정의합니다.
    """

    # XGBoost 기본 파라미터
    XGBOOST_DEFAULT_PARAMS = {
        'objective': 'reg:pseudohubererror',
        'huber_slope': 10,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }

    # LightGBM 기본 파라미터
    LIGHTGBM_DEFAULT_PARAMS = {
        'objective': 'huber',
        'alpha': 10,  # huber delta
        'num_leaves': 31,
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 500,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    # XGBoost 탐색 공간
    XGBOOST_SEARCH_SPACE = {
        'max_depth': (3, 8),
        'learning_rate': (0.01, 0.1),
        'n_estimators': (100, 1000),
        'min_child_weight': (1, 10),
        'subsample': (0.6, 0.9),
        'colsample_bytree': (0.6, 0.9),
        'reg_alpha': (1e-3, 1.0),
        'reg_lambda': (1e-3, 10.0),
        'gamma': (0, 1.0)
    }

    # LightGBM 탐색 공간
    LIGHTGBM_SEARCH_SPACE = {
        'num_leaves': (15, 127),
        'max_depth': (4, 10),
        'learning_rate': (0.01, 0.1),
        'n_estimators': (100, 1000),
        'min_child_samples': (5, 50),
        'subsample': (0.6, 0.9),
        'colsample_bytree': (0.6, 0.9),
        'reg_alpha': (1e-3, 1.0),
        'reg_lambda': (1e-3, 10.0)
    }

    # Optuna 설정
    OPTUNA_N_TRIALS = 100
    EARLY_STOPPING_ROUNDS = 50

    # 교차검증 설정
    CV_N_SPLITS = 5


class EvaluationConfig:
    """평가 관련 설정"""

    # 성공 기준
    RMSE_MINIMUM = 12.0
    RMSE_TARGET = 11.5
    RMSE_STRETCH = 11.0

    MAE_MINIMUM = 9.5
    MAE_TARGET = 9.0
    MAE_STRETCH = 8.5

    WIN_ACC_MINIMUM = 0.63
    WIN_ACC_TARGET = 0.66
    WIN_ACC_STRETCH = 0.68

    WITHIN_5_TARGET = 0.35
    WITHIN_10_TARGET = 0.65


@lru_cache()
def get_settings() -> Settings:
    """
    설정 싱글톤 인스턴스를 반환합니다.

    캐싱을 통해 반복 로딩을 방지합니다.
    """
    return Settings()


# 전역 설정 인스턴스
settings = get_settings()
