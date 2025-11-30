#!/usr/bin/env python3
"""
NBA Score Prediction Model Training Script.

전체 학습 파이프라인을 실행합니다:
1. 데이터 로드 및 전처리
2. 피처 생성
3. 모델 학습 (XGBoost, LightGBM, Ridge)
4. 하이퍼파라미터 최적화
5. 앙상블 구성
6. 평가 및 저장
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger
from src.features.pipeline import FeaturePipeline, FeaturePipelineConfig, SeasonFeatureBuilder
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.models.base import EnsembleModel, compare_models
from src.optimization.optimizer import XGBoostOptimizer, LightGBMOptimizer, RidgeOptimizer, EnsembleOptimizer
from src.evaluation.metrics import calculate_metrics, MetricsReport
from src.evaluation.cross_validation import TimeSeriesCV, SeasonBasedCV, cross_validate_model
from src.evaluation.analysis import ErrorAnalyzer, FeatureImportanceAnalyzer
from config.settings import settings


def load_data(data_dir: Path, seasons: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    데이터 로드.

    Returns:
        (team_epm_df, games_df)
    """
    logger.info(f"Loading data for seasons: {seasons}")

    team_epm_dfs = []
    games_dfs = []

    for season in seasons:
        # Team EPM 데이터
        team_epm_path = data_dir / "raw" / "dnt" / "team_epm" / f"season_{season}.parquet"
        if team_epm_path.exists():
            df = pd.read_parquet(team_epm_path)
            df["season"] = season
            team_epm_dfs.append(df)
            logger.info(f"  Season {season} team_epm: {len(df)} records")

        # 게임 데이터 (NBA Stats API에서)
        games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"
        if games_path.exists():
            df = pd.read_parquet(games_path)
            games_dfs.append(df)
            logger.info(f"  Season {season} games: {len(df)} records")

    if team_epm_dfs:
        team_epm = pd.concat(team_epm_dfs, ignore_index=True)
    else:
        team_epm = pd.DataFrame()

    if games_dfs:
        games = pd.concat(games_dfs, ignore_index=True)
    else:
        games = pd.DataFrame()

    return team_epm, games


def extract_games_from_team_epm(team_epm: pd.DataFrame) -> pd.DataFrame:
    """team_epm 데이터에서 경기 정보 추출"""
    if team_epm.empty:
        return pd.DataFrame()

    games = []
    game_id_counter = 0

    # 날짜별 그룹화
    for date in team_epm["game_dt"].unique():
        day_data = team_epm[team_epm["game_dt"] == date]

        # 각 팀의 상대팀 찾기
        processed_teams = set()

        for _, row in day_data.iterrows():
            team_id = row["team_id"]
            opp_team_id = row.get("opp_team_id")

            if team_id in processed_teams:
                continue

            if opp_team_id:
                processed_teams.add(team_id)
                processed_teams.add(opp_team_id)

                # 홈/어웨이 구분 (실제로는 추가 정보 필요)
                # 여기서는 임의로 지정
                home_id = team_id
                away_id = opp_team_id

                # 점수 계산 (실제 데이터에서 가져와야 함)
                team_row = day_data[day_data["team_id"] == team_id].iloc[0]
                opp_row = day_data[day_data["team_id"] == opp_team_id]

                # team_epm에서 점수 정보 추출
                home_score = team_row.get("pts", 0)
                away_score = team_row.get("opp_pts", 0)

                games.append({
                    "game_id": f"{date}_{home_id}_{away_id}",
                    "game_date": date,
                    "home_team_id": home_id,
                    "away_team_id": away_id,
                    "home_score": home_score,
                    "away_score": away_score,
                    "margin": home_score - away_score,
                    "season": team_row.get("season", 2024)
                })

                game_id_counter += 1

    games_df = pd.DataFrame(games)

    if not games_df.empty:
        games_df = games_df.sort_values("game_date").reset_index(drop=True)

    logger.info(f"Extracted {len(games_df)} games from team_epm data")

    return games_df


def build_features(
    games: pd.DataFrame,
    team_epm: pd.DataFrame,
    config: FeaturePipelineConfig = None
) -> pd.DataFrame:
    """피처 데이터셋 생성"""
    logger.info("Building feature dataset...")

    pipeline = FeaturePipeline(config or FeaturePipelineConfig())

    # 피처 계산
    feature_df = pipeline.build_dataset(
        games=games,
        team_epm=team_epm,
        games_history=games
    )

    # 타겟 추가
    feature_df = pipeline.add_target(feature_df, games)

    logger.info(f"Feature dataset: {feature_df.shape}")

    return feature_df


def prepare_data(
    feature_df: pd.DataFrame,
    target_col: str = "margin",
    exclude_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    학습 데이터 준비.

    Returns:
        (X, y, feature_names)
    """
    exclude_cols = exclude_cols or [
        "game_id", "game_date", "home_team_id", "away_team_id",
        "season", "margin", "home_score", "away_score"
    ]

    # 피처 컬럼 선택
    feature_cols = [c for c in feature_df.columns if c not in exclude_cols]

    # NaN 처리
    X = feature_df[feature_cols].copy()
    y = feature_df[target_col].copy()

    # NaN 제거
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]

    logger.info(f"Prepared data: X={X.shape}, y={len(y)}")
    logger.info(f"Features: {feature_cols}")

    return X, y, feature_cols


def split_by_season(
    feature_df: pd.DataFrame,
    train_seasons: List[int],
    val_season: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """시즌 기준 분할"""
    train_df = feature_df[feature_df["season"].isin(train_seasons)]
    val_df = feature_df[feature_df["season"] == val_season]

    logger.info(f"Train: {len(train_df)} games from seasons {train_seasons}")
    logger.info(f"Val: {len(val_df)} games from season {val_season}")

    return train_df, val_df


def train_baseline_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, object]:
    """기본 모델 학습"""
    models = {}

    # XGBoost
    logger.info("Training XGBoost...")
    xgb_model = XGBoostModel(use_huber=True, huber_delta=10.0)
    xgb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["xgboost"] = xgb_model

    # LightGBM
    logger.info("Training LightGBM...")
    lgb_model = LightGBMModel(use_huber=True, huber_delta=10.0)
    lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["lightgbm"] = lgb_model

    # Ridge
    logger.info("Training Ridge...")
    ridge_model = RidgeModel(
        alphas_for_cv=[0.01, 0.1, 1.0, 10.0, 100.0],
        normalize_features=True
    )
    ridge_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["ridge"] = ridge_model

    return models


def optimize_hyperparameters(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 50
) -> Dict[str, object]:
    """하이퍼파라미터 최적화"""
    logger.info(f"Starting hyperparameter optimization ({n_trials} trials each)...")

    optimized_models = {}

    # XGBoost 최적화
    xgb_optimizer = XGBoostOptimizer(n_trials=n_trials, cv_folds=5)
    xgb_result = xgb_optimizer.optimize(X_train, y_train, show_progress=True)
    optimized_models["xgboost"] = xgb_optimizer.get_best_model()
    logger.info(f"XGBoost best RMSE: {xgb_result.best_score:.4f}")

    # LightGBM 최적화
    lgb_optimizer = LightGBMOptimizer(n_trials=n_trials, cv_folds=5)
    lgb_result = lgb_optimizer.optimize(X_train, y_train, show_progress=True)
    optimized_models["lightgbm"] = lgb_optimizer.get_best_model()
    logger.info(f"LightGBM best RMSE: {lgb_result.best_score:.4f}")

    # Ridge 최적화
    ridge_optimizer = RidgeOptimizer(n_trials=n_trials, cv_folds=5)
    ridge_result = ridge_optimizer.optimize(X_train, y_train, show_progress=True)
    optimized_models["ridge"] = ridge_optimizer.get_best_model()
    logger.info(f"Ridge best RMSE: {ridge_result.best_score:.4f}")

    return optimized_models


def train_optimized_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> Dict[str, object]:
    """최적화된 파라미터로 모델 재학습"""
    logger.info("Training models with optimized parameters...")

    for name, model in models.items():
        model.fit(X_train, y_train, eval_set=(X_val, y_val))

    return models


def create_ensemble(
    models: Dict[str, object],
    X_val: pd.DataFrame,
    y_val: pd.Series
) -> EnsembleModel:
    """앙상블 모델 생성"""
    logger.info("Creating ensemble model...")

    model_list = list(models.values())

    # 앙상블 가중치 최적화
    ensemble_optimizer = EnsembleOptimizer(model_list, n_trials=100)
    weights, best_rmse = ensemble_optimizer.optimize(X_val, y_val)

    # 앙상블 모델 생성
    ensemble = EnsembleModel(model_list, weights=weights)

    logger.info(f"Ensemble weights: {dict(zip(models.keys(), weights))}")
    logger.info(f"Ensemble RMSE: {best_rmse:.4f}")

    return ensemble


def evaluate_models(
    models: Dict[str, object],
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """모델 평가"""
    logger.info("Evaluating models...")

    results = []

    for name, model in models.items():
        metrics = model.evaluate(X_test, y_test)

        results.append({
            "model": name,
            "rmse": metrics.rmse,
            "mae": metrics.mae,
            "win_accuracy": metrics.win_accuracy,
            "within_5": metrics.within_5_accuracy,
            "within_10": metrics.within_10_accuracy
        })

        logger.info(f"{name}: {metrics.summary()}")

    results_df = pd.DataFrame(results).sort_values("rmse")

    return results_df


def save_models(
    models: Dict[str, object],
    output_dir: Path
) -> None:
    """모델 저장"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        model_dir = output_dir / name
        model.save(model_dir)
        logger.info(f"Saved {name} to {model_dir}")


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Train NBA Score Prediction Models")
    parser.add_argument("--data-dir", type=str, default=str(settings.data_dir),
                        help="Data directory path")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2023, 2024],
                        help="Training seasons")
    parser.add_argument("--val-season", type=int, default=2025,
                        help="Validation season")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of optimization trials")
    parser.add_argument("--skip-optimization", action="store_true",
                        help="Skip hyperparameter optimization")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for models")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "models"

    logger.info("=" * 60)
    logger.info("NBA Score Prediction Model Training")
    logger.info("=" * 60)
    logger.info(f"Training seasons: {args.train_seasons}")
    logger.info(f"Validation season: {args.val_season}")

    # 1. 데이터 로드
    all_seasons = args.train_seasons + [args.val_season]
    team_epm, games = load_data(data_dir, all_seasons)

    if team_epm.empty or games.empty:
        logger.error("No data loaded. Please run data collection first.")
        return

    # 2. 피처 생성
    feature_df = build_features(games, team_epm)

    if feature_df.empty:
        logger.error("No features generated.")
        return

    # 3. 데이터 분할
    train_df, val_df = split_by_season(feature_df, args.train_seasons, args.val_season)

    X_train, y_train, feature_names = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)

    if len(X_train) == 0 or len(X_val) == 0:
        logger.error("Insufficient data for training.")
        return

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Validation set: {len(X_val)} samples")

    # 4. 기본 모델 학습
    logger.info("\n" + "=" * 60)
    logger.info("Phase 1: Baseline Model Training")
    logger.info("=" * 60)

    baseline_models = train_baseline_models(X_train, y_train, X_val, y_val)

    # 기본 모델 평가
    logger.info("\nBaseline Model Evaluation:")
    baseline_results = evaluate_models(baseline_models, X_val, y_val)
    print(baseline_results.to_string(index=False))

    # 5. 하이퍼파라미터 최적화 (선택적)
    if not args.skip_optimization:
        logger.info("\n" + "=" * 60)
        logger.info("Phase 2: Hyperparameter Optimization")
        logger.info("=" * 60)

        optimized_models = optimize_hyperparameters(X_train, y_train, args.n_trials)
        optimized_models = train_optimized_models(optimized_models, X_train, y_train, X_val, y_val)

        logger.info("\nOptimized Model Evaluation:")
        optimized_results = evaluate_models(optimized_models, X_val, y_val)
        print(optimized_results.to_string(index=False))

        final_models = optimized_models
    else:
        final_models = baseline_models

    # 6. 앙상블 생성
    logger.info("\n" + "=" * 60)
    logger.info("Phase 3: Ensemble Creation")
    logger.info("=" * 60)

    ensemble = create_ensemble(final_models, X_val, y_val)
    final_models["ensemble"] = ensemble

    # 7. 최종 평가
    logger.info("\n" + "=" * 60)
    logger.info("Final Evaluation")
    logger.info("=" * 60)

    final_results = evaluate_models(final_models, X_val, y_val)
    print("\n" + final_results.to_string(index=False))

    # 성공 기준 확인
    best_model = final_results.iloc[0]
    rmse = best_model["rmse"]
    win_acc = best_model["win_accuracy"]

    logger.info("\n" + "=" * 60)
    logger.info("Success Criteria Check")
    logger.info("=" * 60)
    logger.info(f"Target: RMSE < 11.5, Win Accuracy > 66%")
    logger.info(f"Best Model: {best_model['model']}")
    logger.info(f"  RMSE: {rmse:.4f} {'✓' if rmse < 11.5 else '✗'}")
    logger.info(f"  Win Accuracy: {win_acc:.2%} {'✓' if win_acc > 0.66 else '✗'}")

    if rmse < 11.5 and win_acc > 0.66:
        logger.info("\n✓ SUCCESS: All criteria met!")
    else:
        logger.info("\n✗ NEEDS IMPROVEMENT")

    # 8. 피처 중요도 분석
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance Analysis")
    logger.info("=" * 60)

    analyzer = FeatureImportanceAnalyzer([final_models[k] for k in ["xgboost", "lightgbm", "ridge"]])
    importance_df = analyzer.get_importance_dataframe()

    print("\nTop 15 Features:")
    print(importance_df.head(15).to_string())

    # 9. 모델 저장
    logger.info("\n" + "=" * 60)
    logger.info("Saving Models")
    logger.info("=" * 60)

    save_models(final_models, output_dir)

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
