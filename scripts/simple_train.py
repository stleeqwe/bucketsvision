#!/usr/bin/env python3
"""
Simple Training Script.

team_epm 데이터만 사용하여 간단한 모델을 학습합니다.
피처 파이프라인 대신 직접 피처를 구성합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import logger
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.models.base import EnsembleModel
from src.evaluation.metrics import calculate_metrics
from config.settings import settings


def load_and_merge_data(data_dir: Path, seasons: List[int]) -> pd.DataFrame:
    """
    team_epm과 games 데이터를 로드하고 병합.

    Returns:
        병합된 DataFrame
    """
    logger.info(f"Loading data for seasons: {seasons}")

    all_data = []

    for season in seasons:
        # Team EPM 데이터
        team_epm_path = data_dir / "raw" / "dnt" / "team_epm" / f"season_{season}.parquet"
        games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"

        if not team_epm_path.exists() or not games_path.exists():
            logger.warning(f"Missing data for season {season}")
            continue

        team_epm = pd.read_parquet(team_epm_path)
        games = pd.read_parquet(games_path)

        logger.info(f"Season {season}: {len(team_epm)} team_epm, {len(games)} games")

        # 날짜 형식 통일
        team_epm["game_dt"] = pd.to_datetime(team_epm["game_dt"]).dt.strftime("%Y-%m-%d")
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

        # 각 경기에 대해 피처 생성
        for _, game in games.iterrows():
            game_date = game["game_date"]
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]

            # 해당 날짜 이전의 team_epm 데이터 가져오기
            home_epm = team_epm[
                (team_epm["team_id"] == home_id) &
                (team_epm["game_dt"] < game_date)
            ].sort_values("game_dt", ascending=False)

            away_epm = team_epm[
                (team_epm["team_id"] == away_id) &
                (team_epm["game_dt"] < game_date)
            ].sort_values("game_dt", ascending=False)

            if home_epm.empty or away_epm.empty:
                continue

            # 최근 데이터 사용
            home_latest = home_epm.iloc[0]
            away_latest = away_epm.iloc[0]

            # 피처 구성
            features = {
                "game_id": game["game_id"],
                "game_date": game_date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "season": season,
                "margin": game["margin"],

                # Team EPM 피처
                "team_epm_diff": home_latest["team_epm"] - away_latest["team_epm"],
                "team_oepm_diff": home_latest["team_oepm"] - away_latest["team_oepm"],
                "team_depm_diff": home_latest["team_depm"] - away_latest["team_depm"],

                # Game Optimized EPM
                "team_epm_go_diff": home_latest["team_epm_game_optimized"] - away_latest["team_epm_game_optimized"],
                "team_oepm_go_diff": home_latest["team_oepm_game_optimized"] - away_latest["team_oepm_game_optimized"],
                "team_depm_go_diff": home_latest["team_depm_game_optimized"] - away_latest["team_depm_game_optimized"],

                # SOS (Strength of Schedule)
                "sos_diff": home_latest["sos"] - away_latest["sos"],
                "sos_o_diff": home_latest["sos_o"] - away_latest["sos_o"],
                "sos_d_diff": home_latest["sos_d"] - away_latest["sos_d"],

                # Ranking 피처
                "team_epm_rk_diff": home_latest["team_epm_rk"] - away_latest["team_epm_rk"],
                "team_oepm_rk_diff": home_latest["team_oepm_rk"] - away_latest["team_oepm_rk"],
                "team_depm_rk_diff": home_latest["team_depm_rk"] - away_latest["team_depm_rk"],

                # Z-score 피처
                "team_epm_z_diff": home_latest["team_epm_z"] - away_latest["team_epm_z"],
                "team_oepm_z_diff": home_latest["team_oepm_z"] - away_latest["team_oepm_z"],
                "team_depm_z_diff": home_latest["team_depm_z"] - away_latest["team_depm_z"],

                # 홈 어드밴티지 (상수)
                "home_advantage": 3.0,  # NBA 평균 홈 어드밴티지
            }

            all_data.append(features)

    df = pd.DataFrame(all_data)
    df = df.sort_values("game_date").reset_index(drop=True)

    logger.info(f"Total samples: {len(df)}")

    return df


def prepare_data(
    df: pd.DataFrame,
    target_col: str = "margin"
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """학습 데이터 준비"""
    exclude_cols = ["game_id", "game_date", "home_team_id", "away_team_id", "season", "margin"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # NaN 처리
    X = X.fillna(0)

    logger.info(f"Features: {feature_cols}")
    logger.info(f"X shape: {X.shape}, y shape: {y.shape}")

    return X, y, feature_cols


def main():
    """메인 실행 함수"""
    import argparse

    parser = argparse.ArgumentParser(description="Simple NBA Model Training")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2024],
                        help="Training seasons")
    parser.add_argument("--val-season", type=int, default=2025,
                        help="Validation season")

    args = parser.parse_args()

    data_dir = settings.data_dir

    logger.info("=" * 60)
    logger.info("Simple NBA Score Prediction Training")
    logger.info("=" * 60)

    # 1. 데이터 로드 및 병합
    all_seasons = args.train_seasons + [args.val_season]
    df = load_and_merge_data(data_dir, all_seasons)

    if df.empty:
        logger.error("No data loaded")
        return

    # 2. 학습/검증 분할
    train_df = df[df["season"].isin(args.train_seasons)]
    val_df = df[df["season"] == args.val_season]

    logger.info(f"Train: {len(train_df)} samples")
    logger.info(f"Val: {len(val_df)} samples")

    X_train, y_train, feature_names = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)

    # 3. 모델 학습
    logger.info("\n" + "=" * 60)
    logger.info("Training Models")
    logger.info("=" * 60)

    models = {}

    # XGBoost
    logger.info("\nTraining XGBoost...")
    xgb_model = XGBoostModel(use_huber=True, huber_delta=10.0)
    xgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    models["xgboost"] = xgb_model

    # LightGBM
    logger.info("Training LightGBM...")
    lgb_model = LightGBMModel(use_huber=True, huber_delta=10.0)
    lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    models["lightgbm"] = lgb_model

    # Ridge
    logger.info("Training Ridge...")
    ridge_model = RidgeModel(
        alphas_for_cv=[0.01, 0.1, 1.0, 10.0, 100.0],
        normalize_features=True
    )
    ridge_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    models["ridge"] = ridge_model

    # 4. 평가
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val.values, y_pred, name)

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
    print("\n" + results_df.to_string(index=False))

    # 5. 앙상블
    logger.info("\n" + "=" * 60)
    logger.info("Ensemble Model")
    logger.info("=" * 60)

    # 간단한 평균 앙상블
    predictions = np.column_stack([m.predict(X_val) for m in models.values()])
    ensemble_pred = np.mean(predictions, axis=1)

    ensemble_metrics = calculate_metrics(y_val.values, ensemble_pred, "ensemble")
    logger.info(f"Ensemble: {ensemble_metrics.summary()}")

    # 6. 성공 기준 확인
    best_rmse = min(r["rmse"] for r in results)
    best_win_acc = max(r["win_accuracy"] for r in results)

    logger.info("\n" + "=" * 60)
    logger.info("Success Criteria Check")
    logger.info("=" * 60)
    logger.info(f"Target: RMSE < 11.5, Win Accuracy > 66%")
    logger.info(f"Best RMSE: {best_rmse:.4f} {'✓' if best_rmse < 11.5 else '✗'}")
    logger.info(f"Best Win Acc: {best_win_acc:.2%} {'✓' if best_win_acc > 0.66 else '✗'}")
    logger.info(f"Ensemble RMSE: {ensemble_metrics.rmse:.4f}")
    logger.info(f"Ensemble Win Acc: {ensemble_metrics.win_accuracy:.2%}")

    # 7. 피처 중요도
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance (XGBoost)")
    logger.info("=" * 60)

    importance = xgb_model.get_top_features(15)
    for feat, imp in importance:
        logger.info(f"  {feat}: {imp:.4f}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
