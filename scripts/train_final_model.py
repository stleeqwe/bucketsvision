#!/usr/bin/env python3
"""
Final Model Training Script for 25-26 Season Prediction.

22-23, 23-24, 24-25 시즌 전체를 학습하여 25-26 시즌 예측용 모델을 생성합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from src.utils.logger import logger
from src.models.xgboost_model import XGBoostModel
from src.models.lightgbm_model import LightGBMModel
from src.models.ridge_model import RidgeModel
from src.evaluation.metrics import calculate_metrics
from src.evaluation.cross_validation import TimeSeriesCV
from config.settings import settings


def load_and_merge_data(data_dir: Path, seasons: List[int]) -> pd.DataFrame:
    """team_epm과 games 데이터를 로드하고 병합"""
    logger.info(f"Loading data for seasons: {seasons}")

    all_data = []

    for season in seasons:
        team_epm_path = data_dir / "raw" / "dnt" / "team_epm" / f"season_{season}.parquet"
        games_path = data_dir / "raw" / "nba_stats" / "games" / f"season_{season}.parquet"

        if not team_epm_path.exists() or not games_path.exists():
            logger.warning(f"Missing data for season {season}")
            continue

        team_epm = pd.read_parquet(team_epm_path)
        games = pd.read_parquet(games_path)

        logger.info(f"Season {season}: {len(team_epm)} team_epm, {len(games)} games")

        team_epm["game_dt"] = pd.to_datetime(team_epm["game_dt"]).dt.strftime("%Y-%m-%d")
        games["game_date"] = pd.to_datetime(games["game_date"]).dt.strftime("%Y-%m-%d")

        for _, game in games.iterrows():
            game_date = game["game_date"]
            home_id = game["home_team_id"]
            away_id = game["away_team_id"]

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

            home_latest = home_epm.iloc[0]
            away_latest = away_epm.iloc[0]

            features = {
                "game_id": game["game_id"],
                "game_date": game_date,
                "home_team_id": home_id,
                "away_team_id": away_id,
                "season": season,
                "margin": game["margin"],
                "team_epm_diff": home_latest["team_epm"] - away_latest["team_epm"],
                "team_oepm_diff": home_latest["team_oepm"] - away_latest["team_oepm"],
                "team_depm_diff": home_latest["team_depm"] - away_latest["team_depm"],
                "team_epm_go_diff": home_latest["team_epm_game_optimized"] - away_latest["team_epm_game_optimized"],
                "team_oepm_go_diff": home_latest["team_oepm_game_optimized"] - away_latest["team_oepm_game_optimized"],
                "team_depm_go_diff": home_latest["team_depm_game_optimized"] - away_latest["team_depm_game_optimized"],
                "sos_diff": home_latest["sos"] - away_latest["sos"],
                "sos_o_diff": home_latest["sos_o"] - away_latest["sos_o"],
                "sos_d_diff": home_latest["sos_d"] - away_latest["sos_d"],
                "team_epm_rk_diff": home_latest["team_epm_rk"] - away_latest["team_epm_rk"],
                "team_oepm_rk_diff": home_latest["team_oepm_rk"] - away_latest["team_oepm_rk"],
                "team_depm_rk_diff": home_latest["team_depm_rk"] - away_latest["team_depm_rk"],
                "team_epm_z_diff": home_latest["team_epm_z"] - away_latest["team_epm_z"],
                "team_oepm_z_diff": home_latest["team_oepm_z"] - away_latest["team_oepm_z"],
                "team_depm_z_diff": home_latest["team_depm_z"] - away_latest["team_depm_z"],
                "home_advantage": 3.0,
            }

            all_data.append(features)

    df = pd.DataFrame(all_data)
    df = df.sort_values("game_date").reset_index(drop=True)

    logger.info(f"Total samples: {len(df)}")
    return df


def prepare_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """학습 데이터 준비"""
    exclude_cols = ["game_id", "game_date", "home_team_id", "away_team_id", "season", "margin"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy().fillna(0)
    y = df["margin"].copy()

    return X, y, feature_cols


def create_xgb_objective(X_train, y_train, cv_folds=5):
    """XGBoost Optuna 목적 함수 생성"""
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "huber_slope": trial.suggest_float("huber_slope", 5.0, 20.0),
            "objective": "reg:pseudohubererror",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0
        }

        tscv = TimeSeriesCV(n_splits=cv_folds)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            model = XGBoostModel(params=params, use_huber=True)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

            y_pred = model.predict(X_va)
            rmse = np.sqrt(np.mean((y_va.values - y_pred) ** 2))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


def create_lgb_objective(X_train, y_train, cv_folds=5):
    """LightGBM Optuna 목적 함수 생성"""
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "alpha": trial.suggest_float("alpha", 5.0, 20.0),
            "objective": "huber",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

        tscv = TimeSeriesCV(n_splits=cv_folds)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            model = LightGBMModel(params=params, use_huber=True)
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)

            y_pred = model.predict(X_va)
            rmse = np.sqrt(np.mean((y_va.values - y_pred) ** 2))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


def create_ridge_objective(X_train, y_train, cv_folds=5):
    """Ridge Optuna 목적 함수 생성"""
    def objective(trial):
        alpha = trial.suggest_float("alpha", 0.01, 1000.0, log=True)

        tscv = TimeSeriesCV(n_splits=cv_folds)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X_train):
            X_tr = X_train.iloc[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_va = X_train.iloc[val_idx]
            y_va = y_train.iloc[val_idx]

            model = RidgeModel(alpha=alpha, normalize_features=True)
            model.fit(X_tr, y_tr)

            y_pred = model.predict(X_va)
            rmse = np.sqrt(np.mean((y_va.values - y_pred) ** 2))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    return objective


def optimize_ensemble_weights_cv(models: Dict, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5, n_trials: int = 100):
    """CV 기반 앙상블 가중치 최적화"""

    def objective(trial):
        weights = {}
        for name in models.keys():
            weights[name] = trial.suggest_float(f"w_{name}", 0.0, 1.0)

        total = sum(weights.values())
        if total == 0:
            return float("inf")

        weights = {k: v / total for k, v in weights.items()}

        tscv = TimeSeriesCV(n_splits=cv_folds)
        cv_scores = []

        for train_idx, val_idx in tscv.split(X):
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx]
            X_va = X.iloc[val_idx]
            y_va = y.iloc[val_idx]

            # 각 모델 학습 및 예측
            fold_models = {}
            for name, model_class_info in models.items():
                if name == "xgboost":
                    m = XGBoostModel(params=model_class_info["params"], use_huber=True)
                elif name == "lightgbm":
                    m = LightGBMModel(params=model_class_info["params"], use_huber=True)
                else:  # ridge
                    m = RidgeModel(alpha=model_class_info["params"]["alpha"], normalize_features=True)

                m.fit(X_tr, y_tr)
                fold_models[name] = m

            # 앙상블 예측
            ensemble_pred = np.zeros(len(y_va))
            for name, w in weights.items():
                ensemble_pred += w * fold_models[name].predict(X_va)

            rmse = np.sqrt(np.mean((y_va.values - ensemble_pred) ** 2))
            cv_scores.append(rmse)

        return np.mean(cv_scores)

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_weights = {}
    for name in models.keys():
        best_weights[name] = study.best_params[f"w_{name}"]

    total = sum(best_weights.values())
    best_weights = {k: v / total for k, v in best_weights.items()}

    return best_weights, study.best_value


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="Final Model Training for 25-26 Season")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025],
                        help="Seasons to train on (default: 2023 2024 2025)")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials per model")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Number of CV folds")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    data_dir = settings.data_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "models" / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Final Model Training for 25-26 Season Prediction")
    logger.info("=" * 60)
    logger.info(f"Training seasons: {args.seasons}")
    logger.info(f"N trials: {args.n_trials}")
    logger.info(f"CV folds: {args.cv_folds}")

    # 1. 데이터 로드
    df = load_and_merge_data(data_dir, args.seasons)

    if df.empty:
        logger.error("No data loaded")
        return

    X, y, feature_names = prepare_data(df)
    logger.info(f"Total training samples: {len(X)}")
    logger.info(f"Features: {feature_names}")

    # Optuna 로깅 설정
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_params = {}

    # 2. XGBoost 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing XGBoost...")
    logger.info("=" * 60)

    xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    xgb_study.optimize(
        create_xgb_objective(X, y, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["xgboost"] = xgb_study.best_params
    logger.info(f"XGBoost best CV RMSE: {xgb_study.best_value:.4f}")

    # 3. LightGBM 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing LightGBM...")
    logger.info("=" * 60)

    lgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    lgb_study.optimize(
        create_lgb_objective(X, y, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["lightgbm"] = lgb_study.best_params
    logger.info(f"LightGBM best CV RMSE: {lgb_study.best_value:.4f}")

    # 4. Ridge 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing Ridge...")
    logger.info("=" * 60)

    ridge_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    ridge_study.optimize(
        create_ridge_objective(X, y, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["ridge"] = ridge_study.best_params
    logger.info(f"Ridge best CV RMSE: {ridge_study.best_value:.4f}")

    # 5. 앙상블 가중치 최적화 (CV 기반)
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing Ensemble Weights (CV-based)...")
    logger.info("=" * 60)

    model_configs = {
        "xgboost": {"params": {**xgb_study.best_params, "objective": "reg:pseudohubererror", "random_state": 42, "n_jobs": -1, "verbosity": 0}},
        "lightgbm": {"params": {**lgb_study.best_params, "objective": "huber", "random_state": 42, "n_jobs": -1, "verbose": -1}},
        "ridge": {"params": {"alpha": ridge_study.best_params["alpha"]}}
    }

    ensemble_weights, ensemble_cv_rmse = optimize_ensemble_weights_cv(
        model_configs, X, y, cv_folds=args.cv_folds, n_trials=100
    )
    logger.info(f"Ensemble weights: {ensemble_weights}")
    logger.info(f"Ensemble CV RMSE: {ensemble_cv_rmse:.4f}")

    # 6. 최종 모델 학습 (전체 데이터)
    logger.info("\n" + "=" * 60)
    logger.info("Training Final Models on Full Data...")
    logger.info("=" * 60)

    final_models = {}

    # XGBoost
    xgb_params = {**xgb_study.best_params, "objective": "reg:pseudohubererror", "random_state": 42, "n_jobs": -1, "verbosity": 0}
    xgb_model = XGBoostModel(params=xgb_params, use_huber=True)
    xgb_model.fit(X, y, verbose=False)
    final_models["xgboost"] = xgb_model
    logger.info("XGBoost trained")

    # LightGBM
    lgb_params = {**lgb_study.best_params, "objective": "huber", "random_state": 42, "n_jobs": -1, "verbose": -1}
    lgb_model = LightGBMModel(params=lgb_params, use_huber=True)
    lgb_model.fit(X, y, verbose=False)
    final_models["lightgbm"] = lgb_model
    logger.info("LightGBM trained")

    # Ridge
    ridge_model = RidgeModel(alpha=ridge_study.best_params["alpha"], normalize_features=True)
    ridge_model.fit(X, y)
    final_models["ridge"] = ridge_model
    logger.info("Ridge trained")

    # 7. CV 기반 성능 평가 (Hold-out 없으므로 CV 결과로 대체)
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Performance Summary")
    logger.info("=" * 60)

    cv_results = [
        {"model": "xgboost", "cv_rmse": xgb_study.best_value},
        {"model": "lightgbm", "cv_rmse": lgb_study.best_value},
        {"model": "ridge", "cv_rmse": ridge_study.best_value},
        {"model": "ensemble", "cv_rmse": ensemble_cv_rmse}
    ]

    cv_df = pd.DataFrame(cv_results).sort_values("cv_rmse")
    print("\n" + cv_df.to_string(index=False))

    # 8. 피처 중요도
    logger.info("\n" + "=" * 60)
    logger.info("Feature Importance (XGBoost)")
    logger.info("=" * 60)

    importance = xgb_model.get_top_features(15)
    for feat, imp in importance:
        logger.info(f"  {feat}: {imp:.4f}")

    # 9. 결과 저장
    logger.info("\n" + "=" * 60)
    logger.info("Saving Final Models...")
    logger.info("=" * 60)

    # 최적 파라미터 저장
    params_path = output_dir / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Saved best params to {params_path}")

    # 앙상블 가중치 저장
    weights_path = output_dir / "ensemble_weights.json"
    with open(weights_path, "w") as f:
        json.dump(ensemble_weights, f, indent=2)
    logger.info(f"Saved ensemble weights to {weights_path}")

    # 모델 저장
    for name, model in final_models.items():
        model_path = output_dir / name
        model.save(model_path)
        logger.info(f"Saved {name} to {model_path}")

    # CV 결과 저장
    cv_path = output_dir / "cv_results.csv"
    cv_df.to_csv(cv_path, index=False)
    logger.info(f"Saved CV results to {cv_path}")

    # 피처 이름 저장
    features_path = output_dir / "feature_names.json"
    with open(features_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    logger.info(f"Saved feature names to {features_path}")

    # 메타데이터 저장
    metadata = {
        "training_date": datetime.now().isoformat(),
        "seasons": args.seasons,
        "n_samples": len(X),
        "n_features": len(feature_names),
        "cv_folds": args.cv_folds,
        "n_trials": args.n_trials,
        "best_cv_rmse": float(cv_df["cv_rmse"].min()),
        "ensemble_weights": ensemble_weights
    }
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Models saved to: {output_dir}")
    logger.info(f"Best CV RMSE: {cv_df['cv_rmse'].min():.4f}")
    logger.info("Ready for 25-26 season prediction!")


if __name__ == "__main__":
    main()
