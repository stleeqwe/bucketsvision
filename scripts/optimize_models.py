#!/usr/bin/env python3
"""
Optuna Hyperparameter Optimization Script.

모든 모델의 하이퍼파라미터를 최적화하고 앙상블을 구성합니다.
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
from src.models.base import EnsembleModel
from src.evaluation.metrics import calculate_metrics, MetricsReport
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


def optimize_ensemble_weights(models: Dict, X_val: pd.DataFrame, y_val: pd.Series, n_trials: int = 100):
    """앙상블 가중치 최적화"""
    predictions = {name: model.predict(X_val) for name, model in models.items()}
    y_true = y_val.values

    def objective(trial):
        weights = {}
        for name in models.keys():
            weights[name] = trial.suggest_float(f"w_{name}", 0.0, 1.0)

        total = sum(weights.values())
        if total == 0:
            return float("inf")

        weights = {k: v / total for k, v in weights.items()}

        ensemble_pred = np.zeros(len(y_true))
        for name, w in weights.items():
            ensemble_pred += w * predictions[name]

        rmse = np.sqrt(np.mean((y_true - ensemble_pred) ** 2))
        return rmse

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
    parser = argparse.ArgumentParser(description="Optuna Hyperparameter Optimization")
    parser.add_argument("--train-seasons", type=int, nargs="+", default=[2023, 2024])
    parser.add_argument("--val-season", type=int, default=2025)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    data_dir = settings.data_dir
    output_dir = Path(args.output_dir) if args.output_dir else data_dir / "models"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Optuna Hyperparameter Optimization")
    logger.info("=" * 60)
    logger.info(f"Training seasons: {args.train_seasons}")
    logger.info(f"Validation season: {args.val_season}")
    logger.info(f"N trials: {args.n_trials}")

    # 1. 데이터 로드
    all_seasons = args.train_seasons + [args.val_season]
    df = load_and_merge_data(data_dir, all_seasons)

    if df.empty:
        logger.error("No data loaded")
        return

    train_df = df[df["season"].isin(args.train_seasons)]
    val_df = df[df["season"] == args.val_season]

    X_train, y_train, feature_names = prepare_data(train_df)
    X_val, y_val, _ = prepare_data(val_df)

    logger.info(f"Train: {len(X_train)} samples")
    logger.info(f"Val: {len(X_val)} samples")

    # Optuna 로깅 설정
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    best_params = {}
    best_models = {}

    # 2. XGBoost 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing XGBoost...")
    logger.info("=" * 60)

    xgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    xgb_study.optimize(
        create_xgb_objective(X_train, y_train, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["xgboost"] = xgb_study.best_params
    logger.info(f"XGBoost best CV RMSE: {xgb_study.best_value:.4f}")
    logger.info(f"Best params: {xgb_study.best_params}")

    # 최적 파라미터로 모델 재학습
    xgb_params = {**xgb_study.best_params, "objective": "reg:pseudohubererror", "random_state": 42, "n_jobs": -1, "verbosity": 0}
    xgb_model = XGBoostModel(params=xgb_params, use_huber=True)
    xgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    best_models["xgboost"] = xgb_model

    # 3. LightGBM 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing LightGBM...")
    logger.info("=" * 60)

    lgb_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    lgb_study.optimize(
        create_lgb_objective(X_train, y_train, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["lightgbm"] = lgb_study.best_params
    logger.info(f"LightGBM best CV RMSE: {lgb_study.best_value:.4f}")
    logger.info(f"Best params: {lgb_study.best_params}")

    lgb_params = {**lgb_study.best_params, "objective": "huber", "random_state": 42, "n_jobs": -1, "verbose": -1}
    lgb_model = LightGBMModel(params=lgb_params, use_huber=True)
    lgb_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    best_models["lightgbm"] = lgb_model

    # 4. Ridge 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing Ridge...")
    logger.info("=" * 60)

    ridge_study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    ridge_study.optimize(
        create_ridge_objective(X_train, y_train, args.cv_folds),
        n_trials=args.n_trials,
        show_progress_bar=True
    )

    best_params["ridge"] = ridge_study.best_params
    logger.info(f"Ridge best CV RMSE: {ridge_study.best_value:.4f}")
    logger.info(f"Best params: {ridge_study.best_params}")

    ridge_model = RidgeModel(alpha=ridge_study.best_params["alpha"], normalize_features=True)
    ridge_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    best_models["ridge"] = ridge_model

    # 5. 개별 모델 평가
    logger.info("\n" + "=" * 60)
    logger.info("Individual Model Evaluation")
    logger.info("=" * 60)

    results = []
    for name, model in best_models.items():
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

    # 6. 앙상블 가중치 최적화
    logger.info("\n" + "=" * 60)
    logger.info("Optimizing Ensemble Weights...")
    logger.info("=" * 60)

    ensemble_weights, ensemble_rmse = optimize_ensemble_weights(best_models, X_val, y_val, n_trials=100)
    logger.info(f"Ensemble weights: {ensemble_weights}")
    logger.info(f"Ensemble optimized RMSE: {ensemble_rmse:.4f}")

    # 앙상블 예측 및 평가
    ensemble_pred = np.zeros(len(y_val))
    for name, model in best_models.items():
        ensemble_pred += ensemble_weights[name] * model.predict(X_val)

    ensemble_metrics = calculate_metrics(y_val.values, ensemble_pred, "ensemble")
    logger.info(f"Ensemble: {ensemble_metrics.summary()}")

    results.append({
        "model": "ensemble",
        "rmse": ensemble_metrics.rmse,
        "mae": ensemble_metrics.mae,
        "win_accuracy": ensemble_metrics.win_accuracy,
        "within_5": ensemble_metrics.within_5_accuracy,
        "within_10": ensemble_metrics.within_10_accuracy
    })

    # 7. 최종 결과
    logger.info("\n" + "=" * 60)
    logger.info("Final Results")
    logger.info("=" * 60)

    results_df = pd.DataFrame(results).sort_values("rmse")
    print("\n" + results_df.to_string(index=False))

    best_model = results_df.iloc[0]
    logger.info("\n" + "=" * 60)
    logger.info("Success Criteria Check")
    logger.info("=" * 60)
    logger.info(f"Target: RMSE < 11.5, Win Accuracy > 66%")
    logger.info(f"Best Model: {best_model['model']}")
    logger.info(f"  RMSE: {best_model['rmse']:.4f} {'✓' if best_model['rmse'] < 11.5 else '✗'}")
    logger.info(f"  Win Accuracy: {best_model['win_accuracy']:.2%} {'✓' if best_model['win_accuracy'] > 0.66 else '✗'}")

    # 8. 결과 저장
    logger.info("\n" + "=" * 60)
    logger.info("Saving Results...")
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
    for name, model in best_models.items():
        model_path = output_dir / name
        model.save(model_path)
        logger.info(f"Saved {name} to {model_path}")

    # 결과 저장
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")

    logger.info("\nOptimization complete!")


if __name__ == "__main__":
    main()
