#!/usr/bin/env python3
"""
Phase 1: 통합 데이터셋 준비.

V1 (Ridge), V2 (XGBoost), V3 (Stacking) 모델을 동일한 테스트셋에서 평가하기 위해
통합된 예측 데이터셋을 생성합니다.

실행: python scripts/benchmark/prepare_unified_dataset.py
"""

import sys
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "bucketsvision_v3" / "src"))

import numpy as np
import pandas as pd


def load_test_data():
    """V2/V3가 사용한 테스트 데이터셋 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    # 검증용은 train + val 박스스코어 필요 (과거 데이터 참조)
    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    print(f"Test games: {len(val_games)}")
    print(f"Date range: {val_games['game_date'].min()} ~ {val_games['game_date'].max()}")

    return val_games, all_boxscores, team_stats


def build_v1_features(games_df, team_stats_df):
    """V1 Ridge 모델용 16개 EPM 피처 생성"""
    features_list = []

    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_epm = team_stats_df[
            (team_stats_df['team_id'] == home_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        away_epm = team_stats_df[
            (team_stats_df['team_id'] == away_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        if len(home_epm) == 0 or len(away_epm) == 0:
            continue

        home_latest = home_epm.iloc[0]
        away_latest = away_epm.iloc[0]

        feat = {
            'game_id': game['game_id'],
            'team_epm_diff': home_latest.get('team_epm', 0) - away_latest.get('team_epm', 0),
            'team_oepm_diff': home_latest.get('team_oepm', 0) - away_latest.get('team_oepm', 0),
            'team_depm_diff': home_latest.get('team_depm', 0) - away_latest.get('team_depm', 0),
            'team_epm_go_diff': home_latest.get('team_epm_game_optimized', 0) - away_latest.get('team_epm_game_optimized', 0),
            'team_oepm_go_diff': home_latest.get('team_oepm_game_optimized', 0) - away_latest.get('team_oepm_game_optimized', 0),
            'team_depm_go_diff': home_latest.get('team_depm_game_optimized', 0) - away_latest.get('team_depm_game_optimized', 0),
            'sos_diff': home_latest.get('sos', 0) - away_latest.get('sos', 0),
            'sos_o_diff': home_latest.get('sos_o', 0) - away_latest.get('sos_o', 0),
            'sos_d_diff': home_latest.get('sos_d', 0) - away_latest.get('sos_d', 0),
            'team_epm_rk_diff': home_latest.get('team_epm_rk', 15) - away_latest.get('team_epm_rk', 15),
            'team_oepm_rk_diff': home_latest.get('team_oepm_rk', 15) - away_latest.get('team_oepm_rk', 15),
            'team_depm_rk_diff': home_latest.get('team_depm_rk', 15) - away_latest.get('team_depm_rk', 15),
            'team_epm_z_diff': home_latest.get('team_epm_z', 0) - away_latest.get('team_epm_z', 0),
            'team_oepm_z_diff': home_latest.get('team_oepm_z', 0) - away_latest.get('team_oepm_z', 0),
            'team_depm_z_diff': home_latest.get('team_depm_z', 0) - away_latest.get('team_depm_z', 0),
            'home_advantage': 3.0,
        }
        features_list.append(feat)

    return pd.DataFrame(features_list)


def build_v2_features(games_df, boxscores_df, team_stats_df):
    """V2 XGBoost 모델용 피처 생성"""
    features_list = []

    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_box = boxscores_df[
            (boxscores_df['team_id'] == home_id) &
            (boxscores_df['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)

        away_box = boxscores_df[
            (boxscores_df['team_id'] == away_id) &
            (boxscores_df['game_date'] < game_date)
        ].sort_values('game_date', ascending=False)

        home_epm = team_stats_df[
            (team_stats_df['team_id'] == home_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        away_epm = team_stats_df[
            (team_stats_df['team_id'] == away_id) &
            (team_stats_df['game_dt'] < game_date)
        ].sort_values('game_dt', ascending=False)

        feat = {'game_id': game['game_id']}

        # Team EPM
        feat['team_epm_diff'] = (
            (home_epm['team_epm'].iloc[0] if len(home_epm) > 0 else 0) -
            (away_epm['team_epm'].iloc[0] if len(away_epm) > 0 else 0)
        )
        feat['sos_diff'] = (
            (home_epm['sos'].iloc[0] if len(home_epm) > 0 else 0) -
            (away_epm['sos'].iloc[0] if len(away_epm) > 0 else 0)
        )

        # Win pct
        home_10 = home_box.head(10)
        away_10 = away_box.head(10)
        feat['win_pct_10G_diff'] = (
            ((home_10['result'] == 'W').mean() if len(home_10) > 0 else 0.5) -
            ((away_10['result'] == 'W').mean() if len(away_10) > 0 else 0.5)
        )

        # Box score
        h_box = home_box.head(10)
        a_box = away_box.head(10)

        h_fg_pct = h_box['fg'].sum() / max(h_box['fga'].sum(), 1) if len(h_box) > 0 else 0.46
        a_fg_pct = a_box['fg'].sum() / max(a_box['fga'].sum(), 1) if len(a_box) > 0 else 0.46
        feat['fg_pct_diff'] = h_fg_pct - a_fg_pct

        h_efg = (h_box['fg'].sum() + 0.5 * h_box['fg3'].sum()) / max(h_box['fga'].sum(), 1) if len(h_box) > 0 else 0.50
        a_efg = (a_box['fg'].sum() + 0.5 * a_box['fg3'].sum()) / max(a_box['fga'].sum(), 1) if len(a_box) > 0 else 0.50
        feat['efg_pct_diff'] = h_efg - a_efg

        feat['orb_diff'] = (h_box['orb'].mean() if len(h_box) > 0 else 10) - (a_box['orb'].mean() if len(a_box) > 0 else 10)
        feat['drb_diff'] = (h_box['drb'].mean() if len(h_box) > 0 else 34) - (a_box['drb'].mean() if len(a_box) > 0 else 34)
        feat['ast_diff'] = (h_box['ast'].mean() if len(h_box) > 0 else 25) - (a_box['ast'].mean() if len(a_box) > 0 else 25)
        feat['tov_diff'] = (h_box['tov'].mean() if len(h_box) > 0 else 14) - (a_box['tov'].mean() if len(a_box) > 0 else 14)
        feat['margin_diff'] = (h_box['margin'].mean() if len(h_box) > 0 else 0) - (a_box['margin'].mean() if len(a_box) > 0 else 0)
        feat['pts_diff'] = (h_box['pts'].mean() if len(h_box) > 0 else 110) - (a_box['pts'].mean() if len(a_box) > 0 else 110)

        # EWMA
        if len(home_box) >= 3:
            feat['home_margin_ewma'] = home_box.head(10)['margin'].ewm(span=3, adjust=False).mean().iloc[0]
        else:
            feat['home_margin_ewma'] = 0
        if len(away_box) >= 3:
            feat['away_margin_ewma'] = away_box.head(10)['margin'].ewm(span=3, adjust=False).mean().iloc[0]
        else:
            feat['away_margin_ewma'] = 0
        feat['margin_ewma_diff'] = feat['home_margin_ewma'] - feat['away_margin_ewma']

        # Rest
        if len(home_box) > 0:
            home_rest = (pd.to_datetime(game_date) - pd.to_datetime(home_box.iloc[0]['game_date'])).days - 1
        else:
            home_rest = 3
        if len(away_box) > 0:
            away_rest = (pd.to_datetime(game_date) - pd.to_datetime(away_box.iloc[0]['game_date'])).days - 1
        else:
            away_rest = 3
        feat['rest_diff'] = min(home_rest, 4) - min(away_rest, 4)

        features_list.append(feat)

    return pd.DataFrame(features_list)


def build_v3_features(games_df, boxscores_df, team_stats_df):
    """V3 Stacking 모델용 26개 피처 생성"""
    from features.feature_builder import V3FeatureBuilder

    builder = V3FeatureBuilder()
    X_df = builder.build_features(games_df, boxscores_df, team_stats_df, verbose=False)
    return X_df


def margin_to_probability(margin, k=12.0):
    """점수차를 승률 확률로 변환 (Sigmoid)"""
    return 1 / (1 + np.exp(-margin / k))


def load_v1_model():
    """V1 Ridge 모델 로드"""
    model_dir = PROJECT_ROOT / "data" / "models" / "final" / "ridge"

    with open(model_dir / "model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(model_dir.parent / "feature_names.json", "r") as f:
        feature_names = json.load(f)

    return model, feature_names


def load_v2_model():
    """V2 XGBoost 모델 로드"""
    model_dir = PROJECT_ROOT / "bucketsvision_v2" / "models"

    with open(model_dir / "v2_optuna_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open(model_dir / "v2_feature_names.json", "r") as f:
        feature_names = json.load(f)

    return model, feature_names


def load_v3_model():
    """V3 Stacking 모델 로드"""
    model_dir = PROJECT_ROOT / "bucketsvision_v3" / "models"

    with open(model_dir / "v3_ensemble.pkl", "rb") as f:
        model = pickle.load(f)

    with open(model_dir / "v3_feature_names.json", "r") as f:
        feature_names = json.load(f)

    return model, feature_names


def main():
    print("=" * 70)
    print("  Phase 1: Unified Dataset Preparation")
    print("=" * 70)

    # 1. 테스트 데이터 로드
    print("\n[1/5] Loading Test Data")
    val_games, all_boxscores, team_stats = load_test_data()

    # 실제 결과
    y_true = val_games['home_win'].values
    game_ids = val_games['game_id'].values
    game_dates = val_games['game_date'].values

    # 2. V1 Ridge 예측
    print("\n[2/5] V1 Ridge Predictions")
    v1_model, v1_features = load_v1_model()
    X_v1 = build_v1_features(val_games, team_stats)

    # 피처 정렬
    X_v1_aligned = X_v1[['game_id'] + v1_features].copy()

    # 예측
    X_v1_input = X_v1_aligned[v1_features].fillna(0)
    v1_margin = v1_model.predict(X_v1_input)
    v1_prob = margin_to_probability(v1_margin, k=12.0)
    v1_pred = (v1_margin > 0).astype(int)

    print(f"  Samples: {len(v1_prob)}")
    print(f"  Margin range: [{v1_margin.min():.2f}, {v1_margin.max():.2f}]")
    print(f"  Prob range: [{v1_prob.min():.3f}, {v1_prob.max():.3f}]")

    # 3. V2 XGBoost 예측
    print("\n[3/5] V2 XGBoost Predictions")
    v2_model, v2_features = load_v2_model()
    X_v2 = build_v2_features(val_games, all_boxscores, team_stats)

    # 피처 정렬
    X_v2_aligned = X_v2[v2_features].fillna(0)

    # 예측
    v2_prob = v2_model.predict_proba(X_v2_aligned)[:, 1]
    v2_pred = v2_model.predict(X_v2_aligned)

    print(f"  Samples: {len(v2_prob)}")
    print(f"  Prob range: [{v2_prob.min():.3f}, {v2_prob.max():.3f}]")

    # 4. V3 Stacking 예측
    print("\n[4/5] V3 Stacking Predictions")
    v3_model, v3_features = load_v3_model()
    X_v3 = build_v3_features(val_games, all_boxscores, team_stats)

    # 피처 정렬
    feature_cols = [c for c in X_v3.columns if c != 'game_id']
    X_v3_aligned = X_v3[feature_cols].fillna(0).values

    # 예측
    v3_prob = v3_model.predict_proba(X_v3_aligned)
    v3_pred = (v3_prob >= 0.5).astype(int)

    print(f"  Samples: {len(v3_prob)}")
    print(f"  Prob range: [{v3_prob.min():.3f}, {v3_prob.max():.3f}]")

    # 5. 통합 데이터셋 생성
    print("\n[5/5] Creating Unified Dataset")

    # 게임 ID 기준으로 정렬
    v1_game_ids = X_v1['game_id'].values

    unified_df = pd.DataFrame({
        'game_id': game_ids,
        'game_date': game_dates,
        'y_true': y_true,
        'v1_margin': v1_margin,
        'v1_prob': v1_prob,
        'v1_pred': v1_pred,
        'v2_prob': v2_prob,
        'v2_pred': v2_pred,
        'v3_prob': v3_prob,
        'v3_pred': v3_pred,
    })

    # 기본 정확도 계산
    print("\n" + "=" * 70)
    print("  Initial Accuracy Check")
    print("=" * 70)

    v1_acc = (unified_df['v1_pred'] == unified_df['y_true']).mean()
    v2_acc = (unified_df['v2_pred'] == unified_df['y_true']).mean()
    v3_acc = (unified_df['v3_pred'] == unified_df['y_true']).mean()

    print(f"\n  {'Model':<20} {'Accuracy':<15} {'Samples':<10}")
    print("  " + "-" * 45)
    print(f"  {'V1 (Ridge)':<20} {v1_acc:.4f}          {len(unified_df)}")
    print(f"  {'V2 (XGBoost)':<20} {v2_acc:.4f}          {len(unified_df)}")
    print(f"  {'V3 (Stacking)':<20} {v3_acc:.4f}          {len(unified_df)}")

    # 저장
    output_dir = PROJECT_ROOT / "data" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    unified_df.to_parquet(output_dir / "unified_predictions.parquet", index=False)
    print(f"\n  Saved to: {output_dir / 'unified_predictions.parquet'}")

    # 메타데이터 저장
    metadata = {
        'n_games': len(unified_df),
        'date_range': {
            'min': str(unified_df['game_date'].min()),
            'max': str(unified_df['game_date'].max()),
        },
        'initial_accuracy': {
            'v1_ridge': float(v1_acc),
            'v2_xgboost': float(v2_acc),
            'v3_stacking': float(v3_acc),
        },
        'features': {
            'v1': v1_features,
            'v2': v2_features,
            'v3': feature_cols,
        }
    }

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {output_dir / 'metadata.json'}")

    print("\n" + "=" * 70)
    print("  Phase 1 Complete!")
    print("=" * 70)

    return unified_df


if __name__ == "__main__":
    main()
