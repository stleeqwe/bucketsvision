#!/usr/bin/env python3
"""
모멘텀 피처 최적화 실험.

실험 1: 최근 N경기 윈도우 최적화 (3, 5, 7, 10, 15)
실험 2: 상대강도 가중 모멘텀 피처 테스트

실행: python scripts/experiments/momentum_optimization.py
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss
from typing import Dict, List, Tuple


def load_data():
    """데이터 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    train_games = pd.read_parquet(v2_data_dir / "train_games.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    return train_games, val_games, train_boxscores, all_boxscores, team_stats


def get_team_history(boxscores_df: pd.DataFrame, team_id: int, game_date: str) -> pd.DataFrame:
    """팀의 과거 경기 히스토리"""
    return boxscores_df[
        (boxscores_df['team_id'] == team_id) &
        (boxscores_df['game_date'] < game_date)
    ].sort_values('game_date', ascending=False)


def get_team_epm(team_stats_df: pd.DataFrame, team_id: int, game_date: str) -> pd.DataFrame:
    """팀의 EPM 데이터"""
    return team_stats_df[
        (team_stats_df['team_id'] == team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)


def get_opponent_strength(team_stats_df: pd.DataFrame, opp_team_id: int, game_date: str) -> float:
    """상대팀의 강도 (EPM 기반)"""
    opp_epm = team_stats_df[
        (team_stats_df['team_id'] == opp_team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)

    if len(opp_epm) == 0:
        return 0.0
    return opp_epm.iloc[0].get('team_epm', 0.0)


# =============================================================================
# 실험 1: 윈도우 크기별 모멘텀 피처
# =============================================================================

def build_momentum_features_window(
    games_df: pd.DataFrame,
    boxscores_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    window: int
) -> pd.DataFrame:
    """특정 윈도우 크기로 모멘텀 피처 생성"""
    features_list = []

    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_box = get_team_history(boxscores_df, home_id, game_date)
        away_box = get_team_history(boxscores_df, away_id, game_date)
        home_epm = get_team_epm(team_stats_df, home_id, game_date)
        away_epm = get_team_epm(team_stats_df, away_id, game_date)

        if len(home_epm) == 0 or len(away_epm) == 0:
            continue

        feat = {'game_id': game['game_id']}

        # EPM 기본 피처 (고정)
        feat['team_epm_diff'] = home_epm.iloc[0].get('team_epm', 0) - away_epm.iloc[0].get('team_epm', 0)
        feat['team_oepm_diff'] = home_epm.iloc[0].get('team_oepm', 0) - away_epm.iloc[0].get('team_oepm', 0)
        feat['team_depm_diff'] = home_epm.iloc[0].get('team_depm', 0) - away_epm.iloc[0].get('team_depm', 0)

        # 윈도우 기반 모멘텀 피처
        h_window = home_box.head(window)
        a_window = away_box.head(window)

        # 승률
        h_win_pct = (h_window['result'] == 'W').mean() if len(h_window) > 0 else 0.5
        a_win_pct = (a_window['result'] == 'W').mean() if len(a_window) > 0 else 0.5
        feat[f'last{window}_win_pct_diff'] = h_win_pct - a_win_pct

        # 평균 마진
        h_margin = h_window['margin'].mean() if len(h_window) > 0 else 0
        a_margin = a_window['margin'].mean() if len(a_window) > 0 else 0
        feat[f'last{window}_margin_diff'] = h_margin - a_margin

        # EWMA 마진 (span = window // 2 or 3)
        ewma_span = max(3, window // 2)
        if len(h_window) >= 3:
            h_ewma = h_window['margin'].ewm(span=ewma_span, adjust=False).mean().iloc[0]
        else:
            h_ewma = 0
        if len(a_window) >= 3:
            a_ewma = a_window['margin'].ewm(span=ewma_span, adjust=False).mean().iloc[0]
        else:
            a_ewma = 0
        feat[f'last{window}_ewma_diff'] = h_ewma - a_ewma

        features_list.append(feat)

    return pd.DataFrame(features_list)


def experiment_window_sizes():
    """윈도우 크기별 성능 비교"""
    print("\n" + "=" * 70)
    print("  실험 1: 최근 N경기 윈도우 크기 최적화")
    print("=" * 70)

    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    windows = [3, 5, 7, 10, 15]
    results = []

    for window in windows:
        print(f"\n  Testing window = {window}...")

        # 피처 생성
        X_train_df = build_momentum_features_window(train_games, train_boxscores, team_stats, window)
        X_val_df = build_momentum_features_window(val_games, all_boxscores, team_stats, window)

        # 레이블 매칭
        train_labels = train_games.set_index('game_id')['home_win']
        val_labels = val_games.set_index('game_id')['home_win']

        X_train_df = X_train_df.set_index('game_id')
        X_val_df = X_val_df.set_index('game_id')

        common_train = X_train_df.index.intersection(train_labels.index)
        common_val = X_val_df.index.intersection(val_labels.index)

        feature_cols = [c for c in X_train_df.columns if c != 'game_id']

        X_train = X_train_df.loc[common_train, feature_cols].fillna(0).values
        y_train = train_labels.loc[common_train].values
        X_val = X_val_df.loc[common_val, feature_cols].fillna(0).values
        y_val = val_labels.loc[common_val].values

        # 학습 및 평가
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        val_pred = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]

        acc = accuracy_score(y_val, val_pred)
        brier = brier_score_loss(y_val, val_proba)

        results.append({
            'window': window,
            'accuracy': acc,
            'brier_score': brier,
            'n_samples': len(y_val)
        })

        print(f"    Accuracy: {acc:.4f}, Brier: {brier:.4f}")

    # 결과 정리
    print("\n" + "-" * 50)
    print(f"  {'Window':<10} {'Accuracy':<12} {'Brier Score':<12}")
    print("-" * 50)
    for r in results:
        print(f"  {r['window']:<10} {r['accuracy']:<12.4f} {r['brier_score']:<12.4f}")

    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  Best Window: {best['window']} (Accuracy: {best['accuracy']:.4f})")

    return results


# =============================================================================
# 실험 2: 상대강도 가중 모멘텀 피처
# =============================================================================

def build_strength_adjusted_momentum(
    games_df: pd.DataFrame,
    boxscores_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    window: int = 5
) -> pd.DataFrame:
    """상대강도 가중 모멘텀 피처 생성"""
    features_list = []

    for _, game in games_df.iterrows():
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        game_date = game['game_date']

        home_box = get_team_history(boxscores_df, home_id, game_date)
        away_box = get_team_history(boxscores_df, away_id, game_date)
        home_epm = get_team_epm(team_stats_df, home_id, game_date)
        away_epm = get_team_epm(team_stats_df, away_id, game_date)

        if len(home_epm) == 0 or len(away_epm) == 0:
            continue

        feat = {'game_id': game['game_id']}

        # EPM 기본 피처
        feat['team_epm_diff'] = home_epm.iloc[0].get('team_epm', 0) - away_epm.iloc[0].get('team_epm', 0)
        feat['team_oepm_diff'] = home_epm.iloc[0].get('team_oepm', 0) - away_epm.iloc[0].get('team_oepm', 0)
        feat['team_depm_diff'] = home_epm.iloc[0].get('team_depm', 0) - away_epm.iloc[0].get('team_depm', 0)

        # 일반 모멘텀 (비교용)
        h_window = home_box.head(window)
        a_window = away_box.head(window)

        h_win_pct = (h_window['result'] == 'W').mean() if len(h_window) > 0 else 0.5
        a_win_pct = (a_window['result'] == 'W').mean() if len(a_window) > 0 else 0.5
        feat['win_pct_diff_plain'] = h_win_pct - a_win_pct

        # 상대강도 가중 승률
        h_adj_wins = calc_strength_adjusted_record(h_window, team_stats_df, game_date)
        a_adj_wins = calc_strength_adjusted_record(a_window, team_stats_df, game_date)
        feat['win_pct_diff_adjusted'] = h_adj_wins - a_adj_wins

        # 상대강도 가중 마진
        h_adj_margin = calc_strength_adjusted_margin(h_window, team_stats_df, game_date)
        a_adj_margin = calc_strength_adjusted_margin(a_window, team_stats_df, game_date)
        feat['margin_diff_plain'] = (h_window['margin'].mean() if len(h_window) > 0 else 0) - \
                                    (a_window['margin'].mean() if len(a_window) > 0 else 0)
        feat['margin_diff_adjusted'] = h_adj_margin - a_adj_margin

        # 최근 N경기 평균 상대 강도
        h_opp_strength = calc_avg_opponent_strength(h_window, team_stats_df, game_date)
        a_opp_strength = calc_avg_opponent_strength(a_window, team_stats_df, game_date)
        feat['recent_opp_strength_diff'] = h_opp_strength - a_opp_strength

        features_list.append(feat)

    return pd.DataFrame(features_list)


def calc_strength_adjusted_record(box_df: pd.DataFrame, team_stats_df: pd.DataFrame, game_date: str) -> float:
    """
    상대강도 가중 승률 계산.

    강팀을 이기면 높은 점수, 약팀에게 지면 큰 감점
    공식: Σ(win * (1 + opp_epm/10)) / N
    """
    if len(box_df) == 0:
        return 0.5

    weighted_sum = 0
    weight_sum = 0

    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id', None)
        if opp_id is None:
            continue

        # 상대팀 강도 가져오기
        opp_strength = get_opponent_strength(team_stats_df, opp_id, game_date)

        # 가중치: 강팀일수록 높은 가중치 (1 + epm/10)
        # EPM 범위가 대략 -10 ~ +10이므로, 가중치는 0 ~ 2 범위
        weight = 1 + (opp_strength / 10)
        weight = max(0.5, min(2.0, weight))  # 0.5 ~ 2.0 클리핑

        win = 1 if row['result'] == 'W' else 0
        weighted_sum += win * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.5

    return weighted_sum / weight_sum


def calc_strength_adjusted_margin(box_df: pd.DataFrame, team_stats_df: pd.DataFrame, game_date: str) -> float:
    """
    상대강도 가중 마진 계산.

    강팀 상대로 큰 점수차 승리 → 높은 점수
    약팀 상대로 작은 점수차 승리 → 보통 점수
    """
    if len(box_df) == 0:
        return 0.0

    weighted_sum = 0
    weight_sum = 0

    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id', None)
        if opp_id is None:
            continue

        opp_strength = get_opponent_strength(team_stats_df, opp_id, game_date)

        # 가중치
        weight = 1 + (opp_strength / 10)
        weight = max(0.5, min(2.0, weight))

        margin = row['margin']
        weighted_sum += margin * weight
        weight_sum += weight

    if weight_sum == 0:
        return 0.0

    return weighted_sum / weight_sum


def calc_avg_opponent_strength(box_df: pd.DataFrame, team_stats_df: pd.DataFrame, game_date: str) -> float:
    """최근 N경기 평균 상대 강도"""
    if len(box_df) == 0:
        return 0.0

    strengths = []
    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id', None)
        if opp_id is not None:
            strength = get_opponent_strength(team_stats_df, opp_id, game_date)
            strengths.append(strength)

    return np.mean(strengths) if strengths else 0.0


def experiment_strength_adjusted():
    """상대강도 가중 피처 효과 검증"""
    print("\n" + "=" * 70)
    print("  실험 2: 상대강도 가중 모멘텀 피처")
    print("=" * 70)

    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    print("\n  Building features...")
    X_train_df = build_strength_adjusted_momentum(train_games, train_boxscores, team_stats, window=5)
    X_val_df = build_strength_adjusted_momentum(val_games, all_boxscores, team_stats, window=5)

    # 레이블 매칭
    train_labels = train_games.set_index('game_id')['home_win']
    val_labels = val_games.set_index('game_id')['home_win']

    X_train_df = X_train_df.set_index('game_id')
    X_val_df = X_val_df.set_index('game_id')

    common_train = X_train_df.index.intersection(train_labels.index)
    common_val = X_val_df.index.intersection(val_labels.index)

    y_train = train_labels.loc[common_train].values
    y_val = val_labels.loc[common_val].values

    # 피처 조합별 테스트
    feature_sets = {
        'EPM Only': ['team_epm_diff', 'team_oepm_diff', 'team_depm_diff'],
        'EPM + Plain Momentum': ['team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
                                  'win_pct_diff_plain', 'margin_diff_plain'],
        'EPM + Adjusted Momentum': ['team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
                                     'win_pct_diff_adjusted', 'margin_diff_adjusted'],
        'EPM + Both + Opp Strength': ['team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
                                       'win_pct_diff_plain', 'win_pct_diff_adjusted',
                                       'margin_diff_adjusted', 'recent_opp_strength_diff'],
    }

    results = []

    for name, features in feature_sets.items():
        print(f"\n  Testing: {name}")

        X_train = X_train_df.loc[common_train, features].fillna(0).values
        X_val = X_val_df.loc[common_val, features].fillna(0).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        val_pred = model.predict(X_val_scaled)
        val_proba = model.predict_proba(X_val_scaled)[:, 1]

        acc = accuracy_score(y_val, val_pred)
        brier = brier_score_loss(y_val, val_proba)

        # 피처 중요도
        coef_dict = dict(zip(features, np.abs(model.coef_[0])))

        results.append({
            'name': name,
            'features': features,
            'accuracy': acc,
            'brier_score': brier,
            'coefficients': coef_dict
        })

        print(f"    Accuracy: {acc:.4f}, Brier: {brier:.4f}")

    # 결과 정리
    print("\n" + "-" * 60)
    print(f"  {'Feature Set':<35} {'Accuracy':<12} {'Brier':<12}")
    print("-" * 60)
    for r in results:
        print(f"  {r['name']:<35} {r['accuracy']:<12.4f} {r['brier_score']:<12.4f}")

    # 상대강도 가중 효과 분석
    plain_acc = results[1]['accuracy']  # EPM + Plain
    adj_acc = results[2]['accuracy']    # EPM + Adjusted

    print(f"\n  상대강도 가중 효과: {(adj_acc - plain_acc)*100:+.2f}%p")

    # 피처 계수 비교
    if len(results) >= 4:
        print("\n  피처 계수 (EPM + Both + Opp Strength):")
        for feat, coef in sorted(results[3]['coefficients'].items(), key=lambda x: -x[1]):
            print(f"    {feat:<30} {coef:.4f}")

    return results


def main():
    print("=" * 70)
    print("  모멘텀 피처 최적화 실험")
    print("=" * 70)

    # 실험 1: 윈도우 크기
    window_results = experiment_window_sizes()

    # 실험 2: 상대강도 가중
    strength_results = experiment_strength_adjusted()

    print("\n" + "=" * 70)
    print("  실험 완료!")
    print("=" * 70)

    return window_results, strength_results


if __name__ == "__main__":
    main()
