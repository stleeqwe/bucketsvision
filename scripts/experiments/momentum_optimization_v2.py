#!/usr/bin/env python3
"""
모멘텀 피처 최적화 실험 V2.

상대팀 ID를 matchup에서 추출하여 상대강도 가중 피처 테스트

실행: python scripts/experiments/momentum_optimization_v2.py
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


def load_data():
    """데이터 로드"""
    v2_data_dir = PROJECT_ROOT / "bucketsvision_v2" / "data"

    train_games = pd.read_parquet(v2_data_dir / "train_games.parquet")
    train_boxscores = pd.read_parquet(v2_data_dir / "train_boxscores.parquet")
    val_games = pd.read_parquet(v2_data_dir / "val_games.parquet")
    val_boxscores = pd.read_parquet(v2_data_dir / "val_boxscores.parquet")
    team_stats = pd.read_parquet(v2_data_dir / "team_stats.parquet")

    all_boxscores = pd.concat([train_boxscores, val_boxscores], ignore_index=True)

    # 팀 약어 → ID 매핑 생성
    team_mapping = all_boxscores[['team_abbr', 'team_id']].drop_duplicates()
    team_abbr_to_id = dict(zip(team_mapping['team_abbr'], team_mapping['team_id']))

    # 상대팀 ID 추가
    def extract_opp_team_id(row):
        matchup = row['matchup']
        team_abbr = row['team_abbr']

        # "GSW @ POR" 또는 "GSW vs. LAL" 형식
        if ' @ ' in matchup:
            parts = matchup.split(' @ ')
        elif ' vs. ' in matchup:
            parts = matchup.split(' vs. ')
        else:
            return None

        opp_abbr = parts[1] if parts[0] == team_abbr else parts[0]
        return team_abbr_to_id.get(opp_abbr)

    all_boxscores['opp_team_id'] = all_boxscores.apply(extract_opp_team_id, axis=1)
    train_boxscores['opp_team_id'] = train_boxscores.apply(extract_opp_team_id, axis=1)

    print(f"opp_team_id 추출 완료: {all_boxscores['opp_team_id'].notna().sum()}/{len(all_boxscores)}")

    return train_games, val_games, train_boxscores, all_boxscores, team_stats


def get_team_history(boxscores_df, team_id, game_date):
    """팀의 과거 경기 히스토리"""
    return boxscores_df[
        (boxscores_df['team_id'] == team_id) &
        (boxscores_df['game_date'] < game_date)
    ].sort_values('game_date', ascending=False)


def get_team_epm(team_stats_df, team_id, game_date):
    """팀의 EPM 데이터"""
    df = team_stats_df[
        (team_stats_df['team_id'] == team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)
    return df


def get_opponent_epm(team_stats_df, opp_team_id, game_date):
    """상대팀의 EPM"""
    if pd.isna(opp_team_id):
        return 0.0

    opp_epm = team_stats_df[
        (team_stats_df['team_id'] == opp_team_id) &
        (team_stats_df['game_dt'] < game_date)
    ].sort_values('game_dt', ascending=False)

    if len(opp_epm) == 0:
        return 0.0
    return opp_epm.iloc[0].get('team_epm', 0.0)


def calc_strength_adjusted_win_pct(box_df, team_stats_df, game_date):
    """
    상대강도 가중 승률.

    강팀(EPM > 0)을 이기면 가중치 높게
    약팀(EPM < 0)에게 지면 가중치 높게 (페널티)
    """
    if len(box_df) == 0:
        return 0.5, 0.0  # adjusted_win_pct, avg_opp_strength

    weighted_wins = 0
    total_weight = 0
    opp_strengths = []

    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id')
        if pd.isna(opp_id):
            continue

        opp_epm = get_opponent_epm(team_stats_df, opp_id, game_date)
        opp_strengths.append(opp_epm)

        # 가중치: 기본 1 + EPM/10 (범위: 0.5 ~ 1.5)
        # 강팀일수록 높은 가중치
        weight = 1.0 + (opp_epm / 10.0)
        weight = max(0.5, min(1.5, weight))

        win = 1 if row['result'] == 'W' else 0
        weighted_wins += win * weight
        total_weight += weight

    if total_weight == 0:
        return 0.5, 0.0

    adj_win_pct = weighted_wins / total_weight
    avg_opp_strength = np.mean(opp_strengths) if opp_strengths else 0.0

    return adj_win_pct, avg_opp_strength


def calc_strength_adjusted_margin(box_df, team_stats_df, game_date):
    """
    상대강도 가중 마진.

    강팀 상대로 +10점 승리 → 고평가
    약팀 상대로 +10점 승리 → 보통 평가
    """
    if len(box_df) == 0:
        return 0.0

    weighted_margin = 0
    total_weight = 0

    for _, row in box_df.iterrows():
        opp_id = row.get('opp_team_id')
        if pd.isna(opp_id):
            continue

        opp_epm = get_opponent_epm(team_stats_df, opp_id, game_date)

        weight = 1.0 + (opp_epm / 10.0)
        weight = max(0.5, min(1.5, weight))

        margin = row['margin']
        weighted_margin += margin * weight
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return weighted_margin / total_weight


def build_features(games_df, boxscores_df, team_stats_df, window=5):
    """전체 피처 빌드"""
    features_list = []
    total = len(games_df)

    for idx, (_, game) in enumerate(games_df.iterrows()):
        if (idx + 1) % 500 == 0:
            print(f"    Progress: {idx + 1}/{total}")

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

        # 1. EPM 기본 피처
        feat['team_epm_diff'] = home_epm.iloc[0].get('team_epm', 0) - away_epm.iloc[0].get('team_epm', 0)
        feat['team_oepm_diff'] = home_epm.iloc[0].get('team_oepm', 0) - away_epm.iloc[0].get('team_oepm', 0)
        feat['team_depm_diff'] = home_epm.iloc[0].get('team_depm', 0) - away_epm.iloc[0].get('team_depm', 0)

        # 2. 윈도우 기반 피처
        h_window = home_box.head(window)
        a_window = away_box.head(window)

        # Plain 승률
        h_win_pct = (h_window['result'] == 'W').mean() if len(h_window) > 0 else 0.5
        a_win_pct = (a_window['result'] == 'W').mean() if len(a_window) > 0 else 0.5
        feat['win_pct_diff_plain'] = h_win_pct - a_win_pct

        # Plain 마진
        h_margin = h_window['margin'].mean() if len(h_window) > 0 else 0
        a_margin = a_window['margin'].mean() if len(a_window) > 0 else 0
        feat['margin_diff_plain'] = h_margin - a_margin

        # 상대강도 가중 승률
        h_adj_win, h_opp_str = calc_strength_adjusted_win_pct(h_window, team_stats_df, game_date)
        a_adj_win, a_opp_str = calc_strength_adjusted_win_pct(a_window, team_stats_df, game_date)
        feat['win_pct_diff_adjusted'] = h_adj_win - a_adj_win

        # 상대강도 가중 마진
        h_adj_margin = calc_strength_adjusted_margin(h_window, team_stats_df, game_date)
        a_adj_margin = calc_strength_adjusted_margin(a_window, team_stats_df, game_date)
        feat['margin_diff_adjusted'] = h_adj_margin - a_adj_margin

        # 최근 N경기 평균 상대 강도 차이
        feat['recent_opp_strength_diff'] = h_opp_str - a_opp_str

        # 상대강도 대비 성과 (기대 vs 실제)
        # 강팀과 많이 붙었는데 승률이 높다 → 좋은 팀
        feat['performance_vs_schedule_home'] = h_win_pct - (0.5 - h_opp_str / 20)  # 상대강도 보정된 기대 승률 대비
        feat['performance_vs_schedule_away'] = a_win_pct - (0.5 - a_opp_str / 20)
        feat['performance_vs_schedule_diff'] = feat['performance_vs_schedule_home'] - feat['performance_vs_schedule_away']

        features_list.append(feat)

    return pd.DataFrame(features_list)


def test_feature_set(X_train_df, X_val_df, y_train, y_val, features, name):
    """특정 피처 세트로 모델 학습 및 평가"""
    X_train = X_train_df[features].fillna(0).values
    X_val = X_val_df[features].fillna(0).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    val_pred = model.predict(X_val_scaled)
    val_proba = model.predict_proba(X_val_scaled)[:, 1]

    acc = accuracy_score(y_val, val_pred)
    brier = brier_score_loss(y_val, val_proba)

    coef_dict = dict(zip(features, model.coef_[0]))

    return {
        'name': name,
        'accuracy': acc,
        'brier_score': brier,
        'coefficients': coef_dict
    }


def main():
    print("=" * 70)
    print("  모멘텀 피처 최적화 실험 V2")
    print("  (상대팀 ID 추출 버전)")
    print("=" * 70)

    # 데이터 로드
    print("\n[1/3] 데이터 로드")
    train_games, val_games, train_boxscores, all_boxscores, team_stats = load_data()

    # 피처 빌드
    print("\n[2/3] 피처 빌드")
    print("  Training features...")
    X_train_df = build_features(train_games, train_boxscores, team_stats, window=5)
    print("  Validation features...")
    X_val_df = build_features(val_games, all_boxscores, team_stats, window=5)

    # 레이블 매칭
    train_labels = train_games.set_index('game_id')['home_win']
    val_labels = val_games.set_index('game_id')['home_win']

    X_train_df = X_train_df.set_index('game_id')
    X_val_df = X_val_df.set_index('game_id')

    common_train = X_train_df.index.intersection(train_labels.index)
    common_val = X_val_df.index.intersection(val_labels.index)

    X_train_df = X_train_df.loc[common_train]
    X_val_df = X_val_df.loc[common_val]
    y_train = train_labels.loc[common_train].values
    y_val = val_labels.loc[common_val].values

    print(f"\n  Train samples: {len(y_train)}, Val samples: {len(y_val)}")

    # 피처 조합 테스트
    print("\n[3/3] 피처 조합 테스트")
    print("=" * 70)

    feature_sets = {
        '1. EPM Only': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff'
        ],
        '2. EPM + Plain (last5)': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'win_pct_diff_plain', 'margin_diff_plain'
        ],
        '3. EPM + Adjusted (last5)': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'win_pct_diff_adjusted', 'margin_diff_adjusted'
        ],
        '4. EPM + Plain + Opp Strength': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'win_pct_diff_plain', 'margin_diff_plain',
            'recent_opp_strength_diff'
        ],
        '5. EPM + Adjusted + Opp Strength': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'win_pct_diff_adjusted', 'margin_diff_adjusted',
            'recent_opp_strength_diff'
        ],
        '6. EPM + Performance vs Schedule': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'performance_vs_schedule_diff'
        ],
        '7. All Features': [
            'team_epm_diff', 'team_oepm_diff', 'team_depm_diff',
            'win_pct_diff_plain', 'margin_diff_plain',
            'win_pct_diff_adjusted', 'margin_diff_adjusted',
            'recent_opp_strength_diff', 'performance_vs_schedule_diff'
        ],
    }

    results = []

    for name, features in feature_sets.items():
        result = test_feature_set(X_train_df, X_val_df, y_train, y_val, features, name)
        results.append(result)
        print(f"\n  {name}")
        print(f"    Accuracy: {result['accuracy']:.4f}, Brier: {result['brier_score']:.4f}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("  결과 요약")
    print("=" * 70)
    print(f"\n  {'Feature Set':<40} {'Accuracy':<12} {'Brier':<12}")
    print("  " + "-" * 64)

    for r in results:
        marker = " ★" if r['accuracy'] == max(x['accuracy'] for x in results) else ""
        print(f"  {r['name']:<40} {r['accuracy']:<12.4f} {r['brier_score']:<12.4f}{marker}")

    # 최고 성능 피처 세트의 계수
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\n  Best: {best['name']} (Accuracy: {best['accuracy']:.4f})")

    print("\n  피처 계수 (Best Model):")
    for feat, coef in sorted(best['coefficients'].items(), key=lambda x: -abs(x[1])):
        direction = "+" if coef > 0 else "-"
        print(f"    {feat:<35} {direction}{abs(coef):.4f}")

    # Plain vs Adjusted 비교
    plain_result = results[1]  # EPM + Plain
    adj_result = results[2]    # EPM + Adjusted

    print("\n" + "=" * 70)
    print("  핵심 비교: Plain vs Strength-Adjusted")
    print("=" * 70)
    print(f"\n  Plain Momentum:    {plain_result['accuracy']:.4f}")
    print(f"  Adjusted Momentum: {adj_result['accuracy']:.4f}")
    print(f"  차이: {(adj_result['accuracy'] - plain_result['accuracy'])*100:+.2f}%p")

    return results


if __name__ == "__main__":
    main()
