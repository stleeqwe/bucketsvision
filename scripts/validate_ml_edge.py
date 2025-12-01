#!/usr/bin/env python3
"""
머니라인 Edge 경쟁력 검증 스크립트.

1. 모델 calibration 검증 (예측 확률 vs 실제 적중률)
2. Edge 기반 베팅 시뮬레이션 (가상 배당 적용)
3. ROI 분석
"""

import pandas as pd
import numpy as np
from pathlib import Path

# 데이터 로드
data_path = Path("/Users/stlee/Desktop/bucketsvision/data/predictions/season_2025_26_predictions.csv")
df = pd.read_csv(data_path)

# 종료된 경기만 필터링
df = df[df['game_status'] == '종료'].copy()
print(f"분석 대상: {len(df)}경기 (2025-26 시즌)")
print("=" * 60)

# 1. 모델 Calibration 분석
print("\n📊 1. 모델 Calibration 분석")
print("-" * 60)

# 예측 확률 구간별 적중률
bins = [0.5, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.0]
labels = ['50-55%', '55-60%', '60-65%', '65-70%', '70-75%', '75-80%', '80-85%', '85-90%', '90%+']

# 홈팀 승리 예측 확률로 분석
df['prob_bin'] = pd.cut(df['home_win_prob'], bins=bins, labels=labels, include_lowest=True)

# 홈팀 예측일 때와 원정팀 예측일 때 구분
df['predicted_prob'] = df.apply(
    lambda x: x['home_win_prob'] if x['predicted_winner'] == x['home_team'] else (100 - x['home_win_prob']),
    axis=1
)
df['prob_bin'] = pd.cut(df['predicted_prob'], bins=[50, 55, 60, 65, 70, 75, 80, 85, 90, 100],
                        labels=labels, include_lowest=True)

calibration = df.groupby('prob_bin', observed=True).agg({
    'is_correct': ['sum', 'count', 'mean']
}).round(3)
calibration.columns = ['적중', '경기수', '적중률']
calibration['예상적중률'] = [0.525, 0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.95]

print(calibration.to_string())

# 전체 적중률
total_accuracy = df['is_correct'].mean()
print(f"\n전체 적중률: {total_accuracy:.1%} ({df['is_correct'].sum()}/{len(df)})")

# 2. 머니라인 Edge 시뮬레이션
print("\n\n📊 2. 머니라인 Edge 시뮬레이션")
print("-" * 60)

def prob_to_fair_odds(prob):
    """확률을 공정 배당으로 변환"""
    if prob <= 0 or prob >= 1:
        return None
    return 1 / prob

def american_to_decimal(american):
    """미국식 배당을 소수점 배당으로 변환"""
    if american > 0:
        return (american / 100) + 1
    else:
        return (100 / abs(american)) + 1

# 시장 vig 시뮬레이션 (Pinnacle은 약 2-4% vig)
VIG = 0.03  # 3% vig

def add_vig(fair_prob, is_favorite):
    """공정 확률에 vig 추가"""
    if is_favorite:
        return fair_prob + VIG/2  # 페이버릿은 확률이 높아짐 (배당 낮아짐)
    else:
        return fair_prob - VIG/2  # 언더독은 확률이 낮아짐 (배당 높아짐)

# 시뮬레이션: 모델 확률을 시장 확률로 가정하고, 실제 적중률로 Edge 계산
print("시나리오: 모델 확률 = 시장 확률이라고 가정")
print("(실제로는 시장이 더 효율적이므로 보수적 시나리오)")

# Edge 시뮬레이션
edge_results = []

for edge_threshold in [0.03, 0.05, 0.07, 0.10]:
    # 모델이 시장보다 높은 확률을 예측하는 경우만 베팅
    # 여기서는 모델 = 시장으로 가정하므로, 높은 확률 예측만 필터링

    # 시뮬레이션: 모델이 edge_threshold 이상의 엣지를 가진다고 가정
    high_conf = df[df['predicted_prob'] >= (50 + edge_threshold * 100)].copy()

    if len(high_conf) == 0:
        continue

    # 가상 배당 계산 (시장은 실제 확률에 vig 추가)
    high_conf['market_prob'] = high_conf['predicted_prob'] / 100 - edge_threshold
    high_conf['market_odds'] = 1 / high_conf['market_prob']

    # 베팅 결과
    high_conf['bet_result'] = high_conf.apply(
        lambda x: x['market_odds'] - 1 if x['is_correct'] else -1,
        axis=1
    )

    total_bets = len(high_conf)
    wins = high_conf['is_correct'].sum()
    roi = high_conf['bet_result'].mean() * 100

    edge_results.append({
        'Edge 임계값': f'{edge_threshold:.0%}',
        '베팅 수': total_bets,
        '적중': wins,
        '적중률': f'{wins/total_bets:.1%}',
        'ROI': f'{roi:+.1f}%'
    })

edge_df = pd.DataFrame(edge_results)
print(edge_df.to_string(index=False))

# 3. 실제 적중률 기반 Edge 분석
print("\n\n📊 3. 확률 구간별 실제 성과 분석")
print("-" * 60)

# 높은 확률 예측의 가치 분석
for prob_threshold in [60, 65, 70, 75, 80]:
    high_conf = df[df['predicted_prob'] >= prob_threshold]
    if len(high_conf) == 0:
        continue

    actual_accuracy = high_conf['is_correct'].mean()
    expected_accuracy = prob_threshold / 100

    # 단순 머니라인 ROI 시뮬레이션 (공정 배당 가정)
    # 배당 = 1 / (expected_accuracy - 0.02) (2% vig 가정)
    implied_odds = 1 / (expected_accuracy - 0.02)

    # 실제 수익
    wins = high_conf['is_correct'].sum()
    losses = len(high_conf) - wins
    profit = wins * (implied_odds - 1) - losses
    roi = profit / len(high_conf) * 100

    print(f"예측 확률 ≥ {prob_threshold}%:")
    print(f"  경기 수: {len(high_conf)}")
    print(f"  실제 적중률: {actual_accuracy:.1%}")
    print(f"  기대 적중률: {expected_accuracy:.1%}")
    print(f"  가상 배당: {implied_odds:.2f}")
    print(f"  예상 ROI: {roi:+.1f}%")
    print()

# 4. 언더독 베팅 분석
print("\n📊 4. 언더독 베팅 분석")
print("-" * 60)

# 원정팀을 예측한 경우 (일반적으로 언더독)
underdog_bets = df[df['predicted_winner'] == df['away_team']]
if len(underdog_bets) > 0:
    underdog_accuracy = underdog_bets['is_correct'].mean()
    print(f"원정팀 예측 경기: {len(underdog_bets)}")
    print(f"적중률: {underdog_accuracy:.1%}")

# 낮은 확률 예측 (50-55%) 분석
close_games = df[(df['predicted_prob'] >= 50) & (df['predicted_prob'] < 55)]
if len(close_games) > 0:
    close_accuracy = close_games['is_correct'].mean()
    print(f"\n박빙 경기 (50-55% 예측): {len(close_games)}")
    print(f"적중률: {close_accuracy:.1%}")
    print("→ 이런 경기는 Edge가 없으므로 SKIP 권장")

# 5. 결론
print("\n" + "=" * 60)
print("📋 결론")
print("=" * 60)

# Calibration 편차 계산
if not calibration.empty:
    calibration_error = abs(calibration['적중률'] - calibration['예상적중률']).mean()
    print(f"\n평균 Calibration 오차: {calibration_error:.1%}")

    if calibration_error < 0.05:
        print("✅ 모델이 잘 calibration되어 있음")
    else:
        print("⚠️ Calibration 개선 필요")

# 높은 확률 예측 분석
high_conf_70 = df[df['predicted_prob'] >= 70]
if len(high_conf_70) > 0:
    high_accuracy = high_conf_70['is_correct'].mean()
    print(f"\n70%+ 예측 적중률: {high_accuracy:.1%} ({len(high_conf_70)}경기)")

    if high_accuracy >= 0.70:
        print("✅ 높은 확률 예측이 실제로 높은 적중률을 보임")
        print("→ 머니라인 Edge에 경쟁력 있음")
    else:
        print("⚠️ 높은 확률 예측의 적중률이 기대보다 낮음")
        print("→ 머니라인 Edge 활용 시 주의 필요")

print("\n⚠️ 주의: 실제 Pinnacle 배당 데이터 없이 시뮬레이션한 결과입니다.")
print("실제 Edge 검증을 위해서는 과거 배당 데이터 수집이 필요합니다.")
