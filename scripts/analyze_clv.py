#!/usr/bin/env python3
"""
CLV (Closing Line Value) 분석 스크립트.

스냅샷에 저장된 배당 정보를 분석하여:
1. Edge가 있었던 베팅의 실제 수익률 계산
2. CLV 추적 (모델 vs 시장)
3. 장기 수익성 검증
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import date

import pandas as pd
import numpy as np

# 프로젝트 루트
project_root = Path(__file__).parent.parent


def load_all_snapshots() -> List[Dict]:
    """모든 스냅샷 로드"""
    snapshot_dir = project_root / "data" / "snapshots"
    if not snapshot_dir.exists():
        print("❌ 스냅샷 디렉토리가 없습니다.")
        return []

    snapshots = []
    for filepath in sorted(snapshot_dir.rglob("*_snapshot.json")):
        with open(filepath, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
            snapshots.append(snapshot)

    return snapshots


def extract_betting_data(snapshots: List[Dict]) -> pd.DataFrame:
    """스냅샷에서 베팅 데이터 추출"""
    records = []

    for snapshot in snapshots:
        game_date = snapshot.get("meta", {}).get("game_date_et", "")

        for pred in snapshot.get("predictions", []):
            odds = pred.get("odds")
            if not odds:
                continue

            # 필수 데이터 확인
            ml_home = odds.get("moneyline_home")
            ml_away = odds.get("moneyline_away")
            edge_home = odds.get("edge_home")
            edge_away = odds.get("edge_away")

            if None in [ml_home, ml_away, edge_home, edge_away]:
                continue

            # 경기 결과 확인
            if pred.get("game_status") != 3:
                continue
            if pred.get("is_correct") is None:
                continue

            home_won = pred.get("actual_winner") == pred.get("home_team")

            records.append({
                "date": game_date,
                "game_id": pred.get("game_id"),
                "home_team": pred.get("home_team"),
                "away_team": pred.get("away_team"),
                "model_prob_home": pred.get("home_win_prob"),
                "market_prob_home": odds.get("market_prob_home"),
                "market_prob_away": odds.get("market_prob_away"),
                "ml_home": ml_home,
                "ml_away": ml_away,
                "edge_home": edge_home,
                "edge_away": edge_away,
                "spread_home": odds.get("spread_home"),
                "predicted_winner": pred.get("predicted_winner"),
                "actual_winner": pred.get("actual_winner"),
                "is_correct": pred.get("is_correct"),
                "home_won": home_won,
            })

    return pd.DataFrame(records)


def simulate_betting(df: pd.DataFrame, edge_threshold: float = 3.0) -> Dict:
    """
    Edge 기반 베팅 시뮬레이션.

    Args:
        df: 베팅 데이터
        edge_threshold: 베팅 진입 Edge 임계값 (%)

    Returns:
        시뮬레이션 결과
    """
    if df.empty:
        return {"error": "데이터 없음"}

    bets = []

    for _, row in df.iterrows():
        # 홈팀 Edge 확인
        if row["edge_home"] >= edge_threshold:
            # 홈팀 베팅
            if row["home_won"]:
                profit = row["ml_home"] - 1  # 승리: 배당 - 1
            else:
                profit = -1  # 패배: -1 단위

            bets.append({
                "date": row["date"],
                "game_id": row["game_id"],
                "bet_team": row["home_team"],
                "bet_side": "home",
                "edge": row["edge_home"],
                "odds": row["ml_home"],
                "won": row["home_won"],
                "profit": profit,
            })

        # 원정팀 Edge 확인
        elif row["edge_away"] >= edge_threshold:
            # 원정팀 베팅
            if not row["home_won"]:
                profit = row["ml_away"] - 1
            else:
                profit = -1

            bets.append({
                "date": row["date"],
                "game_id": row["game_id"],
                "bet_team": row["away_team"],
                "bet_side": "away",
                "edge": row["edge_away"],
                "odds": row["ml_away"],
                "won": not row["home_won"],
                "profit": profit,
            })

    if not bets:
        return {
            "edge_threshold": edge_threshold,
            "total_bets": 0,
            "message": "조건에 맞는 베팅 없음"
        }

    bets_df = pd.DataFrame(bets)

    total_bets = len(bets_df)
    wins = bets_df["won"].sum()
    total_profit = bets_df["profit"].sum()
    roi = total_profit / total_bets * 100

    return {
        "edge_threshold": edge_threshold,
        "total_bets": total_bets,
        "wins": wins,
        "losses": total_bets - wins,
        "win_rate": wins / total_bets * 100,
        "total_profit": total_profit,
        "roi": roi,
        "avg_edge": bets_df["edge"].mean(),
        "avg_odds": bets_df["odds"].mean(),
    }


def analyze_edge_performance(df: pd.DataFrame) -> None:
    """Edge별 성과 분석"""
    print("\n" + "=" * 70)
    print("📊 Edge별 베팅 성과 분석")
    print("=" * 70)

    # 다양한 Edge 임계값으로 시뮬레이션
    results = []
    for threshold in [0, 3, 5, 7, 10, 15]:
        result = simulate_betting(df, threshold)
        if "total_bets" in result and result["total_bets"] > 0:
            results.append(result)

    if not results:
        print("❌ 배당 데이터가 있는 경기가 없습니다.")
        return

    # 결과 테이블
    results_df = pd.DataFrame(results)
    print("\n| Edge 임계값 | 베팅 수 | 적중 | 적중률 | 총 수익 | ROI | 평균 Edge | 평균 배당 |")
    print("|-------------|---------|------|--------|---------|-----|-----------|-----------|")

    for _, row in results_df.iterrows():
        print(f"| {row['edge_threshold']:>6.0f}% | {row['total_bets']:>7} | {row['wins']:>4.0f} | "
              f"{row['win_rate']:>5.1f}% | {row['total_profit']:>+7.2f} | {row['roi']:>+5.1f}% | "
              f"{row['avg_edge']:>9.1f}% | {row['avg_odds']:>9.2f} |")


def analyze_by_date(df: pd.DataFrame) -> None:
    """일별 성과 분석"""
    print("\n" + "=" * 70)
    print("📅 일별 Edge 베팅 성과")
    print("=" * 70)

    if df.empty:
        print("❌ 데이터 없음")
        return

    # Edge >= 3% 베팅만
    edge_bets = df[(df["edge_home"] >= 3) | (df["edge_away"] >= 3)].copy()

    if edge_bets.empty:
        print("❌ Edge >= 3% 베팅 없음")
        return

    # 일별 집계
    daily = edge_bets.groupby("date").agg({
        "game_id": "count",
        "is_correct": "sum"
    }).rename(columns={"game_id": "bets", "is_correct": "wins"})

    daily["win_rate"] = daily["wins"] / daily["bets"] * 100

    print("\n최근 10일:")
    print(daily.tail(10).to_string())


def main():
    print("=" * 70)
    print("BucketsVision CLV 분석")
    print("=" * 70)

    # 스냅샷 로드
    snapshots = load_all_snapshots()
    print(f"\n📂 로드된 스냅샷: {len(snapshots)}개")

    if not snapshots:
        print("❌ 스냅샷이 없습니다. 먼저 daily_snapshot.py를 실행하세요.")
        return

    # 베팅 데이터 추출
    df = extract_betting_data(snapshots)
    print(f"📊 배당 데이터 있는 경기: {len(df)}개")

    if df.empty:
        print("\n❌ 배당 정보가 있는 경기가 없습니다.")
        print("💡 오늘부터 스냅샷에 배당 정보가 저장됩니다.")
        print("   일주일 후 다시 분석해보세요.")
        return

    # 기본 통계
    print(f"\n📈 기본 통계:")
    print(f"  - 기간: {df['date'].min()} ~ {df['date'].max()}")
    print(f"  - 총 경기: {len(df)}")
    print(f"  - 모델 적중률: {df['is_correct'].mean()*100:.1f}%")

    # Edge 분석
    edge_home_positive = (df["edge_home"] > 0).sum()
    edge_away_positive = (df["edge_away"] > 0).sum()
    print(f"  - 홈팀 +Edge 경기: {edge_home_positive}")
    print(f"  - 원정팀 +Edge 경기: {edge_away_positive}")

    # Edge별 성과
    analyze_edge_performance(df)

    # 일별 분석
    analyze_by_date(df)

    # 권장사항
    print("\n" + "=" * 70)
    print("💡 권장사항")
    print("=" * 70)

    # Edge >= 5% 결과 확인
    result_5 = simulate_betting(df, 5)
    if result_5.get("total_bets", 0) > 10:
        roi = result_5.get("roi", 0)
        if roi > 0:
            print(f"✅ Edge ≥ 5% 베팅: ROI {roi:+.1f}% → 유망한 전략")
        else:
            print(f"⚠️ Edge ≥ 5% 베팅: ROI {roi:+.1f}% → 추가 검증 필요")
    else:
        print("📊 데이터 축적 중... 최소 20경기 이상 필요")


if __name__ == "__main__":
    main()
