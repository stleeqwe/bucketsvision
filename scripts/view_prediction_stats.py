#!/usr/bin/env python3
"""
예측 통계 조회 스크립트.

저장된 예측 기록을 분석하고 통계를 출력합니다.

사용법:
    # 전체 통계 조회
    python scripts/view_prediction_stats.py

    # 최근 예측 조회
    python scripts/view_prediction_stats.py --recent 20

    # 날짜별 성과 조회
    python scripts/view_prediction_stats.py --daily

    # 특정 날짜 상세 조회
    python scripts/view_prediction_stats.py --date 2025-11-29

    # CSV 내보내기
    python scripts/view_prediction_stats.py --export predictions.csv
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date
import csv

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.prediction_tracker import PredictionTracker


def print_overall_stats(tracker: PredictionTracker, season: int = 2026):
    """전체 통계 출력"""
    stats = tracker.get_stats(season)

    print(f"\n{'='*60}")
    print(f"  시즌 {season} 예측 통계")
    print(f"{'='*60}")

    print(f"\n[전체 성과]")
    print(f"  총 예측:     {stats['total_predictions']}경기")
    print(f"  완료 경기:   {stats['completed_games']}경기")
    print(f"  적중:        {stats['correct_predictions']}경기")
    print(f"  적중률:      {stats['accuracy']:.1%}")
    print(f"  대기 중:     {stats['pending_games']}경기")

    if stats.get('confidence_stats'):
        print(f"\n[신뢰도별 적중률]")
        print(f"  {'신뢰도':<10} {'경기수':>8} {'적중':>8} {'적중률':>10}")
        print(f"  {'-'*40}")

        for conf, data in sorted(stats['confidence_stats'].items()):
            threshold = int(conf.split('_')[1])
            print(f"  {threshold}%+{'':<6} {data['count']:>8} {data['correct']:>8} {data['accuracy']:>10.1%}")


def print_recent_predictions(tracker: PredictionTracker, n: int = 10, season: int = 2026):
    """최근 예측 출력"""
    records = tracker.get_recent_predictions(n, season, only_completed=False)

    print(f"\n{'='*60}")
    print(f"  최근 {n}개 예측")
    print(f"{'='*60}")

    print(f"\n  {'날짜':<12} {'매치업':<15} {'예측':<6} {'확률':>8} {'결과':<10} {'적중':<4}")
    print(f"  {'-'*60}")

    for r in records:
        matchup = f"{r.away_team} @ {r.home_team}"
        prob = f"{r.home_win_prob:.0%}"

        if r.actual_home_score is not None:
            result = f"{r.actual_away_score}-{r.actual_home_score}"
            status = "✓" if r.is_correct else "✗"
        else:
            result = "대기중"
            status = "-"

        print(f"  {r.game_date:<12} {matchup:<15} {r.predicted_winner:<6} {prob:>8} {result:<10} {status:<4}")


def print_daily_performance(tracker: PredictionTracker, season: int = 2026):
    """날짜별 성과 출력"""
    daily_stats = tracker.get_model_performance_by_date(season)

    print(f"\n{'='*60}")
    print(f"  날짜별 성과 (시즌 {season})")
    print(f"{'='*60}")

    print(f"\n  {'날짜':<12} {'경기':>6} {'적중':>6} {'적중률':>10}")
    print(f"  {'-'*40}")

    total_games = 0
    total_correct = 0

    for day in daily_stats:
        print(f"  {day['date']:<12} {day['total']:>6} {day['correct']:>6} {day['accuracy']:>10.1%}")
        total_games += day['total']
        total_correct += day['correct']

    print(f"  {'-'*40}")
    if total_games > 0:
        print(f"  {'합계':<12} {total_games:>6} {total_correct:>6} {total_correct/total_games:>10.1%}")


def print_date_detail(tracker: PredictionTracker, target_date: date, season: int = 2026):
    """특정 날짜 상세 출력"""
    records = tracker.get_predictions_by_date(target_date, season)

    if not records:
        print(f"\n{target_date} 예측 기록이 없습니다.")
        return

    print(f"\n{'='*60}")
    print(f"  {target_date} 예측 상세")
    print(f"{'='*60}")

    correct = 0
    completed = 0

    for r in records:
        matchup = f"{r.away_team} @ {r.home_team}"
        prob = f"{r.home_win_prob:.1%}"

        print(f"\n  [{r.game_id}] {matchup}")
        print(f"    예측: {r.predicted_winner} (홈 승률: {prob})")
        print(f"    EPM 차이: {r.team_epm_diff:+.2f}")
        print(f"    선수 로테이션 EPM 차이: {r.player_rotation_epm_diff:+.2f}")
        print(f"    벤치 강도 차이: {r.bench_strength_diff:+.2f}")

        if r.actual_home_score is not None:
            completed += 1
            result = f"{r.actual_away_score}-{r.actual_home_score}"
            status = "✓ 적중" if r.is_correct else "✗ 오답"
            print(f"    결과: {result} ({r.actual_winner} 승) - {status}")
            if r.is_correct:
                correct += 1
        else:
            print(f"    결과: 대기 중")

    print(f"\n  {'='*40}")
    print(f"  {target_date} 요약: {len(records)}경기 중 {completed}경기 완료, {correct}/{completed} 적중")
    if completed > 0:
        print(f"  적중률: {correct/completed:.1%}")


def export_to_csv(tracker: PredictionTracker, output_path: str, season: int = 2026):
    """CSV로 내보내기"""
    records = tracker.load_season_records(season)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 헤더
        writer.writerow([
            'game_id', 'game_date', 'game_time_et',
            'away_team', 'home_team',
            'model_version', 'home_win_prob', 'predicted_winner', 'predicted_margin',
            'team_epm_diff', 'player_rotation_epm_diff', 'bench_strength_diff',
            'actual_away_score', 'actual_home_score', 'actual_winner', 'is_correct',
            'prediction_time', 'result_updated_at'
        ])

        # 데이터
        for r in sorted(records.values(), key=lambda x: x.game_date):
            writer.writerow([
                r.game_id, r.game_date, r.game_time_et,
                r.away_team, r.home_team,
                r.model_version, r.home_win_prob, r.predicted_winner, r.predicted_margin,
                r.team_epm_diff, r.player_rotation_epm_diff, r.bench_strength_diff,
                r.actual_away_score, r.actual_home_score, r.actual_winner, r.is_correct,
                r.prediction_time, r.result_updated_at
            ])

    print(f"\n내보내기 완료: {output_path} ({len(records)}경기)")


def main():
    parser = argparse.ArgumentParser(description='예측 통계 조회')
    parser.add_argument('--recent', type=int, help='최근 N개 예측 조회')
    parser.add_argument('--daily', action='store_true', help='날짜별 성과 조회')
    parser.add_argument('--date', type=str, help='특정 날짜 상세 조회 (YYYY-MM-DD)')
    parser.add_argument('--export', type=str, help='CSV 내보내기 경로')
    parser.add_argument('--season', type=int, default=2026, help='시즌 연도')

    args = parser.parse_args()

    # 서비스 초기화
    data_dir = PROJECT_ROOT / "data"
    tracker = PredictionTracker(data_dir)

    if args.export:
        export_to_csv(tracker, args.export, args.season)
    elif args.date:
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        print_date_detail(tracker, target_date, args.season)
    elif args.daily:
        print_daily_performance(tracker, args.season)
    elif args.recent:
        print_recent_predictions(tracker, args.recent, args.season)
    else:
        # 기본: 전체 통계
        print_overall_stats(tracker, args.season)
        print_daily_performance(tracker, args.season)


if __name__ == "__main__":
    main()
