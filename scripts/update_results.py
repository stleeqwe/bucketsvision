#!/usr/bin/env python3
"""
경기 결과 업데이트 스크립트.

기록된 예측에 대해 실제 경기 결과를 업데이트하고
적중 여부를 확인합니다.

사용법:
    # 모든 미완료 경기 결과 업데이트
    python scripts/update_results.py

    # 특정 날짜 경기만 업데이트
    python scripts/update_results.py --date 2025-11-29

    # 상세 출력
    python scripts/update_results.py --verbose

자동 실행 (cron):
    # 매일 오전 6시 (ET) - 전날 경기 결과 반영
    0 6 * * * cd /path/to/bucketsvision && python scripts/update_results.py
"""

import sys
from pathlib import Path
import argparse
from datetime import datetime, date, timedelta
import pytz

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.data_loader import DataLoader
from app.services.prediction_tracker import PredictionTracker
from config.constants import TEAM_INFO
from src.utils.helpers import get_season_from_date


def update_results_for_date(
    target_date: date,
    tracker: PredictionTracker,
    loader: DataLoader,
    verbose: bool = False
) -> tuple:
    """
    특정 날짜의 경기 결과 업데이트.

    Returns:
        (업데이트 수, 적중 수)
    """
    if verbose:
        print(f"\n[{target_date}] 결과 업데이트")

    # 경기 결과 가져오기
    games = loader.get_games(target_date)

    updated = 0
    correct = 0

    for game in games:
        game_id = game['game_id']
        game_status = game.get('game_status', 1)
        home_score = game.get('home_score')
        away_score = game.get('away_score')

        # 종료된 경기만 처리
        if game_status != 3 or home_score is None or away_score is None:
            continue

        # 결과 업데이트
        record = tracker.update_result(
            game_id=game_id,
            home_score=home_score,
            away_score=away_score,
            season=get_season_from_date(target_date)
        )

        if record and record.result_updated_at:
            updated += 1
            if record.is_correct:
                correct += 1

            if verbose:
                status = "✓" if record.is_correct else "✗"
                print(f"  {status} [{game_id}] {record.away_team} @ {record.home_team}: "
                      f"{away_score}-{home_score} (예측: {record.predicted_winner})")

    if verbose:
        print(f"  업데이트: {updated}경기, 적중: {correct}경기")

    return updated, correct


def update_all_pending(
    tracker: PredictionTracker,
    loader: DataLoader,
    season: int = 2026,
    verbose: bool = False
) -> tuple:
    """
    모든 미완료 경기 결과 업데이트.

    Returns:
        (업데이트 수, 적중 수)
    """
    print(f"\n{'='*60}")
    print(f"미완료 경기 결과 업데이트 (시즌 {season})")
    print(f"{'='*60}")

    records = tracker.load_season_records(season)

    # 결과가 없는 경기의 날짜들 수집
    pending_dates = set()
    for record in records.values():
        if record.actual_home_score is None:
            try:
                game_date = datetime.strptime(record.game_date, '%Y-%m-%d').date()
                pending_dates.add(game_date)
            except:
                pass

    if not pending_dates:
        print("업데이트할 경기가 없습니다.")
        return 0, 0

    print(f"대기 중인 날짜: {len(pending_dates)}일")

    total_updated = 0
    total_correct = 0

    for game_date in sorted(pending_dates):
        updated, correct = update_results_for_date(
            game_date, tracker, loader, verbose
        )
        total_updated += updated
        total_correct += correct

    print(f"\n총 업데이트: {total_updated}경기, 적중: {total_correct}경기")
    if total_updated > 0:
        print(f"적중률: {total_correct/total_updated*100:.1f}%")

    return total_updated, total_correct


def main():
    parser = argparse.ArgumentParser(description='경기 결과 업데이트')
    parser.add_argument('--date', type=str, help='대상 날짜 (YYYY-MM-DD)')
    parser.add_argument('--verbose', '-v', action='store_true', help='상세 출력')
    parser.add_argument('--season', type=int, default=2026, help='시즌 연도')

    args = parser.parse_args()

    # 서비스 초기화
    data_dir = PROJECT_ROOT / "data"
    loader = DataLoader(data_dir)
    tracker = PredictionTracker(data_dir)

    if args.date:
        # 특정 날짜만 업데이트
        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
        update_results_for_date(target_date, tracker, loader, verbose=True)
    else:
        # 모든 미완료 경기 업데이트
        update_all_pending(tracker, loader, args.season, args.verbose)

    # 현재 통계 출력
    stats = tracker.get_stats(args.season)
    print(f"\n{'='*60}")
    print(f"시즌 {args.season} 전체 통계")
    print(f"{'='*60}")
    print(f"  총 예측: {stats['total_predictions']}경기")
    print(f"  완료: {stats['completed_games']}경기")
    print(f"  적중: {stats['correct_predictions']}경기 ({stats['accuracy']:.1%})")
    print(f"  대기: {stats['pending_games']}경기")

    # 신뢰도별 통계
    if stats.get('confidence_stats'):
        print(f"\n[신뢰도별 정확도]")
        for conf, data in sorted(stats['confidence_stats'].items()):
            threshold = int(conf.split('_')[1])
            print(f"  {threshold}%+: {data['correct']}/{data['count']} ({data['accuracy']:.1%})")


if __name__ == "__main__":
    main()
