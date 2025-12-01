"""
시즌 전체 일일 스냅샷 일괄 생성

기존에 없는 스냅샷만 생성합니다.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.daily_snapshot import create_daily_snapshot, save_snapshot, get_et_today


def main():
    print("=" * 60)
    print("시즌 전체 스냅샷 일괄 생성")
    print("=" * 60)

    season_start = date(2025, 10, 22)
    et_today = get_et_today()

    # 어제까지 (오늘 경기는 아직 진행 중일 수 있음)
    end_date = et_today - timedelta(days=1)

    print(f"기간: {season_start} ~ {end_date}\n")

    snapshot_dir = project_root / "data" / "snapshots"
    created_count = 0
    skipped_count = 0
    no_games_count = 0

    current_date = season_start
    while current_date <= end_date:
        # 이미 존재하는지 확인
        year_dir = snapshot_dir / str(current_date.year)
        filepath = year_dir / f"{current_date.isoformat()}_snapshot.json"

        if filepath.exists():
            skipped_count += 1
            current_date += timedelta(days=1)
            continue

        # 스냅샷 생성
        try:
            snapshot = create_daily_snapshot(current_date)
            if snapshot is None:
                no_games_count += 1
            else:
                saved_path = save_snapshot(snapshot, current_date)
                if saved_path:
                    created_count += 1
                    print(f"   적중률: {snapshot['summary']['accuracy_pct']}%")
        except Exception as e:
            print(f"❌ {current_date} 스냅샷 생성 실패: {e}")

        current_date += timedelta(days=1)

    print(f"\n{'=' * 60}")
    print("완료")
    print(f"{'=' * 60}")
    print(f"생성: {created_count}개")
    print(f"스킵 (이미 존재): {skipped_count}개")
    print(f"경기 없음: {no_games_count}개")


if __name__ == "__main__":
    main()
