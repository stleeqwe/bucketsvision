#!/usr/bin/env python3
"""
BucketsVision Verified Tracking Runner

불변 증빙 시스템을 위한 통합 실행 스크립트.
모든 작업을 적절한 시간에 자동으로 실행합니다.

스케줄:
- 오후 5시 ET (새벽 7시 KST): Pre-game 예측 스냅샷
- 오후 6시 ET (새벽 8시 KST): Closing 배당 캡처
- 새벽 3시 ET (오후 5시 KST): 결과 업데이트 + Paper betting 정산

사용법:
    python scripts/verified_tracking_runner.py pregame    # 경기 전 예측 기록
    python scripts/verified_tracking_runner.py closing    # Closing 배당 캡처
    python scripts/verified_tracking_runner.py postgame   # 결과 업데이트 + 베팅 정산
    python scripts/verified_tracking_runner.py all        # 전체 실행 (테스트용)
    python scripts/verified_tracking_runner.py verify     # 무결성 검증
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

import pytz

PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
LOG_DIR = PROJECT_ROOT / "logs"


def get_et_now():
    """미국 동부 시간 현재"""
    return datetime.now(pytz.timezone('America/New_York'))


def log(message: str):
    """로그 출력 (타임스탬프 포함)"""
    et_now = get_et_now()
    timestamp = et_now.strftime('%Y-%m-%d %H:%M:%S ET')
    print(f"[{timestamp}] {message}")


def run_script(script_name: str, args: list = None) -> bool:
    """스크립트 실행"""
    script_path = SCRIPTS_DIR / script_name

    if not script_path.exists():
        log(f"ERROR: Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    log(f"Running: {script_name} {' '.join(args or [])}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            timeout=600  # 10분 타임아웃
        )

        if result.returncode == 0:
            log(f"SUCCESS: {script_name}")
            return True
        else:
            log(f"FAILED: {script_name} (exit code: {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        log(f"TIMEOUT: {script_name}")
        return False
    except Exception as e:
        log(f"ERROR: {script_name} - {e}")
        return False


def run_pregame():
    """
    Pre-game 작업: 경기 전 예측 스냅샷

    실행 시간: 오후 5시 ET (새벽 7시 KST)
    - 대부분의 NBA 경기는 오후 7시 ET 이후 시작
    - 경기 시작 최소 2시간 전에 예측 확정
    """
    log("="*60)
    log("PRE-GAME SNAPSHOT")
    log("="*60)

    # 1. Pre-game 예측 스냅샷 (해시 체인)
    run_script("pregame_snapshot.py")

    log("Pre-game tasks completed")


def run_closing():
    """
    Closing 배당 캡처

    실행 시간: 오후 6시 ET (새벽 8시 KST)
    - 경기 시작 약 1시간 전 배당 (closing line)
    - Paper betting의 edge 계산에 사용
    """
    log("="*60)
    log("CLOSING ODDS CAPTURE")
    log("="*60)

    # Closing 배당 캡처
    run_script("capture_odds.py", ["--label", "closing"])

    log("Closing odds capture completed")


def run_postgame():
    """
    Post-game 작업: 결과 업데이트 + Paper betting 정산

    실행 시간: 새벽 3시 ET (오후 5시 KST)
    - 어제(ET) 경기 결과 확정
    - 예측 적중 여부 기록
    - Paper betting P&L 정산
    """
    log("="*60)
    log("POST-GAME UPDATE")
    log("="*60)

    # 1. 일일 결과 스냅샷
    log("\n[1/3] Daily result snapshot")
    run_script("daily_snapshot.py")

    # 2. Paper betting 기록 + 결과 업데이트
    log("\n[2/3] Paper betting update")
    run_script("paper_betting.py")

    # 3. Paper betting 리포트 생성
    log("\n[3/3] Generate report")
    run_script("paper_betting.py", ["--report-only"])

    log("Post-game tasks completed")


def run_verify():
    """무결성 검증"""
    log("="*60)
    log("INTEGRITY VERIFICATION")
    log("="*60)

    # Pre-game 스냅샷 체인 검증
    log("\n[1/2] Pregame snapshot chain verification")
    run_script("pregame_snapshot.py", ["--verify"])

    # 일일 스냅샷 검증
    log("\n[2/2] Daily snapshot verification")
    run_script("daily_snapshot.py", ["--verify-all"])

    log("Verification completed")


def run_all():
    """전체 실행 (테스트용)"""
    log("="*60)
    log("RUNNING ALL TASKS (TEST MODE)")
    log("="*60)

    run_pregame()
    print()
    run_closing()
    print()
    run_postgame()
    print()
    run_verify()


def main():
    parser = argparse.ArgumentParser(
        description="BucketsVision Verified Tracking Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
스케줄:
  pregame   오후 5시 ET - 경기 전 예측 스냅샷 (해시 체인)
  closing   오후 6시 ET - Closing 배당 캡처
  postgame  새벽 3시 ET - 결과 업데이트 + Paper betting 정산
  verify    무결성 검증
  all       전체 실행 (테스트용)
        """
    )
    parser.add_argument(
        "task",
        choices=["pregame", "closing", "postgame", "verify", "all"],
        help="실행할 작업"
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)

    if args.task == "pregame":
        run_pregame()
    elif args.task == "closing":
        run_closing()
    elif args.task == "postgame":
        run_postgame()
    elif args.task == "verify":
        run_verify()
    elif args.task == "all":
        run_all()


if __name__ == "__main__":
    main()
