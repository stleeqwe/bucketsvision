#!/bin/bash
# BucketsVision Launchd Setup Script
#
# 사용법:
#   ./setup_launchd.sh install    # launchd 작업 설치
#   ./setup_launchd.sh uninstall  # launchd 작업 제거
#   ./setup_launchd.sh status     # 상태 확인
#   ./setup_launchd.sh test       # 수동 테스트 실행

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
LOG_DIR="$PROJECT_DIR/logs"

# 작업 목록
JOBS=(
    "com.bucketsvision.pregame"
    "com.bucketsvision.closing"
    "com.bucketsvision.postgame"
)

install() {
    echo "=== BucketsVision Launchd 설치 ==="
    echo ""

    # 로그 디렉토리 생성
    mkdir -p "$LOG_DIR"
    echo "✓ 로그 디렉토리 생성: $LOG_DIR"

    # LaunchAgents 디렉토리 생성
    mkdir -p "$LAUNCH_AGENTS_DIR"

    for job in "${JOBS[@]}"; do
        plist_file="$SCRIPT_DIR/${job}.plist"

        if [ ! -f "$plist_file" ]; then
            echo "✗ plist 파일 없음: $plist_file"
            continue
        fi

        # 기존 작업 언로드
        launchctl unload "$LAUNCH_AGENTS_DIR/${job}.plist" 2>/dev/null

        # 복사 및 로드
        cp "$plist_file" "$LAUNCH_AGENTS_DIR/"
        launchctl load "$LAUNCH_AGENTS_DIR/${job}.plist"

        echo "✓ 설치됨: $job"
    done

    echo ""
    echo "=== 스케줄 ==="
    echo "  pregame:  매일 오전 7시 KST (오후 5시 ET) - 경기 전 예측"
    echo "  closing:  매일 오전 8시 KST (오후 6시 ET) - Closing 배당"
    echo "  postgame: 매일 오후 5시 KST (새벽 3시 ET) - 결과 업데이트"
    echo ""
    echo "상태 확인: ./setup_launchd.sh status"
}

uninstall() {
    echo "=== BucketsVision Launchd 제거 ==="
    echo ""

    for job in "${JOBS[@]}"; do
        plist_path="$LAUNCH_AGENTS_DIR/${job}.plist"

        if [ -f "$plist_path" ]; then
            launchctl unload "$plist_path" 2>/dev/null
            rm "$plist_path"
            echo "✓ 제거됨: $job"
        else
            echo "- 없음: $job"
        fi
    done

    echo ""
    echo "제거 완료"
}

status() {
    echo "=== BucketsVision Launchd 상태 ==="
    echo ""

    for job in "${JOBS[@]}"; do
        status=$(launchctl list | grep "$job" || echo "")

        if [ -n "$status" ]; then
            echo "✓ 활성: $job"
            echo "  $status"
        else
            echo "✗ 비활성: $job"
        fi
    done

    echo ""
    echo "=== 최근 로그 ==="
    for logfile in "$LOG_DIR"/*.log; do
        if [ -f "$logfile" ]; then
            echo ""
            echo "--- $(basename "$logfile") ---"
            tail -5 "$logfile" 2>/dev/null || echo "(비어 있음)"
        fi
    done
}

test_run() {
    echo "=== BucketsVision 수동 테스트 ==="
    echo ""

    echo "[1/3] Pre-game 테스트..."
    python3 "$PROJECT_DIR/scripts/verified_tracking_runner.py" pregame

    echo ""
    echo "[2/3] Closing 테스트..."
    python3 "$PROJECT_DIR/scripts/verified_tracking_runner.py" closing

    echo ""
    echo "[3/3] Post-game 테스트..."
    python3 "$PROJECT_DIR/scripts/verified_tracking_runner.py" postgame

    echo ""
    echo "테스트 완료"
}

case "$1" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    status)
        status
        ;;
    test)
        test_run
        ;;
    *)
        echo "사용법: $0 {install|uninstall|status|test}"
        exit 1
        ;;
esac
