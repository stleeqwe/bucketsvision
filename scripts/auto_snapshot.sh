#!/bin/bash
#
# BucketsVision 일일 스냅샷 자동 생성 스크립트
# 매일 오후 6시 (KST) = 오전 4시 (ET) 실행 권장
#

# 프로젝트 디렉토리
PROJECT_DIR="/Users/stlee/Desktop/bucketsvision"
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/snapshot_$(date +%Y%m%d).log"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 로그 시작
echo "========================================" >> "$LOG_FILE"
echo "스냅샷 생성 시작: $(date)" >> "$LOG_FILE"
echo "========================================" >> "$LOG_FILE"

# Python 경로 (시스템 Python 사용)
PYTHON="/usr/bin/python3"

# 스냅샷 생성 (어제 경기)
cd "$PROJECT_DIR"
$PYTHON scripts/daily_snapshot.py >> "$LOG_FILE" 2>&1

# 결과 확인
if [ $? -eq 0 ]; then
    echo "✅ 스냅샷 생성 완료" >> "$LOG_FILE"
else
    echo "❌ 스냅샷 생성 실패" >> "$LOG_FILE"
fi

echo "완료: $(date)" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 30일 이상된 로그 정리
find "$LOG_DIR" -name "snapshot_*.log" -mtime +30 -delete 2>/dev/null
