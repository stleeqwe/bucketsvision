#!/bin/bash
#
# Pinnacle 배당 자동 캡처 스크립트
# launchd에서 실행됨
#

cd /Users/stlee/Desktop/bucketsvision

# 현재 시간 (한국 시간)
HOUR=$(date +%H)

# 라벨 결정
if [ "$HOUR" -eq 8 ]; then
    LABEL="3h_before"
elif [ "$HOUR" -eq 10 ]; then
    LABEL="1h_before"
elif [ "$HOUR" -eq 11 ] || [ "$HOUR" -eq 12 ]; then
    LABEL="closing"
else
    LABEL="manual"
fi

echo "=========================================="
echo "배당 캡처 시작: $(date)"
echo "라벨: $LABEL"
echo "=========================================="

# Python 실행
/usr/bin/python3 scripts/capture_odds.py --label "$LABEL"

echo "완료: $(date)"
