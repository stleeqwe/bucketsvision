# BucketsVision 불변 증빙 시스템

시즌 종료 후 예측 정확도와 Paper Trading 성과를 검증 가능한 형태로 기록하는 시스템입니다.

> **현재 모델**: V5.4 Logistic Regression (5개 피처, 78.05% 정확도)

## 목적

1. **예측 정확도 증빙**: 경기 전 예측이 실제로 적중했는지 검증
2. **Paper Trading 증빙**: 모델 예측 vs Pinnacle 배당 비교를 통한 Edge 기반 가상 베팅 기록

## 시스템 구조

```
┌─────────────────────────────────────────────────────────────────────┐
│                       불변 증빙 시스템                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  [1] Pre-Game Snapshot (경기 전 예측 기록)                          │
│      시간: 오후 5시 ET (새벽 7시 KST)                                │
│      내용: V5.4 모델 예측, 피처, 배당, 타임스탬프                     │
│      특징: SHA-256 해시 체인으로 이전 스냅샷과 연결                   │
│      파일: data/verified_predictions/YYYY/YYYY-MM-DD_pregame.json   │
│                                                                     │
│  [2] Closing Odds Capture (배당 확정)                               │
│      시간: 오후 6시 ET (새벽 8시 KST)                                │
│      내용: 경기 시작 1시간 전 Pinnacle 배당                          │
│      파일: data/odds_history/YYYY-MM-DD/closing_HHMM.json           │
│                                                                     │
│  [3] Post-Game Update (결과 업데이트)                               │
│      시간: 새벽 3시 ET (오후 5시 KST)                                │
│      내용: 실제 결과, 적중 여부, Paper Trading P&L 정산              │
│      파일: data/snapshots/YYYY/YYYY-MM-DD_snapshot.json             │
│             data/paper_betting/bets.json                            │
│             data/paper_betting/BETTING_REPORT.md                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 불변성 보장 메커니즘

### 1. SHA-256 해시 체인

각 Pre-Game 스냅샷은 이전 스냅샷의 해시를 포함합니다:

```json
{
  "chain": {
    "sequence": 42,
    "previous_hash": "abc123...",
    "predictions_hash": "def456...",
    "chain_hash": "ghi789...",
    "timestamp": "2025-12-02T22:00:00Z"
  }
}
```

- **sequence**: 스냅샷 순번 (1부터 시작)
- **previous_hash**: 이전 스냅샷의 chain_hash
- **predictions_hash**: 현재 예측 데이터의 해시
- **chain_hash**: 체인 데이터 전체의 해시
- **timestamp**: UTC 타임스탬프

### 2. 파일 덮어쓰기 방지

이미 존재하는 스냅샷 파일은 절대 덮어쓰지 않습니다.

### 3. 무결성 검증

```bash
# 전체 해시 체인 검증
python scripts/pregame_snapshot.py --verify

# 일일 스냅샷 검증
python scripts/daily_snapshot.py --verify-all
```

## 자동화 스케줄

### 설치

```bash
cd scripts/launchd
./setup_launchd.sh install
```

### 스케줄 확인

| 작업 | 시간 (KST) | 시간 (ET) | 설명 |
|------|-----------|----------|------|
| pregame | 07:00 | 17:00 | 경기 전 예측 스냅샷 |
| closing | 08:00 | 18:00 | Closing 배당 캡처 |
| postgame | 17:00 | 03:00 | 결과 업데이트 + P&L |

### 상태 확인

```bash
./setup_launchd.sh status
```

### 수동 실행

```bash
# 개별 실행
python scripts/verified_tracking_runner.py pregame
python scripts/verified_tracking_runner.py closing
python scripts/verified_tracking_runner.py postgame

# 전체 실행 (테스트용)
python scripts/verified_tracking_runner.py all

# 무결성 검증
python scripts/verified_tracking_runner.py verify
```

## 데이터 구조

### Pre-Game Snapshot

```json
{
  "meta": {
    "version": "2.0",
    "type": "pregame_prediction",
    "model": {
      "name": "V5.4 Logistic Regression",
      "version": "5.4.0",
      "n_features": 5
    },
    "captured_at": "2025-12-02T22:00:00Z",
    "game_date_et": "2025-12-02",
    "total_games": 6
  },
  "predictions": [
    {
      "game_id": "0022500001",
      "home_team": "LAL",
      "away_team": "BOS",
      "prediction": {
        "home_win_prob": 0.5234,
        "predicted_winner": "LAL",
        "confidence": 0.0468
      },
      "features": {
        "team_epm_diff": 1.234,
        "sos_diff": 0.5,
        "bench_strength_diff": 0.8,
        "top5_epm_diff": 1.1,
        "ft_rate_diff": 0.02
      },
      "odds": {
        "bookmaker": "pinnacle",
        "moneyline_home": 1.95,
        "moneyline_away": 1.90,
        "edge_home": 0.0234
      }
    }
  ],
  "chain": {
    "sequence": 42,
    "previous_hash": "abc123...",
    "predictions_hash": "def456...",
    "chain_hash": "ghi789...",
    "timestamp": "2025-12-02T22:00:00Z"
  }
}
```

### Paper Trading Bet Record

```json
{
  "game_id": "0022500001",
  "date": "2025-12-02",
  "home_team": "LAL",
  "away_team": "BOS",
  "bet_side": "home",
  "bet_team": "LAL",
  "bet_odds": 1.95,
  "bet_edge": 0.0523,
  "model_home_prob": 0.5523,
  "unit_size": 100,
  "status": "settled",
  "result": "win",
  "profit": 95.0
}
```

## Paper Trading 규칙

1. **Edge 기준**: ≥ 5% (모델 확률 - Pinnacle fair probability)
2. **베팅 단위**: $100 flat
3. **배당 소스**: Pinnacle Closing Line (경기 1시간 전)
4. **정산 시점**: 경기 종료 후 다음 날 새벽

## 시즌 종료 후 검증

1. **해시 체인 검증**: 모든 스냅샷의 연결 무결성 확인
2. **예측 적중률 계산**: 사전 예측 vs 실제 결과
3. **Paper Trading ROI**: 총 수익 / 총 베팅액

```bash
# 전체 검증 실행
python scripts/verified_tracking_runner.py verify

# 시즌 리포트 생성
python scripts/paper_betting.py --report-only
```

## 로그 파일

```
logs/
├── pregame.log        # Pre-game 스냅샷 로그
├── pregame_error.log
├── closing.log        # Closing 배당 캡처 로그
├── closing_error.log
├── postgame.log       # Post-game 업데이트 로그
└── postgame_error.log
```

## 주의사항

1. **시간대 주의**: 모든 시간은 미국 동부 시간(ET) 기준
2. **해시 체인 연속성**: 하루라도 빠지면 체인이 끊어짐
3. **Pinnacle 배당**: API 제한으로 인해 배당이 없는 경기 발생 가능
4. **시스템 가동**: Mac이 켜져 있어야 launchd 작업 실행됨
