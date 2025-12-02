# 🏀 BucketsVision

NBA 경기 승부를 예측하는 머신러닝 기반 서비스입니다.

## 프로젝트 목표

- **Win Accuracy**: > 66% (승패 예측 정확도)
- **RMSE**: < 11.5 (점수차 예측 오차)

## 현재 성능 (2025-11-26)

### 25-26 시즌 검증 결과 (262경기)

| Model | RMSE | MAE | Win Acc | Within 5 | Within 10 |
|-------|------|-----|---------|----------|-----------|
| Ridge | 13.339 | 10.455 | **72.52%** | 31.68% | 56.11% |
| Ensemble | 13.547 | 10.495 | 68.70% | 31.68% | 56.49% |
| LightGBM | 13.620 | 10.597 | 69.08% | 30.53% | 58.40% |
| XGBoost | 14.017 | 10.765 | 67.56% | 32.82% | 58.02% |

- **Win Accuracy: 72.52%** ✓ (목표 달성)
- **RMSE: 13.339** ✗ (개선 필요)

## 서비스 실행

```bash
# Streamlit 앱 실행
streamlit run app/main.py
```

## 데이터

### 학습 데이터
- **22-23 시즌** (2023): 1,230 경기
- **23-24 시즌** (2024): 1,230 경기
- **24-25 시즌** (2025): 1,230 경기
- **총 학습 샘플**: 3,684 경기

### 검증 데이터
- **25-26 시즌** (2026, 진행중): 262 경기 (2025-10-21 ~ 2025-11-25)

### 데이터 소스
- **Dunks and Threes API**: Team EPM, Player EPM, SOS
- **NBA Stats API**: 경기 결과, 팀 스탯, Four Factors
- **ESPN API**: 부상/결장 정보

## 피처 (16개)

### EPM 기반 피처
| 피처 | 설명 |
|------|------|
| `team_epm_diff` | 팀 EPM 차이 (홈-원정) |
| `team_oepm_diff` | 팀 공격 EPM 차이 |
| `team_depm_diff` | 팀 수비 EPM 차이 |
| `team_epm_go_diff` | 팀 EPM (Game Optimized) 차이 |
| `team_oepm_go_diff` | 팀 공격 EPM (GO) 차이 |
| `team_depm_go_diff` | 팀 수비 EPM (GO) 차이 |
| `sos_diff` | Strength of Schedule 차이 |
| `sos_o_diff` | 공격 SOS 차이 |
| `sos_d_diff` | 수비 SOS 차이 |
| `team_epm_rk_diff` | 팀 EPM 순위 차이 |
| `team_oepm_rk_diff` | 공격 EPM 순위 차이 |
| `team_depm_rk_diff` | 수비 EPM 순위 차이 |
| `team_epm_z_diff` | 팀 EPM Z-score 차이 |
| `team_oepm_z_diff` | 공격 EPM Z-score 차이 |
| `team_depm_z_diff` | 수비 EPM Z-score 차이 |

## 모델

- **Ridge Regression**: L2 정규화 (최종 선택)
- 하이퍼파라미터: Optuna (TPE Sampler, 30 trials)
- CV: 5-Fold Time Series CV

## 프로젝트 구조

```
bucketsvision/
├── app/                      # Streamlit 앱
│   ├── main.py              # 메인 엔트리포인트
│   ├── components/          # UI 컴포넌트
│   └── services/            # 비즈니스 로직
├── config/
│   └── settings.py          # 설정 관리
├── data/
│   ├── raw/                 # 원본 데이터
│   │   ├── dnt/            # D&T API 데이터
│   │   └── nba_stats/      # NBA Stats API 데이터
│   └── models/             # 학습된 모델
│       └── final/          # 최종 모델
├── scripts/                 # 학습/분석 스크립트
├── src/
│   ├── data_collection/    # API 클라이언트
│   ├── features/           # 피처 엔지니어링
│   ├── models/             # 모델 구현
│   ├── prediction/         # 예측 및 조정
│   ├── evaluation/         # 평가 메트릭
│   └── utils/              # 유틸리티
└── notebooks/              # 분석 노트북
```

## 개발 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env에 DNT_API_KEY 설정
```

## 스크립트

```bash
# 데이터 수집
python scripts/collect_historical_data.py --seasons 2023 2024 2025 2026

# 모델 학습
python scripts/train_final_model.py --seasons 2023 2024 2025

# 하이퍼파라미터 최적화
python scripts/optimize_models.py --train-seasons 2023 2024 2025 --n-trials 50
```

## 라이선스

Private Project
