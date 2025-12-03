# BucketsVision

NBA 경기 승부를 예측하는 머신러닝 기반 서비스입니다.

## 현재 모델: V5.4

| 항목 | 값 |
|------|-----|
| 알고리즘 | Logistic Regression (C=0.01, L2) |
| 피처 수 | 5개 |
| 전체 정확도 | **78.05%** |
| 고신뢰도(≥70%) 정확도 | **87.88%** |
| 저신뢰도(<70%) 정확도 | 71.43% |
| 확률 범위 | 8.2% ~ 94.8% |

## 시즌 정보

- **현재 시즌**: 2025-26 (season code: 2026)
- **학습 데이터**: 22-23, 23-24, 24-25 시즌 (3,643 경기)
- **검증 데이터**: 25-26 시즌 (246 경기, 진행중)

## 서비스 실행

```bash
# Streamlit 앱 실행
streamlit run app/main.py

# 또는
python3 -m streamlit run app/main.py
```

## 피처 (5개)

| 피처 | 설명 | 계수 |
|------|------|------|
| team_epm_diff | 팀 EPM 차이 (홈-원정) | +0.407 |
| top5_epm_diff | 상위 5인 EPM 평균 차이 | +0.361 |
| bench_strength_diff | 벤치 선수(6-10위) EPM 차이 | +0.259 |
| ft_rate_diff | Free Throw Rate 차이 | +0.066 |
| sos_diff | Strength of Schedule 차이 | +0.016 |

## 후행 지표: Injury Impact v1.0.0

부상 영향은 예측 후 적용되는 후행 조정(post-prediction adjustment)입니다.

| 항목 | 값 |
|------|-----|
| 알고리즘 | Performance-based (출전 vs 미출전 성과 비교) |
| 공식 | `prob_shift = player_epm × 0.02 × normalized_diff` |
| 조건 | EPM > 0, MPG ≥ 12, 출전율 > 1/3 |

## 데이터 소스

| 소스 | 용도 |
|------|------|
| DNT API | Team EPM, Player EPM, SOS |
| NBA Stats API | 경기 결과, 팀 스탯, 스케줄 |
| ESPN API | 부상/결장 정보 |
| Odds API | Pinnacle 배당 정보 |

## 프로젝트 구조

```
bucketsvision/
├── app/                          # Streamlit 앱
│   ├── main.py                   # 메인 엔트리포인트
│   ├── components/               # UI 컴포넌트
│   └── services/                 # 비즈니스 로직
│       ├── data_loader.py        # 데이터 로딩 및 피처 생성
│       └── predictor_v5.py       # V5.4 예측 서비스
├── bucketsvision_v4/
│   ├── models/                   # 학습된 모델
│   │   ├── v5_4_model.pkl
│   │   ├── v5_4_scaler.pkl
│   │   └── v5_4_feature_names.json
│   └── docs/                     # 모델 문서
├── config/                       # 설정
├── data/                         # 데이터
│   └── raw/dnt/season_epm/       # 시즌별 선수 EPM
├── scripts/                      # 유틸리티 스크립트
├── src/
│   ├── data_collection/          # API 클라이언트
│   ├── features/                 # 피처 엔지니어링
│   │   └── advanced_injury_impact.py
│   └── utils/                    # 유틸리티
└── tests/                        # 테스트
    └── test_v5_4_comprehensive.py
```

## 테스트

```bash
# 종합 검증 테스트
python3 tests/test_v5_4_comprehensive.py

# pytest 상세 테스트
python3 -m pytest tests/test_v5_4_comprehensive.py -v
```

## 개발 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 환경변수 설정
cp .env.example .env
# .env에 DNT_API_KEY, ODDS_API_KEY 설정
```

## 모델 히스토리

| 버전 | 알고리즘 | 피처 수 | 전체 정확도 | 비고 |
|------|----------|---------|-------------|------|
| V4.4 | Logistic Regression | 13 | 76.02% | |
| V5.2 | XGBoost | 11 | 70.8% | 확률 압축 문제 |
| V5.3 | Logistic Regression | 9 | 76.02% | |
| **V5.4** | **Logistic Regression** | **5** | **78.05%** | **현재 사용** |

## 라이선스

Private Project
