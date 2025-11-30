# BucketsVision - Claude Code 가이드

## 핵심 정보

### 현재 시즌 정보 (중요!)
- **현재 진행 시즌**: 2025-26 시즌 (25-26)
- **시즌 시작일**: 2025년 10월
- **시즌 코드**: 2026 (NBA API 기준 종료 연도)
- **예시**: 2025년 11월 경기 → 2026 시즌 데이터 사용

### 시즌 계산 규칙
```python
# 10월 이후는 다음 해 시즌
def get_season_from_date(game_date):
    if game_date.month >= 10:
        return game_date.year + 1
    return game_date.year

# 2025-11-28 → 2026 시즌
# 2026-01-15 → 2026 시즌
# 2026-06-20 → 2026 시즌 (플레이오프)
```

### 절대 하지 말아야 할 실수
1. 2024년 시즌 데이터를 현재 시즌으로 착각하지 말 것
2. `date(2024, ...)` 대신 `date(2025, ...)`를 사용할 것
3. 시즌 테스트 시 반드시 현재 날짜 기준으로 확인

## 프로젝트 구조

### 핵심 파일
- `app/main.py` - Streamlit 메인 앱
- `app/services/data_loader.py` - 데이터 로딩 및 피처 생성
- `app/services/predictor_v4.py` - V4.2 예측 모델
- `src/utils/helpers.py` - 유틸리티 함수 (시즌 계산 등)

### 데이터 소스
- **DNT API**: Team EPM, Player EPM, SOS
- **NBA Stats API**: 경기 결과, 팀 스탯, 스케줄
- **ESPN API**: 부상/결장 정보

### V4.2 모델 피처 (11개)
| 카테고리 | 피처 | 설명 |
|----------|------|------|
| EPM | team_epm_diff | 팀 EPM 차이 |
| EPM | team_oepm_diff | 공격 EPM 차이 |
| EPM | team_depm_diff | 수비 EPM 차이 |
| EPM | sos_diff | SOS 차이 |
| Four Factors | efg_pct_diff | eFG% 차이 |
| Four Factors | ft_rate_diff | FT Rate 차이 |
| 모멘텀 | last5_win_pct_diff | 최근 5경기 승률 차이 |
| 모멘텀 | streak_diff | 연승/연패 차이 |
| 모멘텀 | margin_ewma_diff | 점수차 EWMA 차이 (클리핑 적용) |
| 컨텍스트 | away_road_strength | 원정팀 원정 성적 |
| 리바운드 | orb_diff | 공격 리바운드 차이 |

## 서비스 실행

```bash
# Streamlit 앱 실행
streamlit run app/main.py
# 또는
python3 -m streamlit run app/main.py
```

## 주의사항

### 모델 관련
1. margin_ewma_diff는 ±30으로 클리핑됨 (이상치 방지)
2. 모델 재학습 없이 피처 스케일 변경 금지
3. Scaler와 모델은 반드시 함께 사용

### 시간대 관련
- 경기 날짜: 미국 동부 시간 (ET) 기준
- 한국 시간 표시: ET + 1일 (KST)
- 예: ET 11/28 경기 → KST 11/29 표시

### API 캐시
- 오늘/내일 경기: 캐시 미사용 (실시간 반영)
- 과거 경기: 24시간 캐시

## 개발 시 체크리스트
- [ ] 시즌 연도가 올바른지 확인 (2025-26 시즌 = 2026)
- [ ] 날짜 테스트 시 2025년 기준 사용
- [ ] 모델 피처 순서가 feature_names.json과 일치하는지 확인
- [ ] 스케일링 전/후 값 범위 확인
