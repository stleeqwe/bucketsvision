# Phase 4: UI Layer 리팩토링 계획

## 현재 상태 분석

### main.py 구조 (991줄)

```
main.py
├── [1-26]    상수 정의 (B2B_AWAY_ONLY, B2B_HOME_ONLY, B2B_BOTH)
├── [28-79]   유틸리티 함수
│             ├── apply_b2b_correction()     # 미사용 (dead code)
│             ├── get_et_today()
│             ├── get_kst_now()              # 미사용 (dead code)
│             └── format_date_kst()
├── [81-150]  글로벌 설정
│             ├── 프로젝트 루트 설정
│             ├── 임포트
│             ├── st.set_page_config()
│             └── CSS 스타일 주입
├── [153-174] 서비스 팩토리
│             ├── get_prediction_service()
│             └── get_data_loader()
├── [176-216] 캐시 관리
│             ├── get_cache_date_key()
│             └── get_cache_info()
├── [219-406] Paper Betting 페이지 (~187줄)
│             ├── load_paper_betting_data()
│             └── render_paper_betting_page()
└── [411-991] main() 함수 (~580줄) ← 핵심 리팩토링 대상
              ├── [413-415]  헤더
              ├── [417-663]  사이드바 (~246줄)
              │              ├── 페이지 모드 선택
              │              ├── 팀 로스터 사이드바
              │              ├── 날짜 선택 UI
              │              ├── 모델 정보
              │              └── 캐시 상태
              ├── [665-708]  페이지 라우팅
              ├── [710-975]  예측 페이지 (~265줄)
              │              ├── 날짜 범위 계산
              │              ├── 데이터 로딩
              │              ├── 경기별 예측 루프
              │              └── 통계 요약
              └── [977-987]  푸터
```

### 문제점

| 문제 | 설명 | 영향 |
|------|------|------|
| Monster Function | main()이 580줄 | 유지보수 어려움 |
| 혼재된 책임 | UI + 로직 + 라우팅 혼합 | 테스트 불가 |
| 중복 코드 | 날짜 포맷팅, 통계 계산 반복 | 수정 시 여러 곳 변경 |
| Dead Code | apply_b2b_correction, get_kst_now 미사용 | 코드 복잡도 증가 |
| 하드코딩된 스타일 | CSS가 main.py에 직접 존재 | 테마 변경 어려움 |

---

## 리팩토링 목표

### 목표 구조

```
app/
├── main.py                          # 엔트리포인트 (~100줄)
├── theme.py                         # 테마 + CSS 통합
├── pages/                           # 페이지 렌더러 (신규)
│   ├── __init__.py
│   ├── predictions_page.py          # 예측 페이지
│   ├── paper_betting_page.py        # Paper Betting 페이지
│   └── team_roster_page.py          # 팀 로스터 페이지
├── components/                      # UI 컴포넌트
│   ├── sidebar/                     # 사이드바 컴포넌트 (신규)
│   │   ├── __init__.py
│   │   ├── date_picker.py           # 날짜 선택 UI
│   │   ├── model_info.py            # 모델 정보 표시
│   │   └── cache_status.py          # 캐시 상태 표시
│   ├── game_card_v2.py              # 기존
│   └── team_roster.py               # 기존
└── utils/                           # 유틸리티 (신규)
    ├── __init__.py
    ├── date_utils.py                # 날짜/시간 유틸리티
    └── streamlit_utils.py           # Streamlit 캐시 헬퍼
```

---

## 세부 작업 계획

### Step 1: 유틸리티 추출 (app/utils/)

**파일: app/utils/date_utils.py**
```python
"""날짜/시간 유틸리티."""
from datetime import date, datetime, timedelta
import pytz

def get_et_today() -> date:
    """미국 동부 시간 기준 오늘 날짜"""
    et = pytz.timezone('America/New_York')
    return datetime.now(et).date()

def format_date_kst(game_date: date) -> str:
    """경기 날짜를 한국 시간 기준으로 표시"""
    kst_date = game_date + timedelta(days=1)
    return kst_date.strftime('%Y년 %m월 %d일')

def get_cache_date_key() -> str:
    """ET 오전 5시 기준 캐시 날짜 키"""
    ...

def get_cache_info() -> dict:
    """캐시 정보 반환"""
    ...
```

**파일: app/utils/streamlit_utils.py**
```python
"""Streamlit 관련 유틸리티."""
import streamlit as st
from pathlib import Path

@st.cache_resource
def get_prediction_service():
    """V5.4 예측 서비스 (캐시)"""
    from app.services.predictor_v5 import V5PredictionService
    ...

@st.cache_resource
def get_data_loader(_cache_key: str):
    """데이터 로더 (캐시)"""
    from app.services.data_loader import DataLoader
    ...
```

### Step 2: 사이드바 컴포넌트 추출 (app/components/sidebar/)

**파일: app/components/sidebar/date_picker.py**
```python
"""날짜 선택 사이드바 컴포넌트."""
import streamlit as st
from datetime import date, timedelta
from dataclasses import dataclass
from typing import Tuple

@dataclass
class DateSelection:
    """날짜 선택 결과"""
    mode: str          # daily, weekly, monthly, season
    start_date: date
    end_date: date
    header_text: str

def render_date_picker(et_today: date) -> DateSelection:
    """
    날짜 선택 UI 렌더링.

    Returns:
        DateSelection 결과
    """
    ...
```

**파일: app/components/sidebar/model_info.py**
```python
"""모델 정보 사이드바 컴포넌트."""
import streamlit as st
from typing import Dict

def render_model_info(model_info: Dict) -> None:
    """모델 정보 표시"""
    st.subheader("모델 정보")
    st.metric("모델", model_info.get("model_version", "V5.4"))
    st.metric("피처 수", model_info.get("n_features", 5))
    ...
```

**파일: app/components/sidebar/cache_status.py**
```python
"""캐시 상태 사이드바 컴포넌트."""
import streamlit as st
from typing import Dict

def render_cache_status(cache_info: Dict) -> None:
    """캐시 상태 표시"""
    ...

def render_refresh_button() -> bool:
    """새로고침 버튼 렌더링. 클릭 시 True 반환"""
    ...
```

### Step 3: 페이지 렌더러 추출 (app/pages/)

**파일: app/pages/predictions_page.py**
```python
"""예측 페이지."""
import streamlit as st
from datetime import date
from typing import Dict, List

from app.services.prediction_pipeline import PredictionPipeline, GamePrediction
from app.components.game_card_v2 import render_game_card, render_day_summary
from app.components.sidebar.date_picker import DateSelection

def render_predictions_page(
    pipeline: PredictionPipeline,
    date_selection: DateSelection,
    team_epm: Dict[int, Dict]
) -> None:
    """
    예측 페이지 렌더링.

    PredictionPipeline을 사용하여 예측 로직 단순화.
    """
    st.subheader(date_selection.header_text)

    # 날짜 범위 경기 로딩
    all_predictions = _load_predictions(
        pipeline,
        date_selection.start_date,
        date_selection.end_date
    )

    if not all_predictions:
        render_no_games()
        return

    # 날짜별 렌더링
    _render_predictions_by_date(all_predictions, date_selection.mode)

    # 통계 요약
    _render_statistics_summary(all_predictions, date_selection.mode)

def _load_predictions(
    pipeline: PredictionPipeline,
    start_date: date,
    end_date: date
) -> Dict[date, List[GamePrediction]]:
    """날짜 범위 예측 로딩"""
    ...

def _render_predictions_by_date(
    predictions: Dict[date, List[GamePrediction]],
    date_mode: str
) -> None:
    """날짜별 예측 렌더링"""
    ...

def _render_statistics_summary(
    predictions: Dict[date, List[GamePrediction]],
    date_mode: str
) -> None:
    """통계 요약 렌더링"""
    ...
```

**파일: app/pages/paper_betting_page.py**
```python
"""Paper Betting 페이지."""
import streamlit as st
import json
from pathlib import Path

def render_paper_betting_page() -> None:
    """Paper Betting 대시보드 렌더링"""
    ...
    # 기존 render_paper_betting_page() 함수 이동
```

**파일: app/pages/team_roster_page.py**
```python
"""팀 로스터 페이지 래퍼."""
import streamlit as st
from app.components.team_roster import render_team_roster_page as _render

def render_team_roster_page(team_id: int, team_name: str, team_color: str) -> None:
    """팀 로스터 페이지 렌더링"""
    _render(team_id, team_name, team_color)
```

### Step 4: CSS 통합 (app/theme.py 확장)

**app/theme.py 확장:**
```python
"""BucketsVision 테마 및 스타일."""

COLORS = {
    # 기존 색상
    ...
}

# CSS 스타일 상수
MAIN_STYLES = """
<style>
.stApp {
    background-color: %(bg_primary)s;
}
.main-header {
    font-size: 3rem;
    font-weight: bold;
    ...
}
...
</style>
""" % COLORS

def inject_all_styles() -> None:
    """모든 CSS 스타일 주입"""
    import streamlit as st
    st.markdown(MAIN_STYLES, unsafe_allow_html=True)
    # game_card 스타일도 여기서 주입
```

### Step 5: main.py 리팩토링

**최종 main.py (~100줄):**
```python
"""
🏀 BucketsVision - NBA 승부 예측 서비스

Streamlit 메인 엔트리포인트
"""
import streamlit as st
from pathlib import Path

from app.theme import inject_all_styles
from app.utils.date_utils import get_et_today, get_cache_date_key, get_cache_info
from app.utils.streamlit_utils import get_prediction_service, get_data_loader
from app.components.sidebar.date_picker import render_date_picker
from app.components.sidebar.model_info import render_model_info
from app.components.sidebar.cache_status import render_cache_status, render_refresh_button
from app.pages.predictions_page import render_predictions_page
from app.pages.paper_betting_page import render_paper_betting_page
from app.pages.team_roster_page import render_team_roster_page

# 프로젝트 경로
PROJECT_ROOT = Path(__file__).parent.parent

# 페이지 설정
st.set_page_config(
    page_title="BucketsVision",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 주입
inject_all_styles()


def main():
    """메인 함수"""
    # 헤더
    _render_header()

    # 사이드바 & 페이지 모드
    page_mode = _render_sidebar()

    # 페이지 라우팅
    if page_mode == "paper_betting":
        render_paper_betting_page()
    elif page_mode == "team_roster":
        _handle_team_roster_page()
    else:
        _handle_predictions_page()

    # 푸터
    _render_footer()


def _render_header():
    st.markdown('<div class="main-header">🏀 BucketsVision</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI 기반 NBA 승부 예측 | V5.4</div>', unsafe_allow_html=True)


def _render_sidebar() -> str:
    """사이드바 렌더링, 페이지 모드 반환"""
    with st.sidebar:
        st.header("메뉴")
        page_mode = st.radio(
            "페이지 선택",
            options=["predictions", "paper_betting", "team_roster"],
            format_func=_format_page_mode,
            label_visibility="collapsed"
        )

        if page_mode == "predictions":
            st.markdown("---")
            date_selection = render_date_picker(get_et_today())
            st.session_state.date_selection = date_selection

            st.markdown("---")
            predictor = get_prediction_service()
            render_model_info(predictor.get_model_info())

            st.markdown("---")
            render_cache_status(get_cache_info())
            if render_refresh_button():
                st.cache_data.clear()
                st.cache_resource.clear()
                st.rerun()

        return page_mode


def _handle_predictions_page():
    """예측 페이지 처리"""
    et_today = get_et_today()
    cache_key = get_cache_date_key()

    pipeline = PredictionPipeline(
        data_dir=PROJECT_ROOT / "data",
        model_dir=PROJECT_ROOT / "bucketsvision_v4" / "models"
    )

    team_epm = pipeline.loader.load_team_epm(et_today)
    date_selection = st.session_state.get("date_selection")

    render_predictions_page(pipeline, date_selection, team_epm)


def _handle_team_roster_page():
    """팀 로스터 페이지 처리"""
    ...


def _render_footer():
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.8rem;">'
        '⚠️ 본 예측은 참고용입니다.'
        '</div>',
        unsafe_allow_html=True
    )


def _format_page_mode(x):
    return {"predictions": "🏀 경기 예측", "team_roster": "👥 팀 로스터", "paper_betting": "💰 Paper Betting"}.get(x, x)


if __name__ == "__main__":
    main()
```

---

## 마이그레이션 순서

### 단계별 진행 (Green-to-Green)

```
Step 1: 유틸리티 추출
├── app/utils/date_utils.py 생성
├── app/utils/streamlit_utils.py 생성
├── main.py에서 import 변경
└── 테스트 실행

Step 2: 사이드바 컴포넌트 추출
├── app/components/sidebar/date_picker.py 생성
├── app/components/sidebar/model_info.py 생성
├── app/components/sidebar/cache_status.py 생성
├── main.py에서 사용
└── 테스트 실행

Step 3: 페이지 렌더러 추출
├── app/pages/paper_betting_page.py 생성
├── app/pages/team_roster_page.py 생성
├── app/pages/predictions_page.py 생성 (PredictionPipeline 사용)
├── main.py에서 import 및 호출
└── 테스트 실행

Step 4: Dead Code 제거
├── apply_b2b_correction() 제거
├── get_kst_now() 제거
├── B2B 상수 제거 (미사용 시)
└── 테스트 실행

Step 5: CSS 통합
├── theme.py 확장
├── main.py에서 스타일 코드 제거
└── 테스트 실행
```

---

## 예상 결과

### 코드 라인 수 변화

| 파일 | 이전 | 이후 |
|------|------|------|
| main.py | 991줄 | ~100줄 |
| app/utils/date_utils.py | - | ~60줄 |
| app/utils/streamlit_utils.py | - | ~30줄 |
| app/components/sidebar/ | - | ~150줄 |
| app/pages/predictions_page.py | - | ~200줄 |
| app/pages/paper_betting_page.py | - | ~180줄 |
| **총 신규 코드** | - | ~620줄 |

### 개선 효과

| 항목 | 이전 | 이후 |
|------|------|------|
| main() 함수 크기 | 580줄 | ~50줄 |
| 단일 책임 | ❌ 혼재 | ✅ 분리 |
| 테스트 가능성 | ❌ 어려움 | ✅ 용이 |
| 재사용성 | ❌ 없음 | ✅ 컴포넌트 |
| 유지보수성 | ❌ 어려움 | ✅ 용이 |

---

## 위험 관리

### 잠재적 위험

1. **Streamlit Session State**
   - 위험: 컴포넌트 분리 시 상태 공유 문제
   - 대응: st.session_state를 명시적으로 전달

2. **CSS 스코프**
   - 위험: 스타일 충돌
   - 대응: 클래스명 네임스페이스 사용

3. **임포트 순환**
   - 위험: 모듈 간 순환 참조
   - 대응: 의존성 방향 명확히 정의

### 롤백 전략

- 각 Step 완료 후 git commit
- 테스트 실패 시 이전 commit으로 롤백
- 기존 main.py를 main_backup.py로 보관

---

## 검증 체크리스트

### 기능 검증

- [ ] 예측 페이지 정상 렌더링
- [ ] Paper Betting 페이지 정상 렌더링
- [ ] 팀 로스터 페이지 정상 렌더링
- [ ] 날짜 선택 (일별/주간/월간/시즌) 동작
- [ ] 캐시 새로고침 동작
- [ ] 부상 정보 표시
- [ ] 배당 정보 표시

### 성능 검증

- [ ] 페이지 로딩 시간 동일 유지
- [ ] 메모리 사용량 동일 유지

### 코드 품질

- [ ] 모든 기존 테스트 통과
- [ ] 새 컴포넌트 단위 테스트 추가
- [ ] 린트 경고 없음
