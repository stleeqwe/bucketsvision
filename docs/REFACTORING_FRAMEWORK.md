# BucketsVision 코드 리팩토링 프레임워크

**버전**: 1.0.0
**작성일**: 2025-12-03
**대상 시스템**: BucketsVision V5.4 NBA 예측 시스템

---

## 1. 개요

### 1.1 목적
본 문서는 BucketsVision 프로젝트의 코드 품질 개선을 위한 체계적인 리팩토링 프레임워크를 제시합니다.
코드 복잡도 감소, 중복 제거, 아키텍처 개선을 통해 유지보수성과 확장성을 향상시킵니다.

### 1.2 현재 상태 분석 요약

| 파일 | 라인 수 | 메서드 수 | 주요 이슈 |
|------|--------|----------|----------|
| `data_loader.py` | 1,381 | 42 | God Class, 높은 복잡도 |
| `main.py` | 1,033 | 15+ | 모놀리식 구조, 깊은 중첩 |
| `advanced_injury_impact.py` | 622 | 18 | 적절한 구조 |
| `predictor_v5.py` | 219 | 6 | 중복 로직 |

### 1.3 리팩토링 원칙

1. **단일 책임 원칙 (SRP)**: 각 클래스/함수는 하나의 책임만 가짐
2. **DRY (Don't Repeat Yourself)**: 중복 코드 제거
3. **낮은 결합도**: 모듈 간 의존성 최소화
4. **높은 응집도**: 관련 기능을 하나의 모듈에 집중
5. **테스트 가능성**: 모든 리팩토링은 테스트로 검증

---

## 2. 코드 품질 이슈 분석

### 2.1 심각도별 이슈 목록

#### CRITICAL (우선 처리)

| 이슈 | 파일 | 라인 | 설명 |
|------|------|------|------|
| God Class | data_loader.py | 전체 | 42개 메서드, 5가지 책임 혼재 |
| Monster Function | main.py | 453-1033 | 581줄 단일 함수 |
| 높은 순환 복잡도 | data_loader.py | 339-557 | `get_games()` 복잡도 15+ |

#### HIGH (조기 해결)

| 이슈 | 파일 | 라인 | 설명 |
|------|------|------|------|
| 피처 빌더 중복 | data_loader.py | 746-1139 | 4개 유사 메서드 |
| 부상 조정 중복 | main.py, predictor_v5.py | 31-67, 111-149 | 동일 로직 2곳 |
| 깊은 중첩 | main.py | 793-970 | 5단계 중첩 |

#### MEDIUM (점진적 개선)

| 이슈 | 파일 | 라인 | 설명 |
|------|------|------|------|
| 긴 파라미터 목록 | game_card_v2.py | 렌더 함수 | 20+ 파라미터 |
| Primitive Obsession | data_loader.py | 1315-1340 | Dict 대신 타입 사용 |
| Feature Envy | main.py | 854-964 | 외부 구조 직접 접근 |

---

## 3. 목표 아키텍처

### 3.1 현재 구조 vs 목표 구조

```
현재 구조                              목표 구조
─────────────────────────────────────────────────────────────────
app/                                   app/
├── main.py (1033 lines)               ├── main.py (~400 lines)
├── services/                          ├── services/
│   ├── data_loader.py (1381 lines)    │   ├── data_loader.py (~300 lines)
│   ├── predictor_v5.py                │   ├── predictor_v5.py
│   └── dnt_api.py                     │   ├── prediction_pipeline.py (NEW)
└── components/                        │   ├── injury_adjuster.py (NEW)
    ├── game_card_v2.py                │   └── dnt_api.py
    └── team_roster.py                 ├── feature_builders/ (NEW)
                                       │   ├── base_builder.py
                                       │   ├── v4_builder.py
                                       │   ├── v5_2_builder.py
                                       │   └── v5_4_builder.py
                                       ├── calculators/ (NEW)
                                       │   ├── stat_calculator.py
                                       │   └── log_processor.py
                                       └── components/
                                           ├── game_card_v2.py
                                           ├── game_renderer.py (NEW)
                                           └── team_roster.py
```

### 3.2 모듈 책임 분리

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                   │
│  [책임: UI 오케스트레이션, Streamlit 상태 관리]                    │
└───────────────────────────┬─────────────────────────────────────┘
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
┌───────────────────────┐       ┌───────────────────────┐
│   PredictionPipeline  │       │   GameRenderer        │
│   [예측 파이프라인]     │       │   [게임 카드 렌더링]   │
└───────────┬───────────┘       └───────────────────────┘
            │
  ┌─────────┼─────────┐
  ▼         ▼         ▼
┌─────┐ ┌────────┐ ┌────────────┐
│Data │ │Feature │ │Injury      │
│Loader│ │Builder │ │Adjuster   │
└─────┘ └────────┘ └────────────┘
```

---

## 4. 리팩토링 패턴 및 기법

### 4.1 Extract Class (클래스 추출)

**적용 대상**: `DataLoader` God Class

**Before:**
```python
class DataLoader:
    def load_team_epm(self): ...
    def load_player_epm(self): ...
    def build_v4_features(self): ...
    def build_v5_4_features(self): ...
    def _calc_efg(self): ...
    def _calc_ft_rate(self): ...
    def get_injury_summary(self): ...
    def calculate_injury_impact(self): ...
    # ... 42개 메서드
```

**After:**
```python
class DataLoader:
    """데이터 로딩만 담당"""
    def load_team_epm(self): ...
    def load_player_epm(self): ...
    def load_team_game_logs(self): ...

class FeatureBuilder:
    """피처 빌딩 담당"""
    def build(self, home_id, away_id, team_epm, date): ...

class StatCalculator:
    """통계 계산 담당"""
    def calc_efg(self, games): ...
    def calc_ft_rate(self, games): ...

class InjuryAnalyzer:
    """부상 분석 담당"""
    def get_summary(self, team_abbr, date, team_epm): ...
```

### 4.2 Extract Method (메서드 추출)

**적용 대상**: `get_games()` 메서드 (219줄)

**Before:**
```python
def get_games(self, game_date: date) -> List[Dict]:
    # ... 219줄의 복잡한 로직
    # 게임 상태 판단
    if raw_status == 3 or "Final" in status_text:
        game_status = 3
    elif live_period > 0 or raw_status == 2:
        game_status = 2
    else:
        game_status = 1

    # 점수 추출
    game_scores = live_scores.get(game_id, {})
    home_score = game_scores.get(home_team_id)
    away_score = game_scores.get(away_team_id)
    # ...
```

**After:**
```python
def get_games(self, game_date: date) -> List[Dict]:
    scoreboard = self._fetch_scoreboard(game_date)
    results = self._fetch_game_results(game_date)

    games = []
    for row in scoreboard:
        game_status = self._determine_game_status(row, results)
        scores = self._extract_scores(row, results)
        b2b = self._check_b2b(row, game_date)

        games.append(self._build_game_dict(row, game_status, scores, b2b))

    return games

def _determine_game_status(self, row, results) -> int:
    """게임 상태 판단 로직 분리"""
    raw_status = int(row.get("GAME_STATUS_ID", 1))
    status_text = str(row.get("GAME_STATUS_TEXT", ""))

    if raw_status == 3 or "Final" in status_text:
        return 3
    elif row.get("LIVE_PERIOD", 0) > 0 or raw_status == 2:
        return 2
    return 1

def _extract_scores(self, row, results) -> Tuple[Optional[int], Optional[int]]:
    """점수 추출 로직 분리"""
    # ...
```

### 4.3 Template Method Pattern (템플릿 메서드 패턴)

**적용 대상**: 피처 빌더들 (V4, V4.3, V5.2, V5.4)

**Before:**
```python
# 4개의 유사한 메서드
def build_v4_features(self, home_id, away_id, team_epm, date):
    logs = self.load_team_game_logs(date)
    home_stats = self._compute_team_stats(home_id, logs, date)
    away_stats = self._compute_team_stats(away_id, logs, date)
    # ... 11개 피처 계산

def build_v5_4_features(self, home_id, away_id, team_epm, date):
    logs = self.load_team_game_logs(date)
    # ... 5개 피처 계산
```

**After:**
```python
from abc import ABC, abstractmethod

class BaseFeatureBuilder(ABC):
    """피처 빌더 기본 클래스"""

    def __init__(self, loader: 'DataLoader'):
        self.loader = loader
        self.log_processor = LogProcessor()

    def build(self, home_id: int, away_id: int,
              team_epm: Dict, target_date: date) -> Dict[str, float]:
        """템플릿 메서드"""
        logs = self.loader.load_team_game_logs(target_date)
        home_logs = self.log_processor.filter_team_logs(logs, home_id, target_date)
        away_logs = self.log_processor.filter_team_logs(logs, away_id, target_date)

        return self._compute_features(
            home_id, away_id, team_epm, home_logs, away_logs, target_date
        )

    @abstractmethod
    def _compute_features(self, home_id, away_id, team_epm,
                         home_logs, away_logs, target_date) -> Dict[str, float]:
        """하위 클래스에서 구현"""
        pass

class V54FeatureBuilder(BaseFeatureBuilder):
    """V5.4 피처 빌더"""

    FEATURES = ['team_epm_diff', 'sos_diff', 'bench_strength_diff',
                'top5_epm_diff', 'ft_rate_diff']

    def _compute_features(self, home_id, away_id, team_epm,
                         home_logs, away_logs, target_date) -> Dict[str, float]:
        season = get_season_from_date(target_date)
        home_recent = home_logs.head(10)
        away_recent = away_logs.head(10)

        return {
            'team_epm_diff': self._get_epm_diff(home_id, away_id, team_epm),
            'sos_diff': self._get_sos_diff(home_id, away_id, team_epm),
            'bench_strength_diff': self._get_bench_diff(home_id, away_id, season),
            'top5_epm_diff': self._get_top5_diff(home_id, away_id, season),
            'ft_rate_diff': self._get_ft_rate_diff(home_recent, away_recent),
        }
```

### 4.4 Replace Primitive with Object (원시 타입을 객체로 대체)

**적용 대상**: 부상 정보 딕셔너리

**Before:**
```python
out_details.append({
    "name": inj.player_name,
    "status": "Out",
    "detail": inj.detail,
    "epm": round(result.epm, 2),
    "mpg": round(result.mpg, 1),
    "prob_shift": round(result.prob_shift * 100, 1),
    "played_games": result.played_games,
    "missed_games": result.missed_games,
})
```

**After:**
```python
@dataclass
class PlayerInjuryDetail:
    """선수 부상 상세 정보"""
    name: str
    status: str  # "Out" | "GTD"
    detail: str
    epm: float
    mpg: float
    prob_shift_pct: float
    played_games: int
    missed_games: int
    schedule_diff: float = 0.0
    skip_reason: Optional[str] = None

    @classmethod
    def from_result(cls, injury, result, status: str) -> 'PlayerInjuryDetail':
        """팩토리 메서드"""
        return cls(
            name=injury.player_name,
            status=status,
            detail=injury.detail,
            epm=round(result.epm, 2),
            mpg=round(result.mpg, 1),
            prob_shift_pct=round(result.prob_shift * 100, 1),
            played_games=result.played_games,
            missed_games=result.missed_games,
            schedule_diff=round(result.schedule_diff, 3),
        )

    def to_dict(self) -> Dict:
        """UI 렌더링용 딕셔너리 변환"""
        return asdict(self)
```

### 4.5 Introduce Parameter Object (파라미터 객체 도입)

**적용 대상**: `render_game_card()` 함수 (20+ 파라미터)

**Before:**
```python
def render_game_card(
    home_team: str,
    away_team: str,
    home_name: str,
    away_name: str,
    home_color: str,
    away_color: str,
    game_time: str,
    predicted_margin: float,
    home_win_prob: float,
    game_status: int,
    home_score: Optional[int],
    away_score: Optional[int],
    home_b2b: bool,
    away_b2b: bool,
    hide_result: bool,
    odds_info: Optional[Dict],
    game_id: str,
    enable_custom_input: bool = False,
    home_injury_summary: Optional[Dict] = None,
    away_injury_summary: Optional[Dict] = None,
) -> None:
```

**After:**
```python
@dataclass
class TeamInfo:
    """팀 정보"""
    abbr: str
    name: str
    color: str
    score: Optional[int] = None
    is_b2b: bool = False
    injury_summary: Optional[Dict] = None

@dataclass
class GameCardData:
    """게임 카드 렌더링 데이터"""
    game_id: str
    game_time: str
    game_status: int
    home: TeamInfo
    away: TeamInfo
    predicted_margin: float
    home_win_prob: float
    odds_info: Optional[Dict] = None
    hide_result: bool = False
    enable_custom_input: bool = False

def render_game_card(data: GameCardData) -> None:
    """단순화된 시그니처"""
    st.markdown(f"### {data.away.abbr} @ {data.home.abbr}")
    st.metric("Home Win Prob", f"{data.home_win_prob:.1%}")
    # ...
```

### 4.6 Facade Pattern (파사드 패턴)

**적용 대상**: 예측 파이프라인

**Before:**
```python
# main.py에서 직접 호출
features = loader.build_v5_4_features(home_id, away_id, team_epm, game_date)
base_prob = predictor.predict_proba(features)
home_summary = loader.get_injury_summary(home_abbr, game_date, team_epm)
away_summary = loader.get_injury_summary(away_abbr, game_date, team_epm)
home_shift = home_summary.get("total_prob_shift", 0.0)
away_shift = away_summary.get("total_prob_shift", 0.0)
final_prob = predictor.apply_injury_adjustment(base_prob, home_shift, away_shift)
```

**After:**
```python
# app/services/prediction_pipeline.py
@dataclass
class PredictionResult:
    """예측 결과"""
    base_prob: float
    adjusted_prob: float
    home_injury_shift: float
    away_injury_shift: float
    features: Dict[str, float]
    home_injuries: List[PlayerInjuryDetail]
    away_injuries: List[PlayerInjuryDetail]

class PredictionPipeline:
    """예측 파이프라인 파사드"""

    def __init__(self, loader: DataLoader, predictor: V5PredictionService):
        self.loader = loader
        self.predictor = predictor
        self.feature_builder = V54FeatureBuilder(loader)
        self.injury_adjuster = InjuryAdjuster()

    def predict_game(self, game: Dict, game_date: date,
                    team_epm: Dict) -> PredictionResult:
        """단일 메서드로 전체 예측 수행"""
        home_id = game['home_team_id']
        away_id = game['away_team_id']
        home_abbr = self.loader.get_team_info(home_id)['abbr']
        away_abbr = self.loader.get_team_info(away_id)['abbr']

        # 1. 피처 생성
        features = self.feature_builder.build(home_id, away_id, team_epm, game_date)

        # 2. 기본 예측
        base_prob = self.predictor.predict_proba(features)

        # 3. 부상 정보
        home_summary = self.loader.get_injury_summary(home_abbr, game_date, team_epm)
        away_summary = self.loader.get_injury_summary(away_abbr, game_date, team_epm)

        # 4. 부상 조정
        adjusted_prob = self.injury_adjuster.apply(
            base_prob,
            home_summary.get("total_prob_shift", 0.0),
            away_summary.get("total_prob_shift", 0.0)
        )

        return PredictionResult(
            base_prob=base_prob,
            adjusted_prob=adjusted_prob,
            home_injury_shift=home_summary.get("total_prob_shift", 0.0),
            away_injury_shift=away_summary.get("total_prob_shift", 0.0),
            features=features,
            home_injuries=home_summary.get("out_players", []),
            away_injuries=away_summary.get("out_players", []),
        )

# main.py에서 사용
pipeline = PredictionPipeline(loader, predictor)
result = pipeline.predict_game(game, game_date, team_epm)
```

---

## 5. 단계별 리팩토링 계획

### Phase 1: Foundation (저위험) - 1주차

#### 1.1 유틸리티 클래스 추출

**작업 내용:**
```python
# app/services/utils/log_processor.py
class LogProcessor:
    @staticmethod
    def safe_diff(h_val, a_val, default=0) -> float:
        """안전한 차이 계산"""
        h = h_val if h_val is not None else default
        a = a_val if a_val is not None else default
        return float(h - a)

    @staticmethod
    def filter_team_logs(logs: pd.DataFrame, team_id: int,
                        before_date: date) -> pd.DataFrame:
        """팀 로그 필터링"""
        if logs.empty:
            return pd.DataFrame()

        team_logs = logs[logs['team_id'] == team_id].copy()
        team_logs['game_date'] = pd.to_datetime(team_logs['game_date'])
        team_logs = team_logs[team_logs['game_date'] < pd.to_datetime(before_date)]
        return team_logs.sort_values('game_date', ascending=False)

    @staticmethod
    def get_recent_games(logs: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """최근 경기 추출"""
        return logs.head(window) if not logs.empty else pd.DataFrame()
```

**테스트:**
```python
def test_safe_diff():
    assert LogProcessor.safe_diff(5, 3) == 2
    assert LogProcessor.safe_diff(None, 3, default=0) == -3
    assert LogProcessor.safe_diff(5, None, default=0) == 5
```

#### 1.2 InjuryAdjuster 서비스 추출

**작업 내용:**
```python
# app/services/injury_adjuster.py
class InjuryAdjuster:
    """부상 조정 서비스 (Single Source of Truth)"""

    MIN_PROB = 0.01
    MAX_PROB = 0.99

    @staticmethod
    def apply(base_prob: float, home_shift: float, away_shift: float) -> float:
        """
        부상 조정 적용.

        Args:
            base_prob: 기본 예측 확률
            home_shift: 홈팀 부상 영향 (%, 양수)
            away_shift: 원정팀 부상 영향 (%, 양수)

        Returns:
            조정된 확률
        """
        home_shift = max(home_shift, 0) / 100.0
        away_shift = max(away_shift, 0) / 100.0
        net_shift = away_shift - home_shift

        adjusted = base_prob + net_shift
        return max(InjuryAdjuster.MIN_PROB,
                  min(InjuryAdjuster.MAX_PROB, adjusted))
```

**마이그레이션:**
1. `main.py`의 `apply_injury_correction()` 함수 삭제
2. `predictor_v5.py`의 `apply_injury_adjustment()` 메서드를 `InjuryAdjuster` 위임으로 변경
3. 모든 호출부 업데이트

#### 1.3 데이터 클래스 정의

**작업 내용:**
```python
# app/models/data_types.py
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

@dataclass
class TeamInfo:
    """팀 정보"""
    abbr: str
    name: str
    color: str
    team_id: int

@dataclass
class PlayerInjuryDetail:
    """선수 부상 상세"""
    name: str
    status: str
    detail: str
    epm: float
    mpg: float
    prob_shift_pct: float
    played_games: int
    missed_games: int
    schedule_diff: float = 0.0
    skip_reason: Optional[str] = None

@dataclass
class InjurySummary:
    """팀 부상 요약"""
    out_players: List[PlayerInjuryDetail]
    gtd_players: List[PlayerInjuryDetail]
    total_prob_shift: float

@dataclass
class PredictionResult:
    """예측 결과"""
    base_prob: float
    adjusted_prob: float
    features: Dict[str, float]
    home_injury: InjurySummary
    away_injury: InjurySummary
```

### Phase 2: Core Refactoring (중위험) - 2-3주차

#### 2.1 피처 빌더 추출

**디렉토리 구조:**
```
app/services/feature_builders/
├── __init__.py
├── base_builder.py
├── stat_calculator.py
├── v4_builder.py
├── v4_3_builder.py
├── v5_2_builder.py
└── v5_4_builder.py
```

**기본 클래스:**
```python
# app/services/feature_builders/base_builder.py
from abc import ABC, abstractmethod

class BaseFeatureBuilder(ABC):
    """피처 빌더 기본 클래스"""

    def __init__(self, loader, stat_calc: 'StatCalculator'):
        self.loader = loader
        self.stat_calc = stat_calc
        self.log_processor = LogProcessor()

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """피처 이름 목록"""
        pass

    def build(self, home_id: int, away_id: int,
              team_epm: Dict, target_date: date) -> Dict[str, float]:
        """템플릿 메서드"""
        logs = self.loader.load_team_game_logs(target_date)
        home_logs = self.log_processor.filter_team_logs(logs, home_id, target_date)
        away_logs = self.log_processor.filter_team_logs(logs, away_id, target_date)

        features = self._compute_features(
            home_id, away_id, team_epm, home_logs, away_logs, target_date
        )

        # 검증
        assert set(features.keys()) == set(self.feature_names)
        return features

    @abstractmethod
    def _compute_features(self, home_id, away_id, team_epm,
                         home_logs, away_logs, target_date) -> Dict[str, float]:
        pass
```

**V5.4 구현:**
```python
# app/services/feature_builders/v5_4_builder.py
class V54FeatureBuilder(BaseFeatureBuilder):
    """V5.4 피처 빌더 (5개 피처)"""

    @property
    def feature_names(self) -> List[str]:
        return ['team_epm_diff', 'sos_diff', 'bench_strength_diff',
                'top5_epm_diff', 'ft_rate_diff']

    def _compute_features(self, home_id, away_id, team_epm,
                         home_logs, away_logs, target_date) -> Dict[str, float]:
        season = get_season_from_date(target_date)
        home_recent = self.log_processor.get_recent_games(home_logs)
        away_recent = self.log_processor.get_recent_games(away_logs)

        home_epm = team_epm.get(home_id, {})
        away_epm = team_epm.get(away_id, {})

        return {
            'team_epm_diff': self.log_processor.safe_diff(
                home_epm.get('team_epm'), away_epm.get('team_epm')
            ),
            'sos_diff': self.log_processor.safe_diff(
                home_epm.get('sos'), away_epm.get('sos')
            ),
            'bench_strength_diff': (
                self.stat_calc.calc_bench_strength(home_id, season) -
                self.stat_calc.calc_bench_strength(away_id, season)
            ),
            'top5_epm_diff': (
                self.stat_calc.calc_top5_epm(home_id, season) -
                self.stat_calc.calc_top5_epm(away_id, season)
            ),
            'ft_rate_diff': (
                self.stat_calc.calc_ft_rate(home_recent) -
                self.stat_calc.calc_ft_rate(away_recent)
            ),
        }
```

#### 2.2 get_games() 메서드 분해

**Before:** 219줄 단일 메서드
**After:** 7개 작은 메서드

```python
class DataLoader:
    def get_games(self, game_date: date) -> List[Dict]:
        """경기 스케줄 조회 (리팩토링 후)"""
        scoreboard = self._fetch_scoreboard(game_date)
        if scoreboard.empty:
            return []

        game_results = self._fetch_game_results(game_date)
        live_scores = self._extract_live_scores(scoreboard)
        team_game_dates = self._build_team_game_dates(game_results)

        games = []
        for row in scoreboard.itertuples():
            game = self._process_game_row(
                row, game_date, game_results, live_scores, team_game_dates
            )
            games.append(game)

        return sorted(games, key=lambda g: g['game_id'])

    def _fetch_scoreboard(self, game_date: date) -> pd.DataFrame:
        """스코어보드 API 호출"""
        # ...

    def _fetch_game_results(self, game_date: date) -> Dict:
        """LeagueGameFinder 결과 조회"""
        # ...

    def _determine_game_status(self, row, game_results) -> int:
        """경기 상태 판단"""
        raw_status = int(row.GAME_STATUS_ID)
        status_text = str(row.GAME_STATUS_TEXT)

        if raw_status == 3 or "Final" in status_text:
            return 3
        elif row.LIVE_PERIOD > 0 or raw_status == 2:
            return 2
        return 1

    def _extract_scores(self, row, live_scores, game_results) -> Tuple[int, int]:
        """점수 추출"""
        # ...
```

### Phase 3: Architecture (고위험) - 4-5주차

#### 3.1 PredictionPipeline 서비스 생성

```python
# app/services/prediction_pipeline.py
class PredictionPipeline:
    """예측 파이프라인 오케스트레이터"""

    def __init__(self, loader: DataLoader, predictor: V5PredictionService):
        self.loader = loader
        self.predictor = predictor
        self.feature_builder = V54FeatureBuilder(loader, StatCalculator(loader))
        self.injury_adjuster = InjuryAdjuster()

    def predict_game(self, game: Dict, game_date: date,
                    team_epm: Dict) -> PredictionResult:
        """단일 경기 예측"""
        home_id = game['home_team_id']
        away_id = game['away_team_id']

        # 피처 생성
        features = self.feature_builder.build(home_id, away_id, team_epm, game_date)

        # 기본 예측
        base_prob = self.predictor.predict_proba(features)

        # 부상 정보 (예정된 경기만)
        home_injury = self._get_injury_summary(game, 'home', game_date, team_epm)
        away_injury = self._get_injury_summary(game, 'away', game_date, team_epm)

        # 최종 확률
        adjusted_prob = self.injury_adjuster.apply(
            base_prob,
            home_injury.total_prob_shift,
            away_injury.total_prob_shift
        )

        return PredictionResult(
            base_prob=base_prob,
            adjusted_prob=adjusted_prob,
            features=features,
            home_injury=home_injury,
            away_injury=away_injury,
        )

    def predict_all_games(self, game_date: date) -> List[Tuple[Dict, PredictionResult]]:
        """날짜별 전체 경기 예측"""
        games = self.loader.get_games(game_date)
        team_epm = self.loader.load_team_epm(game_date)

        results = []
        for game in games:
            if game['game_status'] != 3:  # 종료 경기 제외
                result = self.predict_game(game, game_date, team_epm)
                results.append((game, result))

        return results
```

#### 3.2 DataLoader 슬림화

```python
# 리팩토링 후 DataLoader (약 300줄)
class DataLoader:
    """데이터 로딩 전용 클래스"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.nba_client = NBAStatsClient()
        self.espn_client = ESPNInjuryClient()
        self.dnt_client = DNTApiClient()
        self.odds_client = OddsAPIClient()
        self._init_caches()

    # 데이터 로딩 메서드만 유지
    def load_team_epm(self, target_date: date) -> Dict[int, Dict]: ...
    def load_player_epm(self, season: int) -> pd.DataFrame: ...
    def load_team_game_logs(self, target_date: date) -> pd.DataFrame: ...
    def load_player_game_logs(self, target_date: date) -> pd.DataFrame: ...
    def get_games(self, game_date: date) -> List[Dict]: ...
    def get_injuries(self, team_abbr: str) -> List[ESPNInjury]: ...
    def get_game_odds(self, home_abbr: str, away_abbr: str) -> Optional[Dict]: ...

    # 캐시 관리
    def clear_cache(self): ...

    # 유틸리티
    def get_team_info(self, team_id: int) -> Dict: ...
```

### Phase 4: UI Layer (중위험) - 6주차

#### 4.1 GameRenderer 컴포넌트 추출

```python
# app/components/game_renderer.py
class GameRenderer:
    """게임 렌더링 컴포넌트"""

    def __init__(self, pipeline: PredictionPipeline):
        self.pipeline = pipeline

    def render_daily_games(self, game_date: date, team_epm: Dict):
        """일별 경기 렌더링"""
        games = self.pipeline.loader.get_games(game_date)

        for game in games:
            result = self.pipeline.predict_game(game, game_date, team_epm)
            self._render_game_card(game, result)

    def render_date_range_games(self, start_date: date, end_date: date,
                               team_epm: Dict):
        """기간별 경기 렌더링"""
        current = start_date
        while current <= end_date:
            self.render_daily_games(current, team_epm)
            current += timedelta(days=1)

    def _render_game_card(self, game: Dict, result: PredictionResult):
        """개별 게임 카드 렌더링"""
        card_data = self._build_card_data(game, result)
        render_game_card(card_data)

    def _build_card_data(self, game: Dict, result: PredictionResult) -> GameCardData:
        """GameCardData 객체 생성"""
        # ...
```

---

## 6. 테스트 전략

### 6.1 리팩토링 전 테스트 추가

```python
# tests/test_refactoring_baseline.py
class TestRefactoringBaseline:
    """리팩토링 전 동작 보장 테스트"""

    def test_prediction_result_unchanged(self, loader, predictor, sample_game):
        """예측 결과 동일성 검증"""
        game, date = sample_game
        team_epm = loader.load_team_epm(date)

        # 현재 구현 결과 저장
        features = loader.build_v5_4_features(
            game['home_team_id'], game['away_team_id'], team_epm, date
        )
        base_prob = predictor.predict_proba(features)

        # 스냅샷으로 저장
        assert 0.01 <= base_prob <= 0.99
        # 리팩토링 후에도 동일 결과 보장
```

### 6.2 각 Phase별 테스트

```python
# Phase 1 테스트
def test_log_processor_safe_diff():
    assert LogProcessor.safe_diff(5, 3) == 2.0
    assert LogProcessor.safe_diff(None, 3) == -3.0

def test_injury_adjuster_no_change():
    assert InjuryAdjuster.apply(0.5, 0, 0) == 0.5

def test_injury_adjuster_home_injury():
    result = InjuryAdjuster.apply(0.5, 10, 0)
    assert result == 0.4  # 50% - 10%

# Phase 2 테스트
def test_v54_feature_builder():
    builder = V54FeatureBuilder(mock_loader, mock_calc)
    features = builder.build(home_id, away_id, team_epm, date)

    assert len(features) == 5
    assert set(features.keys()) == set(builder.feature_names)

# Phase 3 테스트
def test_prediction_pipeline_e2e():
    pipeline = PredictionPipeline(loader, predictor)
    result = pipeline.predict_game(game, date, team_epm)

    assert 0.01 <= result.base_prob <= 0.99
    assert 0.01 <= result.adjusted_prob <= 0.99
```

### 6.3 회귀 테스트 체크리스트

```markdown
## 리팩토링 회귀 테스트 체크리스트

### 예측 정확도
- [ ] 동일 입력에 대해 동일 예측 결과 출력
- [ ] 확률 범위 1% ~ 99% 유지
- [ ] 피처 5개 생성 (V5.4)

### API 연동
- [ ] DNT API 30개 팀 EPM 로드
- [ ] NBA Stats API 경기 스케줄 조회
- [ ] ESPN 부상 정보 조회

### 부상 조정
- [ ] 부상 없으면 확률 변화 없음
- [ ] 홈팀 부상 → 홈 승률 감소
- [ ] 원정팀 부상 → 홈 승률 증가
- [ ] 확률 경계 유지

### 캐시
- [ ] 날짜별 팀 EPM 캐싱 동작
- [ ] 캐시 초기화 정상 작동

### 성능
- [ ] 단일 예측 < 500ms
- [ ] 10경기 예측 < 5초
```

---

## 7. 리팩토링 체크리스트

### 7.1 코드 품질 체크리스트

```markdown
## 메서드 리팩토링 체크리스트

### 메서드 크기
- [ ] 모든 메서드 50줄 이하
- [ ] 중첩 깊이 3단계 이하
- [ ] 순환 복잡도 10 이하

### 파라미터
- [ ] 파라미터 5개 이하
- [ ] 원시 타입 대신 객체 사용

### 책임
- [ ] 단일 책임 원칙 준수
- [ ] 메서드명이 동작을 명확히 설명

### 중복
- [ ] 코드 중복 제거
- [ ] 공통 로직 추출
```

### 7.2 클래스 리팩토링 체크리스트

```markdown
## 클래스 리팩토링 체크리스트

### 크기
- [ ] 클래스당 300줄 이하
- [ ] 메서드 15개 이하

### 응집도
- [ ] 관련 메서드만 포함
- [ ] 단일 책임

### 결합도
- [ ] 의존성 주입 사용
- [ ] 인터페이스 분리
```

---

## 8. 성공 지표

### 8.1 코드 메트릭 목표

| 메트릭 | 현재 | 목표 |
|--------|------|------|
| DataLoader 라인 수 | 1,381 | < 400 |
| main.py 라인 수 | 1,033 | < 600 |
| 평균 메서드 복잡도 | ~8 | < 5 |
| 최대 메서드 라인 | 219 | < 50 |
| 코드 중복률 | ~15% | < 5% |

### 8.2 품질 목표

| 목표 | 측정 방법 |
|------|----------|
| 테스트 커버리지 | 80% 이상 |
| 단위 테스트 격리 | 각 모듈 독립 테스트 가능 |
| 기능 동일성 | 모든 회귀 테스트 통과 |
| 성능 유지 | 응답 시간 10% 이내 변동 |

---

## 9. 위험 관리

### 9.1 위험 요소 및 완화 전략

| 위험 | 확률 | 영향 | 완화 전략 |
|------|------|------|----------|
| 예측 정확도 변화 | 중 | 높음 | 리팩토링 전 스냅샷 테스트 |
| 성능 저하 | 낮 | 중 | 프로파일링 후 최적화 |
| 하위 호환성 파괴 | 중 | 높음 | 어댑터 패턴 사용 |
| 일정 지연 | 중 | 중 | 점진적 리팩토링 |

### 9.2 롤백 전략

```bash
# 각 Phase 완료 후 태그 생성
git tag -a "refactor-phase1-complete" -m "Phase 1: Foundation complete"
git tag -a "refactor-phase2-complete" -m "Phase 2: Core refactoring complete"

# 문제 발생 시 롤백
git checkout refactor-phase1-complete
```

---

## 10. 부록

### 10.1 관련 문서

- [INTEGRATION_TEST_PLAN.md](INTEGRATION_TEST_PLAN.md) - 통합 테스트 계획
- [CLAUDE.md](/CLAUDE.md) - 프로젝트 가이드
- [MODEL_COMPARISON_REPORT.md](/bucketsvision_v4/docs/MODEL_COMPARISON_REPORT.md) - 모델 비교

### 10.2 참고 자료

- Martin Fowler, "Refactoring: Improving the Design of Existing Code"
- Clean Code by Robert C. Martin
- Design Patterns by Gang of Four

### 10.3 버전 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0.0 | 2025-12-03 | 초기 버전 |

---

*이 문서는 BucketsVision 프로젝트의 코드 품질 개선을 위한 리팩토링 가이드입니다.*
