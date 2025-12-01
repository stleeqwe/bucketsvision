"""
팀 로스터 및 선수 스탯 컴포넌트.

NBA API를 사용하여 팀 로스터와 선수별 시즌 스탯을 표시합니다.
DNT API의 EPM 데이터와 매칭하여 선수별 EPM을 표시합니다.
부상 정보 표시 기능 포함 (Out=빨강, GTD=노랑).

최적화: 앱 시작 시 전체 선수 스탯을 한 번에 로드하여 캐시.
"""

import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import pandas as pd
import streamlit as st

from nba_api.stats.endpoints import commonteamroster, leaguedashplayerstats
from config.constants import TEAM_INFO


# EPM 데이터 캐시
_epm_cache: Optional[pd.DataFrame] = None


# ============================================================
# 전체 선수 스탯 캐시 (앱 시작 시 로드)
# ============================================================

@st.cache_data(ttl=3600, show_spinner="선수 스탯 로딩 중...")
def load_all_player_stats(season: str = "2025-26") -> pd.DataFrame:
    """
    리그 전체 선수 스탯을 한 번에 로드 (캐시 1시간).

    leaguedashplayerstats API를 사용하여 한 번의 호출로
    모든 선수의 시즌 스탯을 가져옵니다.
    """
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            per_mode_detailed='PerGame'
        )
        df = stats.get_data_frames()[0]

        # 컬럼명 정리
        df = df.rename(columns={
            'PLAYER_ID': 'player_id',
            'PLAYER_NAME': 'player_name',
            'TEAM_ID': 'team_id',
            'TEAM_ABBREVIATION': 'team_abbr',
            'AGE': 'age',
            'GP': 'gp',
            'MIN': 'min',
            'PTS': 'pts',
            'REB': 'reb',
            'AST': 'ast',
            'STL': 'stl',
            'BLK': 'blk',
            'TOV': 'tov',
            'FG_PCT': 'fg_pct',
            'FG3_PCT': 'fg3_pct',
            'FT_PCT': 'ft_pct',
            'FG3M': 'fg3m',
            'FG3A': 'fg3a',
            'FTM': 'ftm',
            'FTA': 'fta',
            'OREB': 'oreb',
            'DREB': 'dreb',
        })

        # 이름 정규화 컬럼 추가
        df['name_normalized'] = df['player_name'].apply(_normalize_name)

        return df

    except Exception as e:
        st.error(f"선수 스탯 로드 실패: {e}")
        return pd.DataFrame()


def _normalize_name(name: str) -> str:
    """이름 정규화 (소문자, 악센트 제거, 공백 정리)"""
    if pd.isna(name) or not name:
        return ""
    # 악센트/특수문자 제거
    normalized = unicodedata.normalize('NFD', name)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    # 소문자로, 공백 정리
    return normalized.lower().strip()


@st.cache_data(ttl=3600, show_spinner=False)
def load_player_epm() -> pd.DataFrame:
    """DNT API의 선수 EPM 데이터 로드"""
    epm_path = Path("data/raw/dnt/season_epm/season_2026.parquet")
    if epm_path.exists():
        df = pd.read_parquet(epm_path)
        # 이름 정규화 컬럼 추가
        df['name_normalized'] = df['player_name'].apply(_normalize_name)
        return df
    return pd.DataFrame()


def find_player_epm(player_name: str, team_id: int, epm_df: pd.DataFrame) -> Optional[float]:
    """
    선수 이름으로 EPM 찾기 (퍼지 매칭 지원).

    Args:
        player_name: NBA API 선수 이름
        team_id: 팀 ID
        epm_df: EPM DataFrame

    Returns:
        선수의 총합 EPM 또는 None
    """
    if epm_df.empty:
        return None

    normalized = _normalize_name(player_name)

    # 1. 팀 필터링 후 정확한 매칭 시도
    team_players = epm_df[epm_df['team_id'] == team_id]

    exact_match = team_players[team_players['name_normalized'] == normalized]
    if not exact_match.empty:
        return round(exact_match.iloc[0]['tot'], 1)

    # 2. 퍼지 매칭 (70% 이상 일치)
    best_ratio = 0.0
    best_epm = None

    for _, player in team_players.iterrows():
        ratio = SequenceMatcher(None, normalized, player['name_normalized']).ratio()
        if ratio > best_ratio and ratio > 0.7:
            best_ratio = ratio
            best_epm = player['tot']

    if best_epm is not None:
        return round(best_epm, 1)

    # 3. 팀 무관 전체 검색 (동명이인 주의)
    all_matches = epm_df[epm_df['name_normalized'] == normalized]
    if len(all_matches) == 1:
        return round(all_matches.iloc[0]['tot'], 1)

    return None


def get_team_options() -> List[Tuple[str, int]]:
    """팀 선택 옵션 목록 반환 (이름 알파벳 순)"""
    teams = [(info["name"], team_id) for team_id, info in TEAM_INFO.items()]
    return sorted(teams, key=lambda x: x[0])


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_team_roster(team_id: int, season: str = "2025-26") -> pd.DataFrame:
    """팀 로스터 조회 (캐시 1시간)"""
    try:
        roster = commonteamroster.CommonTeamRoster(
            team_id=team_id,
            season=season
        )
        df = roster.common_team_roster.get_data_frame()
        return df
    except Exception as e:
        st.error(f"로스터 조회 실패: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_player_season_stats(player_id: int) -> Optional[Dict]:
    """선수 현재 시즌 스탯 조회"""
    try:
        time.sleep(0.2)  # Rate limiting
        stats = playercareerstats.PlayerCareerStats(
            player_id=player_id,
            per_mode36='PerGame'
        )
        df = stats.season_totals_regular_season.get_data_frame()

        # 현재 시즌 (2025-26) 스탯
        current_season = df[df['SEASON_ID'] == '2025-26']
        if not current_season.empty:
            row = current_season.iloc[0]
            return {
                'GP': int(row.get('GP', 0)),
                'MIN': round(row.get('MIN', 0), 1),
                'PTS': round(row.get('PTS', 0), 1),
                'REB': round(row.get('REB', 0), 1),
                'AST': round(row.get('AST', 0), 1),
                'STL': round(row.get('STL', 0), 1),
                'BLK': round(row.get('BLK', 0), 1),
                'TOV': round(row.get('TOV', 0), 1),
                'FG%': round(row.get('FG_PCT', 0) * 100, 1),
                '3P%': round(row.get('FG3_PCT', 0) * 100, 1),
                'FT%': round(row.get('FT_PCT', 0) * 100, 1),
                '3PM': round(row.get('FG3M', 0), 1),
                '3PA': round(row.get('FG3A', 0), 1),
                'FTM': round(row.get('FTM', 0), 1),
                'FTA': round(row.get('FTA', 0), 1),
                'OREB': round(row.get('OREB', 0), 1),
                'DREB': round(row.get('DREB', 0), 1),
            }
        return None
    except Exception:
        return None


def fetch_all_player_stats(roster_df: pd.DataFrame, team_id: int, progress_bar) -> pd.DataFrame:
    """모든 선수의 스탯을 가져와서 DataFrame으로 반환"""
    all_stats = []
    total = len(roster_df)

    # EPM 데이터 로드
    epm_df = load_player_epm()

    for idx, (_, player) in enumerate(roster_df.iterrows()):
        player_id = player['PLAYER_ID']
        player_name = player['PLAYER']
        player_num = player.get('NUM', '')
        player_age = player.get('AGE', 0)

        # 진행률 업데이트
        progress_bar.progress((idx + 1) / total, text=f"로딩 중... {player_name}")

        stats = fetch_player_season_stats(player_id)

        # EPM 찾기 (NBA API 이름 → DNT API EPM)
        player_epm = find_player_epm(player_name, team_id, epm_df)

        if stats:
            stats['번호'] = player_num
            stats['선수'] = player_name
            stats['나이'] = player_age
            stats['EPM'] = player_epm if player_epm is not None else '-'
            all_stats.append(stats)
        else:
            # 스탯이 없는 선수는 0으로 표시
            all_stats.append({
                '번호': player_num,
                '선수': player_name,
                '나이': player_age,
                'EPM': player_epm if player_epm is not None else '-',
                'GP': 0, 'MIN': 0, 'PTS': 0, 'REB': 0, 'AST': 0,
                'STL': 0, 'BLK': 0, 'TOV': 0,
                'FG%': 0, '3P%': 0, 'FT%': 0,
                '3PM': 0, '3PA': 0, 'FTM': 0, 'FTA': 0,
                'OREB': 0, 'DREB': 0
            })

    progress_bar.empty()

    if all_stats:
        df = pd.DataFrame(all_stats)
        # 컬럼 순서 재정렬 (EPM이 4열)
        cols = ['번호', '선수', '나이', 'EPM', 'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
                'FG%', '3P%', '3PM', '3PA', 'FT%', 'FTM', 'FTA',
                'OREB', 'DREB']
        df = df[cols]
        # MIN 기준 내림차순 정렬
        df = df.sort_values('MIN', ascending=False)
        return df

    return pd.DataFrame()


def render_team_roster_page(team_id: int, team_name: str, team_color: str) -> None:
    """팀 로스터 페이지 전체 렌더링"""

    # 헤더
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {team_color}, {team_color}99);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
    ">
        <h2 style="margin: 0; color: white;">{team_name}</h2>
        <span style="color: rgba(255,255,255,0.8);">2025-26 시즌 선수 스탯</span>
    </div>
    """, unsafe_allow_html=True)

    # 로스터 로드
    with st.spinner("로스터 로딩 중..."):
        roster_df = fetch_team_roster(team_id)

    if roster_df.empty:
        st.error("로스터를 불러올 수 없습니다.")
        return

    st.caption(f"총 {len(roster_df)}명의 선수")

    # 진행률 표시
    progress_bar = st.progress(0, text="선수 스탯 로딩 중...")

    # 모든 선수 스탯 로드 (team_id 전달하여 EPM 매칭)
    stats_df = fetch_all_player_stats(roster_df, team_id, progress_bar)

    if stats_df.empty:
        st.warning("선수 스탯을 불러올 수 없습니다.")
        return

    # 스타일 적용: 가운데 정렬, 폰트 크기
    st.markdown("""
    <style>
    /* 테이블 폰트 크기 */
    [data-testid="stDataFrame"] {
        font-size: 16px !important;
    }
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        font-size: 16px !important;
        text-align: center !important;
    }
    /* 테이블 셀 가운데 정렬 */
    [data-testid="stDataFrame"] div[data-testid="StyledLinkIconContainer"] {
        text-align: center !important;
        justify-content: center !important;
    }
    [data-testid="stDataFrame"] [class*="cell"] {
        text-align: center !important;
        justify-content: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 스탯 테이블 표시
    st.dataframe(
        stats_df,
        use_container_width=True,
        hide_index=True,
        height=600,
        column_config={
            "번호": st.column_config.TextColumn("번호", width=50),
            "선수": st.column_config.TextColumn("선수", width=130),
            "나이": st.column_config.NumberColumn("나이", help="선수 나이", format="%d", width=50),
            "EPM": st.column_config.TextColumn("EPM", help="Estimated Plus-Minus (DNT)"),
            "GP": st.column_config.NumberColumn("GP", help="경기 수", format="%d"),
            "MIN": st.column_config.NumberColumn("MIN", help="평균 출전 시간"),
            "PTS": st.column_config.NumberColumn("PTS", help="평균 득점"),
            "REB": st.column_config.NumberColumn("REB", help="평균 리바운드"),
            "AST": st.column_config.NumberColumn("AST", help="평균 어시스트"),
            "STL": st.column_config.NumberColumn("STL", help="평균 스틸"),
            "BLK": st.column_config.NumberColumn("BLK", help="평균 블록"),
            "TOV": st.column_config.NumberColumn("TOV", help="평균 턴오버"),
            "FG%": st.column_config.NumberColumn("FG%", help="필드골 성공률", format="%.1f%%"),
            "3P%": st.column_config.NumberColumn("3P%", help="3점 성공률", format="%.1f%%"),
            "3PM": st.column_config.NumberColumn("3PM", help="3점 성공"),
            "3PA": st.column_config.NumberColumn("3PA", help="3점 시도"),
            "FT%": st.column_config.NumberColumn("FT%", help="자유투 성공률", format="%.1f%%"),
            "FTM": st.column_config.NumberColumn("FTM", help="자유투 성공"),
            "FTA": st.column_config.NumberColumn("FTA", help="자유투 시도"),
            "OREB": st.column_config.NumberColumn("OREB", help="공격 리바운드"),
            "DREB": st.column_config.NumberColumn("DREB", help="수비 리바운드"),
        }
    )

    # 범례
    st.markdown("""
    <div style="color: #9ca3af; font-size: 0.75rem; margin-top: 10px;">
    EPM: Estimated Plus-Minus (DNT) | GP: 경기수 | MIN: 출전시간 | PTS: 득점 | REB: 리바운드 | AST: 어시스트<br>
    STL: 스틸 | BLK: 블록 | TOV: 턴오버 | FG%: 필드골% | 3P%: 3점% | FT%: 자유투% | OREB/DREB: 공격/수비 리바운드
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# 부상 표시 기능 (게임 상세 화면용)
# ============================================================

def get_injury_status_map(injuries: List) -> Dict[str, str]:
    """
    부상 정보를 선수명 -> 상태 맵으로 변환.

    Args:
        injuries: ESPNInjury 리스트

    Returns:
        {선수명: 상태} 딕셔너리 (상태: "Out", "GTD")
    """
    status_map = {}

    for injury in injuries:
        player_name = injury.player_name
        status = injury.status

        # Out 상태
        if status == "Out":
            status_map[player_name] = "Out"
        # Day-To-Day / Questionable
        elif status == "Day-To-Day" or injury.fantasy_status == "GTD":
            status_map[player_name] = "GTD"

    return status_map


def fetch_roster_with_injuries(
    team_id: int,
    team_abbr: str,
    injuries: List,
    season: str = "2025-26",
    show_progress: bool = True
) -> pd.DataFrame:
    """
    부상 정보가 포함된 로스터 DataFrame 반환.

    최적화: 전체 선수 스탯 캐시에서 팀 데이터만 필터링.
    로스터 API에서 선수 번호를 가져와 병합.

    Args:
        team_id: 팀 ID
        team_abbr: 팀 약어
        injuries: ESPNInjury 리스트
        season: 시즌
        show_progress: 진행률 표시 여부 (미사용, 호환성 유지)

    Returns:
        부상 상태 컬럼이 추가된 선수 스탯 DataFrame
    """
    # 전체 선수 스탯 캐시에서 로드
    all_stats = load_all_player_stats(season)
    if all_stats.empty:
        return pd.DataFrame()

    # 해당 팀 선수만 필터링
    team_stats = all_stats[all_stats['team_id'] == team_id].copy()
    if team_stats.empty:
        # team_abbr로 재시도
        team_stats = all_stats[all_stats['team_abbr'] == team_abbr].copy()

    if team_stats.empty:
        return pd.DataFrame()

    # 로스터 API에서 선수 번호 가져오기
    roster_df = fetch_team_roster(team_id, season)
    jersey_map = {}
    if not roster_df.empty:
        for _, row in roster_df.iterrows():
            player_name = row.get('PLAYER', '')
            jersey_num = row.get('NUM', '')
            if player_name and jersey_num:
                jersey_map[_normalize_name(player_name)] = jersey_num

    def get_jersey_number(player_name: str) -> str:
        normalized = _normalize_name(player_name)
        if normalized in jersey_map:
            return jersey_map[normalized]
        # 퍼지 매칭
        for name, num in jersey_map.items():
            if SequenceMatcher(None, normalized, name).ratio() > 0.8:
                return num
        return ''

    team_stats['jersey'] = team_stats['player_name'].apply(get_jersey_number)

    # EPM 데이터 로드 및 매칭
    epm_df = load_player_epm()

    def get_player_epm(row) -> str:
        epm = find_player_epm(row['player_name'], team_id, epm_df)
        return str(epm) if epm is not None else '-'

    team_stats['EPM'] = team_stats.apply(get_player_epm, axis=1)

    # 부상 상태 맵 생성
    injury_map = get_injury_status_map(injuries)

    def get_injury_status(player_name: str) -> str:
        if player_name in injury_map:
            return injury_map[player_name]
        normalized = _normalize_name(player_name)
        for inj_name, status in injury_map.items():
            if SequenceMatcher(None, normalized, _normalize_name(inj_name)).ratio() > 0.8:
                return status
        return ""

    team_stats['상태'] = team_stats['player_name'].apply(get_injury_status)

    # 출력용 DataFrame 구성
    result_df = pd.DataFrame({
        '번호': team_stats['jersey'],
        '선수': team_stats['player_name'],
        '나이': team_stats['age'].astype(int),
        'EPM': team_stats['EPM'],
        'GP': team_stats['gp'].astype(int),
        'MIN': team_stats['min'].round(1),
        'PTS': team_stats['pts'].round(1),
        'REB': team_stats['reb'].round(1),
        'AST': team_stats['ast'].round(1),
        'STL': team_stats['stl'].round(1),
        'BLK': team_stats['blk'].round(1),
        'TOV': team_stats['tov'].round(1),
        'FG%': (team_stats['fg_pct'] * 100).round(1),
        '3P%': (team_stats['fg3_pct'] * 100).round(1),
        'FT%': (team_stats['ft_pct'] * 100).round(1),
        '상태': team_stats['상태'],
    })

    # MIN 기준 내림차순 정렬
    result_df = result_df.sort_values('MIN', ascending=False).reset_index(drop=True)

    return result_df


def render_roster_table_with_injuries(
    stats_df: pd.DataFrame,
    team_name: str,
    team_color: str,
    compact: bool = False
) -> None:
    """
    부상 표시가 포함된 로스터 테이블 렌더링.

    Args:
        stats_df: 선수 스탯 DataFrame (상태 컬럼 포함)
        team_name: 팀 이름
        team_color: 팀 색상
        compact: 간소화 모드 (상세 화면에서 사용)
    """
    if stats_df.empty:
        st.warning(f"{team_name} 로스터를 불러올 수 없습니다.")
        return

    # 헤더
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {team_color}, {team_color}99);
        padding: 12px 20px;
        border-radius: 10px;
        margin-bottom: 12px;
    ">
        <h4 style="margin: 0; color: white;">{team_name} 로스터</h4>
    </div>
    """, unsafe_allow_html=True)

    # 부상자 범례
    injury_legend = ""
    out_count = (stats_df['상태'] == 'Out').sum()
    gtd_count = (stats_df['상태'] == 'GTD').sum()

    if out_count > 0 or gtd_count > 0:
        parts = []
        if out_count > 0:
            parts.append(f'<span style="color: #ef4444;">🔴 Out: {out_count}명</span>')
        if gtd_count > 0:
            parts.append(f'<span style="color: #eab308;">🟡 GTD: {gtd_count}명</span>')
        injury_legend = f'<div style="font-size: 0.8rem; margin-bottom: 8px;">{" | ".join(parts)}</div>'
        st.markdown(injury_legend, unsafe_allow_html=True)

    # 선수명에 부상 상태 표시 추가
    def format_player_name(row) -> str:
        name = row['선수']
        status = row.get('상태', '')

        if status == 'Out':
            return f"🔴 {name}"
        elif status == 'GTD':
            return f"🟡 {name}"
        return name

    display_df = stats_df.copy()
    display_df['선수'] = display_df.apply(format_player_name, axis=1)

    # 상태 컬럼 제거 (이미 선수명에 표시됨)
    if '상태' in display_df.columns:
        display_df = display_df.drop(columns=['상태'])

    # 컴팩트 모드: 주요 컬럼만 표시
    if compact:
        compact_cols = ['번호', '선수', 'EPM', 'MIN', 'PTS', 'REB', 'AST']
        display_df = display_df[[c for c in compact_cols if c in display_df.columns]]
        height = 350
    else:
        height = 500

    # 스타일
    st.markdown("""
    <style>
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th {
        font-size: 14px !important;
        text-align: center !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # 테이블 렌더링
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        height=height,
        column_config={
            "번호": st.column_config.TextColumn("번호", width=50),
            "선수": st.column_config.TextColumn("선수", width=140),
            "나이": st.column_config.NumberColumn("나이", format="%d", width=45),
            "EPM": st.column_config.TextColumn("EPM", width=55),
            "GP": st.column_config.NumberColumn("GP", format="%d", width=45),
            "MIN": st.column_config.NumberColumn("MIN", format="%.1f", width=55),
            "PTS": st.column_config.NumberColumn("PTS", format="%.1f", width=55),
            "REB": st.column_config.NumberColumn("REB", format="%.1f", width=55),
            "AST": st.column_config.NumberColumn("AST", format="%.1f", width=55),
            "STL": st.column_config.NumberColumn("STL", format="%.1f", width=50),
            "BLK": st.column_config.NumberColumn("BLK", format="%.1f", width=50),
            "TOV": st.column_config.NumberColumn("TOV", format="%.1f", width=50),
            "FG%": st.column_config.NumberColumn("FG%", format="%.1f%%", width=55),
            "3P%": st.column_config.NumberColumn("3P%", format="%.1f%%", width=55),
            "FT%": st.column_config.NumberColumn("FT%", format="%.1f%%", width=55),
        }
    )
