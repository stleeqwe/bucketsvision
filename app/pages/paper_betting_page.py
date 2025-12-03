"""
Paper Betting 페이지.

리팩토링 Phase 4: main.py에서 추출.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st


def render_paper_betting_page(project_root: Optional[Path] = None) -> None:
    """
    Paper Betting 대시보드 렌더링.

    Args:
        project_root: 프로젝트 루트 (None이면 자동 탐지)
    """
    st.subheader("💰 Paper Betting Dashboard")

    if project_root is None:
        project_root = Path(__file__).parent.parent.parent

    data = _load_paper_betting_data(project_root)

    if not data:
        st.warning("Paper Betting 데이터가 없습니다. 스크립트를 먼저 실행해주세요.")
        st.code("python scripts/paper_betting.py", language="bash")
        return

    summary = data.get("summary", {})
    bets = data.get("bets", [])
    metadata = data.get("metadata", {})

    # 요약 통계
    _render_summary_stats(summary)

    # 설정 정보
    edge_threshold = metadata.get("edge_threshold", 0.08)
    unit_size = metadata.get("unit_size", 100)
    st.caption(f"⚙️ Edge 기준: ≥{edge_threshold * 100:.0f}% | Unit: ${unit_size}")

    st.markdown("---")

    # 베팅 기록
    _render_betting_history(bets)


def _load_paper_betting_data(project_root: Path) -> Optional[Dict]:
    """Paper Betting 데이터 로드"""
    bets_file = project_root / "data" / "paper_betting" / "bets.json"
    if bets_file.exists():
        with open(bets_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def _render_summary_stats(summary: Dict) -> None:
    """요약 통계 렌더링"""
    st.markdown("### 📊 Overall Performance")

    total_bets = summary.get("total_bets", 0)
    wins = summary.get("wins", 0)
    losses = summary.get("losses", 0)
    pending = summary.get("pending", 0)
    total_profit = summary.get("total_profit", 0)
    roi = summary.get("roi", 0)

    settled = wins + losses
    win_rate = (wins / settled * 100) if settled > 0 else 0

    # 메트릭 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("총 베팅", f"{total_bets}건")
    with col2:
        st.metric("승률", f"{win_rate:.1f}%" if settled > 0 else "-")
    with col3:
        profit_color = "normal" if total_profit >= 0 else "inverse"
        st.metric("총 수익", f"${total_profit:+,.0f}", delta_color=profit_color)
    with col4:
        st.metric("ROI", f"{roi:+.1f}%")

    # 상세 통계
    st.markdown(f"""
    <div style="
        background: #1a1a2e;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    ">
        <div style="display: flex; justify-content: space-around; text-align: center;">
            <div>
                <div style="color: #22c55e; font-size: 2rem; font-weight: bold;">{wins}</div>
                <div style="color: #888;">승리</div>
            </div>
            <div>
                <div style="color: #ef4444; font-size: 2rem; font-weight: bold;">{losses}</div>
                <div style="color: #888;">패배</div>
            </div>
            <div>
                <div style="color: #f59e0b; font-size: 2rem; font-weight: bold;">{pending}</div>
                <div style="color: #888;">대기중</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_betting_history(bets: List[Dict]) -> None:
    """베팅 기록 렌더링"""
    st.markdown("### 📋 Betting History")

    if not bets:
        st.info("아직 베팅 기록이 없습니다.")
        return

    # 날짜별 그룹핑
    daily_bets = defaultdict(list)
    for bet in bets:
        daily_bets[bet['date']].append(bet)

    # 최신순 정렬
    for bet_date in sorted(daily_bets.keys(), reverse=True):
        day_bets = daily_bets[bet_date]
        _render_day_bets(bet_date, day_bets)


def _render_day_bets(bet_date: str, day_bets: List[Dict]) -> None:
    """일별 베팅 기록 렌더링"""
    # 날짜별 소계
    day_profit = sum(
        b.get('profit', 0) or 0
        for b in day_bets
        if b['status'] == 'settled'
    )
    day_wins = sum(1 for b in day_bets if b.get('result') == 'win')
    day_losses = sum(1 for b in day_bets if b.get('result') == 'loss')
    day_pending = sum(1 for b in day_bets if b['status'] == 'pending')

    # 날짜 헤더
    profit_emoji = "🟢" if day_profit > 0 else ("🔴" if day_profit < 0 else "⚪")
    pending_str = f" | ⏳ {day_pending} pending" if day_pending > 0 else ""

    if day_wins + day_losses > 0:
        st.markdown(
            f"#### {bet_date} — {day_wins}W-{day_losses}L "
            f"{profit_emoji} ${day_profit:+,.0f}{pending_str}"
        )
    else:
        st.markdown(f"#### {bet_date}{pending_str}")

    # 개별 베팅
    for bet in day_bets:
        _render_single_bet(bet)

    st.markdown("")


def _render_single_bet(bet: Dict) -> None:
    """단일 베팅 렌더링"""
    status = bet['status']
    bet_team = bet['bet_team']
    bet_odds = bet['bet_odds']
    edge = bet['bet_edge'] * 100
    home_team = bet['home_team']
    away_team = bet['away_team']

    if status == 'settled':
        result = bet.get('result')
        profit = bet.get('profit', 0)
        home_score = bet.get('home_score', '?')
        away_score = bet.get('away_score', '?')

        if result == 'win':
            emoji = "✅"
            color = "#22c55e"
        else:
            emoji = "❌"
            color = "#ef4444"

        st.markdown(f"""
        <div style="
            background: #1e293b;
            border-left: 4px solid {color};
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    {emoji} <strong>{bet_team}</strong> @{bet_odds:.2f}
                    <span style="color: #64748b; font-size: 0.85rem;">
                        | Edge {edge:.1f}% | {away_team} @ {home_team}
                    </span>
                </div>
                <div>
                    <span style="color: #94a3b8;">[{away_score}-{home_score}]</span>
                    <span style="color: {color}; font-weight: bold; margin-left: 10px;">
                        {'+' if profit > 0 else ''}{profit:.0f}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Pending
        potential = bet.get('potential_profit', 0)
        st.markdown(f"""
        <div style="
            background: #1e293b;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 0 8px 8px 0;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    ⏳ <strong>{bet_team}</strong> @{bet_odds:.2f}
                    <span style="color: #64748b; font-size: 0.85rem;">
                        | Edge {edge:.1f}% | {away_team} @ {home_team}
                    </span>
                </div>
                <div style="color: #94a3b8;">
                    (potential: +${potential:.0f})
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
