"""
게임 카드 컴포넌트 V2 - 순수 Streamlit 네이티브 버전.

st.container, st.columns, st.markdown을 조합하여 구현.
커스텀 입력이 카드 내부에 자연스럽게 통합됨.
"""

import streamlit as st
from typing import Dict, List, Optional


# 색상 상수
HOME_COLOR = "#3b82f6"
AWAY_COLOR = "#ef4444"
SUCCESS_COLOR = "#22c55e"
FAIL_COLOR = "#ef4444"
LIVE_COLOR = "#eab308"


def inject_card_styles():
    """전역 CSS 스타일 주입 - 컨테이너 스타일링용."""
    st.markdown("""
    <style>
    /* Streamlit 컨테이너 기본 스타일 제거 */
    div[data-testid="stVerticalBlock"] > div:has(> div.game-card-wrapper) {
        background: transparent;
    }

    /* 컬럼 패딩 조정 */
    div[data-testid="column"] {
        padding: 0 8px;
    }

    /* number_input 스타일 */
    div[data-testid="stNumberInput"] input {
        background-color: #1f2937;
        border-color: #374151;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)


def calculate_betting_edge(
    model_prob: float,
    ml_home: Optional[float],
    ml_away: Optional[float],
) -> Optional[Dict]:
    """모델 확률 vs Pinnacle 머니라인 기반 엣지 계산."""
    if ml_home is None or ml_away is None:
        return None
    if ml_home <= 1 or ml_away <= 1:
        return None

    implied_home = 1 / ml_home
    implied_away = 1 / ml_away
    total_implied = implied_home + implied_away

    fair_home = implied_home / total_implied
    fair_away = implied_away / total_implied

    edge_home = model_prob - fair_home
    edge_away = (1 - model_prob) - fair_away

    ev_home = model_prob * (ml_home - 1) - (1 - model_prob)
    ev_away = (1 - model_prob) * (ml_away - 1) - model_prob

    if edge_home > 0.08:
        bet_side = 'home'
    elif edge_away > 0.08:
        bet_side = 'away'
    else:
        bet_side = 'none'

    return {
        'pinnacle_home_prob': fair_home,
        'pinnacle_away_prob': fair_away,
        'edge_home': edge_home,
        'edge_away': edge_away,
        'ev_home': ev_home,
        'ev_away': ev_away,
        'bet_side': bet_side,
    }


def render_injury_section(
    home_team: str,
    away_team: str,
    home_injury_summary: Optional[Dict],
    away_injury_summary: Optional[Dict],
    game_id: str,
    on_gtd_toggle: Optional[callable] = None,
) -> Dict[str, bool]:
    """
    부상자 정보 섹션 렌더링 (간소화 버전).

    모든 Out/GTD 선수를 나열하고, impact가 있는 선수만 수치 표시.
    """
    gtd_states = {}

    home_out = home_injury_summary.get("out_players", []) if home_injury_summary else []
    home_gtd = home_injury_summary.get("gtd_players", []) if home_injury_summary else []
    away_out = away_injury_summary.get("out_players", []) if away_injury_summary else []
    away_gtd = away_injury_summary.get("gtd_players", []) if away_injury_summary else []

    has_injuries = home_out or home_gtd or away_out or away_gtd

    if not has_injuries:
        return gtd_states

    st.markdown('''
        <div style="background: #111827; border-radius: 8px; padding: 12px; margin-top: 10px;">
            <div style="color: #6b7280; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px;">🏥 부상자 명단</div>
    ''', unsafe_allow_html=True)

    # 홈팀 부상자
    if home_out or home_gtd:
        st.markdown(f'''
            <div style="margin-bottom: 8px;">
                <span style="color: {HOME_COLOR}; font-weight: 700; font-size: 0.9rem;">{home_team}</span>
            </div>
        ''', unsafe_allow_html=True)

        # Out 선수
        for player in home_out:
            _render_injury_player_simple(player, "Out", "#ef4444")

        # GTD 선수
        for player in home_gtd:
            _render_injury_player_simple(player, "GTD", "#f59e0b")

    # 원정팀 부상자
    if away_out or away_gtd:
        st.markdown(f'''
            <div style="margin: 10px 0 8px 0;">
                <span style="color: {AWAY_COLOR}; font-weight: 700; font-size: 0.9rem;">{away_team}</span>
            </div>
        ''', unsafe_allow_html=True)

        # Out 선수
        for player in away_out:
            _render_injury_player_simple(player, "Out", "#ef4444")

        # GTD 선수
        for player in away_gtd:
            _render_injury_player_simple(player, "GTD", "#f59e0b")

    st.markdown('</div>', unsafe_allow_html=True)

    return gtd_states


def _render_injury_player_simple(player: Dict, status: str, status_color: str):
    """부상 선수 행 렌더링 (V2 - prob_shift 기반)."""
    name = player.get("name", "Unknown")
    prob_shift = player.get("prob_shift", 0.0)  # 이미 % 단위

    # GTD는 노란색, Out은 빨간색 (단순화)
    badge_color = "#eab308" if status == "GTD" else "#ef4444"

    # prob_shift가 있는 경우에만 수치 표시
    if prob_shift > 0:
        st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #1f2937;">
                <div>
                    <span style="color: #e5e7eb; font-weight: 500; font-size: 0.85rem;">{name}</span>
                    <span style="color: {badge_color}; font-size: 0.7rem; font-weight: 600; margin-left: 6px;">{status}</span>
                </div>
                <span style="color: #9ca3af; font-size: 0.8rem;">-{prob_shift:.1f}%</span>
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 5px 0; border-bottom: 1px solid #1f2937;">
                <div>
                    <span style="color: #6b7280; font-size: 0.85rem;">{name}</span>
                    <span style="color: {badge_color}; font-size: 0.7rem; font-weight: 600; margin-left: 6px;">{status}</span>
                </div>
            </div>
        ''', unsafe_allow_html=True)


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
    adjusted_margin: Optional[float] = None,
    adjusted_win_prob: Optional[float] = None,
    game_status: int = 1,
    home_score: Optional[int] = None,
    away_score: Optional[int] = None,
    home_b2b: bool = False,
    away_b2b: bool = False,
    hide_result: bool = False,
    odds_info: Optional[Dict] = None,
    game_id: Optional[str] = None,
    enable_custom_input: bool = False,
    home_injury_summary: Optional[Dict] = None,
    away_injury_summary: Optional[Dict] = None,
    home_prob_shift: float = 0.0,
    away_prob_shift: float = 0.0,
) -> Dict[str, bool]:
    """
    순수 Streamlit 네이티브 게임 카드 렌더링.

    Returns:
        GTD 선수 포함 여부 딕셔너리 (예정 경기일 경우)
    """
    gtd_states = {}

    final_home_prob = adjusted_win_prob if adjusted_win_prob is not None else home_win_prob
    final_away_prob = 1 - final_home_prob
    home_pct = final_home_prob * 100
    away_pct = final_away_prob * 100

    predicted_home_win = final_home_prob >= 0.5
    is_finished = game_status == 3 and home_score is not None and away_score is not None
    show_result = is_finished and not hide_result
    is_live = game_status == 2 and home_score is not None and away_score is not None

    actual_home_win = None
    is_correct = None
    if is_finished:
        actual_home_win = home_score > away_score
        is_correct = predicted_home_win == actual_home_win

    # 카드 스타일 결정
    if show_result:
        border_color = SUCCESS_COLOR if is_correct else FAIL_COLOR
        # 적중: 초록 음영, 미적중: 빨강 음영 (눈에 띄게)
        bg_gradient = f"linear-gradient(145deg, {'#1a4d2e' if is_correct else '#4d1a1a'} 0%, {'#0f3d1f' if is_correct else '#3d0f0f'} 100%)"
    elif is_live:
        border_color = LIVE_COLOR
        bg_gradient = "linear-gradient(145deg, #4d3d1a 0%, #3d2f0f 100%)"
    else:
        border_color = "#2d3748"
        bg_gradient = "#161b22"

    # 상태 텍스트
    if game_status == 3:
        status_text = "Final"
        status_color = "#6b7280"
    elif game_status == 2:
        status_text = "🔴 Live"
        status_color = LIVE_COLOR
    else:
        status_text = game_time
        status_color = "#9ca3af"

    # 카드 시작 - 전체를 감싸는 div
    card_style = f"border: 2px solid {border_color}; border-radius: 12px; overflow: hidden; margin: 12px 0; background: {bg_gradient};"

    st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)

    # 카드 내부 패딩 컨테이너
    st.markdown('<div style="padding: 20px;">', unsafe_allow_html=True)

    # 적중/실패 배지 (종료된 경기)
    if show_result:
        result_text = "✓ 적중" if is_correct else "✗ 실패"
        result_color = SUCCESS_COLOR if is_correct else FAIL_COLOR
        st.markdown(f'''
            <div style="text-align: right; margin-bottom: 8px;">
                <span style="color: {result_color}; font-size: 0.8rem; font-weight: 600;">{result_text}</span>
            </div>
        ''', unsafe_allow_html=True)

    # 상태 배지 (시간/Final/Live)
    st.markdown(f'''
        <div style="text-align: center; margin-bottom: 16px;">
            <span style="background: #374151; color: {status_color}; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem;">{status_text}</span>
        </div>
    ''', unsafe_allow_html=True)

    # 팀 정보 레이아웃
    col_home, col_vs, col_away = st.columns([2, 1, 2])

    with col_home:
        # B2B 배지
        b2b_badge = f'<span style="background: #f59e0b33; color: #f59e0b; padding: 2px 6px; border-radius: 8px; font-size: 0.65rem; margin-left: 4px;">B2B</span>' if home_b2b else ""

        st.markdown(f'''
            <div style="text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 800; color: {HOME_COLOR};">{home_team}</div>
                <div style="font-size: 0.8rem; color: #9ca3af;">{home_name}</div>
                <div style="margin-top: 6px;">
                    <span style="background: {HOME_COLOR}22; color: {HOME_COLOR}; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem;">HOME</span>{b2b_badge}
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 점수 (종료/라이브)
        if is_finished or is_live:
            is_home_winner = (is_finished and actual_home_win) or (is_live and home_score > away_score)
            score_color = "#ffffff" if is_home_winner else "#4b5563"
            st.markdown(f'<div style="text-align: center; font-size: 2rem; font-weight: 800; color: {score_color}; margin-top: 8px;">{home_score}</div>', unsafe_allow_html=True)

        # 확률
        prob_opacity = "0.7" if is_finished else "1"
        prob_size = "1.2rem" if (is_finished or is_live) else "1.6rem"
        st.markdown(f'<div style="text-align: center; font-size: {prob_size}; font-weight: 700; color: {HOME_COLOR}; margin-top: 8px; opacity: {prob_opacity};">{final_home_prob:.1%}</div>', unsafe_allow_html=True)

    with col_vs:
        st.markdown('<div style="display: flex; align-items: center; justify-content: center; height: 100%; min-height: 80px;"><span style="color: #4b5563; font-size: 1.1rem; font-weight: 600;">VS</span></div>', unsafe_allow_html=True)

    with col_away:
        # B2B 배지
        b2b_badge = f'<span style="background: #f59e0b33; color: #f59e0b; padding: 2px 6px; border-radius: 8px; font-size: 0.65rem; margin-left: 4px;">B2B</span>' if away_b2b else ""

        st.markdown(f'''
            <div style="text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 800; color: {AWAY_COLOR};">{away_team}</div>
                <div style="font-size: 0.8rem; color: #9ca3af;">{away_name}</div>
                <div style="margin-top: 6px;">
                    <span style="background: {AWAY_COLOR}22; color: {AWAY_COLOR}; padding: 2px 10px; border-radius: 12px; font-size: 0.7rem;">AWAY</span>{b2b_badge}
                </div>
            </div>
        ''', unsafe_allow_html=True)

        # 점수 (종료/라이브)
        if is_finished or is_live:
            is_away_winner = (is_finished and not actual_home_win) or (is_live and away_score > home_score)
            score_color = "#ffffff" if is_away_winner else "#4b5563"
            st.markdown(f'<div style="text-align: center; font-size: 2rem; font-weight: 800; color: {score_color}; margin-top: 8px;">{away_score}</div>', unsafe_allow_html=True)

        # 확률
        prob_opacity = "0.7" if is_finished else "1"
        prob_size = "1.2rem" if (is_finished or is_live) else "1.6rem"
        st.markdown(f'<div style="text-align: center; font-size: {prob_size}; font-weight: 700; color: {AWAY_COLOR}; margin-top: 8px; opacity: {prob_opacity};">{final_away_prob:.1%}</div>', unsafe_allow_html=True)

    # 확률 바
    st.markdown(f'''
        <div style="height: 8px; background: #1f2937; border-radius: 4px; overflow: hidden; display: flex; margin: 16px 0;">
            <div style="width: {home_pct:.1f}%; height: 100%; background: {HOME_COLOR};"></div>
            <div style="width: {away_pct:.1f}%; height: 100%; background: {AWAY_COLOR};"></div>
        </div>
    ''', unsafe_allow_html=True)

    # 하단 섹션 렌더링
    _render_bottom_section(
        home_team, away_team, final_home_prob, final_away_prob,
        predicted_margin, adjusted_margin, game_status, is_finished, show_result,
        home_score, away_score, actual_home_win, odds_info,
        game_id, enable_custom_input
    )

    # 부상자 정보 섹션 (예정된 경기만)
    if game_status == 1 and (home_injury_summary or away_injury_summary):
        gtd_states = render_injury_section(
            home_team=home_team,
            away_team=away_team,
            home_injury_summary=home_injury_summary,
            away_injury_summary=away_injury_summary,
            game_id=game_id or f"{home_team}_{away_team}",
        )

    # 패딩 컨테이너 종료
    st.markdown('</div>', unsafe_allow_html=True)

    # 카드 종료
    st.markdown('</div>', unsafe_allow_html=True)

    return gtd_states


def _render_bottom_section(
    home_team, away_team, final_home_prob, final_away_prob,
    predicted_margin, adjusted_margin, game_status, is_finished, show_result,
    home_score, away_score, actual_home_win, odds_info,
    game_id, enable_custom_input
):
    """하단 섹션 렌더링 (상태별 분기)."""

    # 종료된 경기: 예측 vs 실제
    if is_finished and show_result:
        actual_margin = home_score - away_score
        final_predicted = adjusted_margin if adjusted_margin is not None else predicted_margin
        error = abs(final_predicted - actual_margin)

        pred_text = f"{home_team} -{abs(final_predicted):.1f}" if final_predicted > 0 else f"{away_team} -{abs(final_predicted):.1f}"
        if actual_margin > 0:
            actual_text = f"{home_team} -{abs(actual_margin)}"
        elif actual_margin < 0:
            actual_text = f"{away_team} -{abs(actual_margin)}"
        else:
            actual_text = "TIE"

        error_color = "#22c55e" if error <= 5 else ("#eab308" if error <= 10 else "#ef4444")

        st.markdown(f'''
            <div style="display: flex; justify-content: space-between; background: #111827; border-radius: 8px; padding: 12px; margin-top: 8px;">
                <div style="text-align: center; flex: 1;">
                    <div style="color: #6b7280; font-size: 0.7rem;">예측</div>
                    <div style="color: #9ca3af; font-weight: 600;">{pred_text}</div>
                </div>
                <div style="text-align: center; flex: 0 0 80px;">
                    <div style="color: #6b7280; font-size: 0.7rem;">오차</div>
                    <div style="color: {error_color}; font-weight: 700; font-size: 1.1rem;">{error:.1f}pt</div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="color: #6b7280; font-size: 0.7rem;">실제</div>
                    <div style="color: #e5e7eb; font-weight: 600;">{actual_text}</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
        return

    # 예정 경기 + 배당 있음: Edge 테이블
    if game_status == 1 and odds_info:
        ml_home = odds_info.get("moneyline_home")
        ml_away = odds_info.get("moneyline_away")

        if ml_home is not None and ml_away is not None:
            edge_data = calculate_betting_edge(final_home_prob, ml_home, ml_away)

            if edge_data:
                def edge_color(e):
                    if e >= 0.05: return "#10b981"
                    if e >= 0.03: return "#22c55e"
                    if e > 0: return "#facc15"
                    return "#6b7280"

                def ev_color(v):
                    return "#22c55e" if v > 0 else "#ef4444"

                he, ae = edge_data['edge_home'], edge_data['edge_away']
                hev, aev = edge_data['ev_home'], edge_data['ev_away']
                hp, ap = edge_data['pinnacle_home_prob'], edge_data['pinnacle_away_prob']

                st.markdown(f'''
                    <div style="background: #111827; border-radius: 8px; padding: 12px; margin-top: 8px;">
                        <div style="color: #6b7280; font-size: 0.8rem; font-weight: 600; margin-bottom: 10px;">💰 모델 Edge (vs Pinnacle)</div>
                        <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                            <tr style="color: #4b5563; font-size: 0.7rem;">
                                <td style="padding: 6px 4px; width: 35%;">팀</td>
                                <td style="text-align: right; padding: 6px 4px; width: 16%;">모델</td>
                                <td style="text-align: right; padding: 6px 4px; width: 16%;">시장</td>
                                <td style="text-align: right; padding: 6px 4px; width: 16%;">Edge</td>
                                <td style="text-align: right; padding: 6px 4px; width: 17%;">EV</td>
                            </tr>
                            <tr style="border-top: 1px solid #1f2937;">
                                <td style="color: {HOME_COLOR}; font-weight: 600; padding: 8px 4px;">{home_team} <span style="color: #4b5563; font-size: 0.75rem;">@{ml_home:.2f}</span></td>
                                <td style="text-align: right; color: #e5e7eb; font-weight: 600; padding: 8px 4px;">{final_home_prob*100:.1f}%</td>
                                <td style="text-align: right; color: #9ca3af; padding: 8px 4px;">{hp*100:.1f}%</td>
                                <td style="text-align: right; color: {edge_color(he)}; font-weight: 700; padding: 8px 4px;">{he*100:+.1f}%</td>
                                <td style="text-align: right; color: #9ca3af; padding: 8px 4px;">{hev*100:+.1f}%</td>
                            </tr>
                            <tr style="border-top: 1px solid #1f2937;">
                                <td style="color: {AWAY_COLOR}; font-weight: 600; padding: 8px 4px;">{away_team} <span style="color: #4b5563; font-size: 0.75rem;">@{ml_away:.2f}</span></td>
                                <td style="text-align: right; color: #e5e7eb; font-weight: 600; padding: 8px 4px;">{final_away_prob*100:.1f}%</td>
                                <td style="text-align: right; color: #9ca3af; padding: 8px 4px;">{ap*100:.1f}%</td>
                                <td style="text-align: right; color: {edge_color(ae)}; font-weight: 700; padding: 8px 4px;">{ae*100:+.1f}%</td>
                                <td style="text-align: right; color: #9ca3af; padding: 8px 4px;">{aev*100:+.1f}%</td>
                            </tr>
                        </table>
                    </div>
                ''', unsafe_allow_html=True)

                # 커스텀 입력 섹션 (카드 내부에 통합!)
                if enable_custom_input and game_id:
                    st.markdown('''
                        <div style="background: #0f1419; border-radius: 8px; padding: 12px; margin-top: 10px;">
                            <div style="color: #6b7280; font-size: 0.8rem; font-weight: 600; margin-bottom: 8px;">🎯 내 확률로 Edge 계산</div>
                        </div>
                    ''', unsafe_allow_html=True)

                    col1, col2 = st.columns([1, 2])

                    with col1:
                        custom_prob = st.number_input(
                            f"{home_team} 승률 (%)",
                            min_value=1.0,
                            max_value=99.0,
                            value=round(final_home_prob * 100, 1),
                            step=1.0,
                            key=f"custom_{game_id}",
                            label_visibility="visible"
                        )

                    with col2:
                        custom_edge = calculate_betting_edge(custom_prob / 100.0, ml_home, ml_away)
                        if custom_edge:
                            if custom_edge['edge_home'] > 0.08:
                                rec_team = home_team
                                rec_edge = custom_edge['edge_home']
                                rec_ev = custom_edge['ev_home']
                                rec_odds = ml_home
                                team_color = HOME_COLOR
                                box_bg = f"linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.05))"
                                box_border = f"1px solid rgba(59, 130, 246, 0.4)"
                            elif custom_edge['edge_away'] > 0.08:
                                rec_team = away_team
                                rec_edge = custom_edge['edge_away']
                                rec_ev = custom_edge['ev_away']
                                rec_odds = ml_away
                                team_color = AWAY_COLOR
                                box_bg = f"linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05))"
                                box_border = f"1px solid rgba(239, 68, 68, 0.4)"
                            else:
                                rec_team = None

                            if rec_team:
                                st.markdown(f'''
                                    <div style="background: {box_bg}; border: {box_border}; border-radius: 10px; padding: 12px 16px; margin-top: 8px;">
                                        <div>
                                            <span style="font-weight: 700; font-size: 1rem; color: {team_color};">{rec_team}</span>
                                            <span style="color: #6b7280; font-size: 0.85rem;">@{rec_odds:.2f}</span>
                                        </div>
                                        <div style="margin-top: 4px;">
                                            <span style="color: #22c55e; font-weight: 600;">Edge {rec_edge*100:+.1f}%</span>
                                            <span style="color: #4b5563;"> | </span>
                                            <span style="color: #a3e635;">EV {rec_ev*100:+.1f}%</span>
                                        </div>
                                        <div style="color: #9ca3af; font-size: 0.7rem; margin-top: 4px;">✓ 베팅 추천</div>
                                    </div>
                                ''', unsafe_allow_html=True)
                            else:
                                che, cae = custom_edge['edge_home'], custom_edge['edge_away']
                                st.markdown(f'''
                                    <div style="background: #1f293766; border: 1px solid #374151; border-radius: 10px; padding: 12px 16px; margin-top: 8px;">
                                        <div style="color: #6b7280; font-size: 0.85rem; text-align: center;">Edge 8% 미만 - 베팅 미권장</div>
                                        <div style="color: #4b5563; font-size: 0.7rem; text-align: center; margin-top: 2px;">{home_team} {che*100:+.1f}% | {away_team} {cae*100:+.1f}%</div>
                                    </div>
                                ''', unsafe_allow_html=True)

                return

        # 스프레드만 있는 경우
        if odds_info.get("spread_home") is not None:
            spread = odds_info["spread_home"]
            spread_text = f"{home_team} {spread:+.1f}" if spread < 0 else f"{away_team} {-spread:+.1f}"
            st.markdown(f'''
                <div style="background: #111827; border-radius: 8px; padding: 10px; margin-top: 8px; display: flex; justify-content: space-between;">
                    <span style="color: #6b7280; font-size: 0.8rem;">Pinnacle 스프레드</span>
                    <span style="color: #e5e7eb; font-weight: 600;">{spread_text}</span>
                </div>
            ''', unsafe_allow_html=True)
            return

    # 예정 경기 (배당 없음): 예측 스프레드
    if game_status == 1 and not odds_info:
        spread_text = f"{home_team} {-predicted_margin:+.1f}" if predicted_margin > 0 else f"{away_team} {predicted_margin:+.1f}"
        st.markdown(f'''
            <div style="background: #111827; border-radius: 8px; padding: 12px; margin-top: 8px; text-align: center;">
                <div style="color: #6b7280; font-size: 0.7rem;">예측 스프레드</div>
                <div style="color: #e5e7eb; font-weight: 700; font-size: 1.1rem;">{spread_text}</div>
            </div>
        ''', unsafe_allow_html=True)


def render_no_games() -> None:
    """경기 없음 메시지."""
    st.info("📅 해당 날짜에 예정된 경기가 없습니다.")


def render_day_summary(total: int, correct: int, mae: Optional[float] = None) -> None:
    """일일 요약 표시."""
    if total == 0:
        return

    accuracy = correct / total * 100
    acc_color = SUCCESS_COLOR if accuracy >= 50 else FAIL_COLOR

    mae_html = ""
    if mae is not None:
        mae_color = SUCCESS_COLOR if mae <= 10 else (LIVE_COLOR if mae <= 13 else FAIL_COLOR)
        mae_html = f'<div style="display: inline-block; background: #1e293b; border-radius: 8px; padding: 8px 16px; margin-left: 16px;"><div style="font-size: 0.7rem; color: #64748b;">평균 오차</div><div style="font-size: 1.2rem; font-weight: 700; color: {mae_color};">{mae:.1f}pt</div></div>'

    html = f'''<div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%); border: 1px solid #2d4a6f; border-radius: 12px; padding: 20px; margin: 20px 0; text-align: center;">
<div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">오늘의 예측 성과</div>
<div style="display: flex; justify-content: center; align-items: center;">
<div style="display: inline-block; background: #1e293b; border-radius: 8px; padding: 8px 16px;">
<div style="font-size: 0.7rem; color: #64748b;">적중률</div>
<div style="font-size: 1.8rem; font-weight: 800; color: {acc_color};">{accuracy:.1f}%</div>
</div>{mae_html}</div>
<div style="font-size: 0.85rem; color: #64748b; margin-top: 12px;">{total}경기 중 {correct}경기 적중</div>
</div>'''
    st.markdown(html, unsafe_allow_html=True)
