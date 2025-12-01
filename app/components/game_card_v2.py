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

    if edge_home > 0.03:
        bet_side = 'home'
    elif edge_away > 0.03:
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
) -> None:
    """순수 Streamlit 네이티브 게임 카드 렌더링."""

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
        bg_gradient = f"linear-gradient(145deg, {'#1a2e1a' if is_correct else '#2e1a1a'} 0%, #161b26 100%)"
        result_text = "✓ 적중" if is_correct else "✗ 실패"
        result_bg = SUCCESS_COLOR if is_correct else FAIL_COLOR
    elif is_live:
        border_color = LIVE_COLOR
        bg_gradient = "linear-gradient(145deg, #2a2517 0%, #161b26 100%)"
        result_text = None
        result_bg = None
    else:
        border_color = "#2d3748"
        bg_gradient = "linear-gradient(145deg, #1e2433 0%, #161b26 100%)"
        result_text = None
        result_bg = None

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
    card_style = f"border: 2px solid {border_color}; border-radius: 16px; padding: 20px; margin: 12px 0; background: {bg_gradient};"

    st.markdown(f'<div style="{card_style}">', unsafe_allow_html=True)

    # 결과 배지 (종료된 경기)
    if result_text:
        st.markdown(f'''
            <div style="text-align: right; margin-bottom: -10px;">
                <span style="background: {result_bg}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.75rem; font-weight: 700;">{result_text}</span>
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
            score_color = "#ffffff" if (is_finished and actual_home_win) or (is_live and home_score > away_score) else "#6b7280"
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
            score_color = "#ffffff" if (is_finished and not actual_home_win) or (is_live and away_score > home_score) else "#6b7280"
            st.markdown(f'<div style="text-align: center; font-size: 2rem; font-weight: 800; color: {score_color}; margin-top: 8px;">{away_score}</div>', unsafe_allow_html=True)

        # 확률
        prob_opacity = "0.7" if is_finished else "1"
        prob_size = "1.2rem" if (is_finished or is_live) else "1.6rem"
        st.markdown(f'<div style="text-align: center; font-size: {prob_size}; font-weight: 700; color: {AWAY_COLOR}; margin-top: 8px; opacity: {prob_opacity};">{final_away_prob:.1%}</div>', unsafe_allow_html=True)

    # 확률 바
    st.markdown(f'''
        <div style="height: 12px; background: #1f2937; border-radius: 6px; overflow: hidden; display: flex; margin: 16px 0;">
            <div style="width: {home_pct:.1f}%; height: 100%; background: linear-gradient(90deg, {HOME_COLOR}cc, {HOME_COLOR});"></div>
            <div style="width: {away_pct:.1f}%; height: 100%; background: linear-gradient(90deg, {AWAY_COLOR}, {AWAY_COLOR}cc);"></div>
        </div>
    ''', unsafe_allow_html=True)

    # 하단 섹션 렌더링
    _render_bottom_section(
        home_team, away_team, final_home_prob, final_away_prob,
        predicted_margin, adjusted_margin, game_status, is_finished, show_result,
        home_score, away_score, actual_home_win, odds_info,
        game_id, enable_custom_input
    )

    # 카드 종료
    st.markdown('</div>', unsafe_allow_html=True)


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
                    <div style="color: #fff; font-weight: 600;">{actual_text}</div>
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
                    <div style="background: #111827; border-radius: 10px; padding: 14px; margin-top: 8px;">
                        <div style="color: #f59e0b; font-size: 0.85rem; font-weight: 700; margin-bottom: 12px;">💰 모델 Edge (vs Pinnacle)</div>
                        <table style="width: 100%; font-size: 0.85rem; border-collapse: collapse;">
                            <tr style="color: #9ca3af; font-size: 0.75rem; font-weight: 600;">
                                <td style="padding: 8px 4px;">팀</td>
                                <td style="text-align: center; padding: 8px 4px;">🤖 모델</td>
                                <td style="text-align: center; padding: 8px 4px;">📊 시장</td>
                                <td style="text-align: center; padding: 8px 4px;">Edge</td>
                                <td style="text-align: center; padding: 8px 4px;">EV</td>
                            </tr>
                            <tr style="border-top: 1px solid #374151;">
                                <td style="color: {HOME_COLOR}; font-weight: 700; padding: 10px 4px; font-size: 0.9rem;">{home_team} <span style="color: #6b7280; font-size: 0.75rem;">@{ml_home:.2f}</span></td>
                                <td style="text-align: center; color: #ffffff; font-weight: 700; padding: 10px 4px; font-size: 1rem;">{final_home_prob*100:.1f}%</td>
                                <td style="text-align: center; color: #fbbf24; font-weight: 600; padding: 10px 4px; font-size: 1rem;">{hp*100:.1f}%</td>
                                <td style="text-align: center; color: {edge_color(he)}; font-weight: 800; padding: 10px 4px; font-size: 1rem;">{he*100:+.1f}%</td>
                                <td style="text-align: center; color: {ev_color(hev)}; font-weight: 700; padding: 10px 4px; font-size: 0.9rem;">{hev*100:+.1f}%</td>
                            </tr>
                            <tr style="border-top: 1px solid #374151;">
                                <td style="color: {AWAY_COLOR}; font-weight: 700; padding: 10px 4px; font-size: 0.9rem;">{away_team} <span style="color: #6b7280; font-size: 0.75rem;">@{ml_away:.2f}</span></td>
                                <td style="text-align: center; color: #ffffff; font-weight: 700; padding: 10px 4px; font-size: 1rem;">{final_away_prob*100:.1f}%</td>
                                <td style="text-align: center; color: #fbbf24; font-weight: 600; padding: 10px 4px; font-size: 1rem;">{ap*100:.1f}%</td>
                                <td style="text-align: center; color: {edge_color(ae)}; font-weight: 800; padding: 10px 4px; font-size: 1rem;">{ae*100:+.1f}%</td>
                                <td style="text-align: center; color: {ev_color(aev)}; font-weight: 700; padding: 10px 4px; font-size: 0.9rem;">{aev*100:+.1f}%</td>
                            </tr>
                        </table>
                    </div>
                ''', unsafe_allow_html=True)

                # 커스텀 입력 섹션 (카드 내부에 통합!)
                if enable_custom_input and game_id:
                    st.markdown('''
                        <div style="background: #0d1117; border: 1px dashed #374151; border-radius: 10px; padding: 14px; margin-top: 12px;">
                            <div style="color: #f59e0b; font-size: 0.85rem; font-weight: 600; margin-bottom: 8px;">🎯 내 확률로 Edge 계산</div>
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
                            if custom_edge['edge_home'] > 0.03:
                                rec_team = home_team
                                rec_edge = custom_edge['edge_home']
                                rec_ev = custom_edge['ev_home']
                                rec_odds = ml_home
                                team_color = HOME_COLOR
                                box_bg = f"linear-gradient(135deg, rgba(59, 130, 246, 0.15), rgba(59, 130, 246, 0.05))"
                                box_border = f"1px solid rgba(59, 130, 246, 0.4)"
                            elif custom_edge['edge_away'] > 0.03:
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
                                        <div style="color: #6b7280; font-size: 0.85rem; text-align: center;">Edge 3% 미만 - 베팅 미권장</div>
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
        mae_html = f'''
            <div style="display: inline-block; background: #1e293b; border-radius: 8px; padding: 8px 16px; margin-left: 16px;">
                <div style="font-size: 0.7rem; color: #64748b;">평균 오차</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {mae_color};">{mae:.1f}pt</div>
            </div>
        '''

    st.markdown(f'''
        <div style="background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%); border: 1px solid #2d4a6f; border-radius: 12px; padding: 20px; margin: 20px 0; text-align: center;">
            <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">오늘의 예측 성과</div>
            <div style="display: flex; justify-content: center; align-items: center;">
                <div style="display: inline-block; background: #1e293b; border-radius: 8px; padding: 8px 16px;">
                    <div style="font-size: 0.7rem; color: #64748b;">적중률</div>
                    <div style="font-size: 1.8rem; font-weight: 800; color: {acc_color};">{accuracy:.1f}%</div>
                </div>
                {mae_html}
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-top: 12px;">{total}경기 중 {correct}경기 적중</div>
        </div>
    ''', unsafe_allow_html=True)
