"""
경기 카드 컴포넌트.

단일 경기 예측 결과를 카드 형태로 표시합니다.
"""

import streamlit as st
from streamlit.components.v1 import html
from typing import Dict, List, Optional


# 색상 상수
HOME_COLOR = "#3b82f6"  # 파란색
AWAY_COLOR = "#ef4444"  # 빨간색
SUCCESS_COLOR = "#22c55e"  # 녹색 (적중)
FAIL_COLOR = "#ef4444"  # 빨간색 (실패)
EDGE_POSITIVE_COLOR = "#22c55e"  # 녹색 (양의 Edge)
EDGE_NEGATIVE_COLOR = "#f59e0b"  # 주황색 (음의 Edge)


def _render_market_line(
    odds_info: Optional[Dict],
    predicted_margin: float,
    home_team: str,
    away_team: str,
) -> str:
    """시장 배당 라인 렌더링"""
    if not odds_info or odds_info.get("spread_home") is None:
        return ""

    spread_home = odds_info["spread_home"]
    bookmaker = odds_info.get("bookmaker", "").upper()

    # 모델 예측 vs 시장 라인 비교
    # predicted_margin > 0: 홈팀 우세
    # spread_home < 0: 홈팀이 핸디캡 극복해야 (홈팀 우세 예상)
    model_spread = -predicted_margin  # 모델 예측을 스프레드 형식으로 변환

    # Edge 계산 (모델 스프레드 - 시장 스프레드)
    edge = model_spread - spread_home

    # 스프레드 표시 (예: HOU -11.5)
    if spread_home < 0:
        spread_text = f"{home_team} {spread_home:+.1f}"
    else:
        spread_text = f"{away_team} {-spread_home:+.1f}"

    # Edge 색상 및 표시
    if abs(edge) < 1.0:
        edge_color = "#9ca3af"  # 회색 (중립)
        edge_label = "시장과 일치"
    elif edge > 0:
        edge_color = EDGE_POSITIVE_COLOR
        edge_label = f"Edge +{edge:.1f}점"
    else:
        edge_color = EDGE_NEGATIVE_COLOR
        edge_label = f"Edge {edge:.1f}점"

    return f'''
    <!-- 시장 라인 -->
    <div style="
        margin-top: 16px;
        padding: 12px 16px;
        background: #1f2937;
        border-radius: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    ">
        <div style="display: flex; align-items: center; gap: 8px;">
            <span style="color: #6b7280; font-size: 0.75rem;">시장 라인</span>
            <span style="color: #ffffff; font-weight: 600; font-size: 0.9rem;">{spread_text}</span>
            <span style="color: #4b5563; font-size: 0.65rem;">({bookmaker})</span>
        </div>
        <div style="
            background: {edge_color}22;
            color: {edge_color};
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
        ">{edge_label}</div>
    </div>
    '''


def _render_prediction_detail(
    predicted_margin: float,
    adjusted_margin: Optional[float],
    home_team: str,
    away_team: str,
    home_injuries: List[Dict],
    away_injuries: List[Dict],
    home_injury_impact: float,
    away_injury_impact: float,
    home_score: Optional[int],
    away_score: Optional[int],
    is_finished: bool,
    show_result: bool,
    odds_info: Optional[Dict] = None,
) -> str:
    """예측 상세 섹션 렌더링 (카드 내부)"""

    # 예측 마진 정보
    margin_team = home_team if predicted_margin > 0 else away_team
    margin_sign = "+" if predicted_margin > 0 else ""

    # 부상 정보 HTML
    injury_html = ""
    has_injury = (home_injuries or away_injuries) and adjusted_margin is not None

    if has_injury:
        home_inj_text = ""
        away_inj_text = ""

        if home_injuries:
            names = ", ".join([f"{p['name']}" for p in home_injuries[:2]])
            if len(home_injuries) > 2:
                names += f" +{len(home_injuries)-2}"
            home_inj_text = f'''
                <div style="display: flex; align-items: center; gap: 6px; color: #f87171;">
                    <span style="font-size: 0.8rem;">{home_team}</span>
                    <span style="font-size: 0.7rem; color: #9ca3af;">{names}</span>
                    <span style="font-size: 0.7rem; font-weight: 600;">({home_injury_impact:+.1f}pt)</span>
                </div>
            '''

        if away_injuries:
            names = ", ".join([f"{p['name']}" for p in away_injuries[:2]])
            if len(away_injuries) > 2:
                names += f" +{len(away_injuries)-2}"
            away_inj_text = f'''
                <div style="display: flex; align-items: center; gap: 6px; color: #f87171;">
                    <span style="font-size: 0.8rem;">{away_team}</span>
                    <span style="font-size: 0.7rem; color: #9ca3af;">{names}</span>
                    <span style="font-size: 0.7rem; font-weight: 600;">({away_injury_impact:+.1f}pt)</span>
                </div>
            '''

        injury_html = f'''
            <div style="
                display: flex;
                flex-direction: column;
                gap: 4px;
                padding: 8px 12px;
                background: #7f1d1d22;
                border-radius: 8px;
                border-left: 3px solid #dc2626;
            ">
                <div style="font-size: 0.65rem; color: #6b7280; margin-bottom: 2px;">부상 결장</div>
                {home_inj_text}
                {away_inj_text}
            </div>
        '''

    # 종료된 경기: 오차 비교
    comparison_html = ""
    if is_finished and show_result and home_score is not None and away_score is not None:
        actual_margin = home_score - away_score
        final_predicted = adjusted_margin if adjusted_margin is not None else predicted_margin
        error = abs(final_predicted - actual_margin)

        actual_team = home_team if actual_margin > 0 else away_team
        actual_sign = "+" if actual_margin > 0 else ""

        pred_team = home_team if final_predicted > 0 else away_team
        pred_sign = "+" if final_predicted > 0 else ""

        # 오차 수준
        if error <= 5:
            error_color = "#22c55e"
        elif error <= 10:
            error_color = "#eab308"
        else:
            error_color = "#ef4444"

        comparison_html = f'''
            <div style="
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 12px;
                background: #111827;
                border-radius: 8px;
            ">
                <div style="text-align: center; flex: 1;">
                    <div style="color: #6b7280; font-size: 0.65rem;">예측</div>
                    <div style="color: #9ca3af; font-weight: 600; font-size: 0.85rem;">
                        {pred_team} {pred_sign}{final_predicted:.1f}
                    </div>
                </div>
                <div style="text-align: center; flex: 0 0 70px;">
                    <div style="color: #6b7280; font-size: 0.65rem;">오차</div>
                    <div style="color: {error_color}; font-weight: 700; font-size: 1rem;">
                        {error:.1f}pt
                    </div>
                </div>
                <div style="text-align: center; flex: 1;">
                    <div style="color: #6b7280; font-size: 0.65rem;">실제</div>
                    <div style="color: #fff; font-weight: 600; font-size: 0.85rem;">
                        {actual_team} {actual_sign}{actual_margin}
                    </div>
                </div>
            </div>
        '''
    elif not is_finished:
        # 예정 경기: 예측 + 시장 라인 + Edge
        if odds_info and odds_info.get("spread_home") is not None:
            spread_home = odds_info["spread_home"]
            bookmaker = odds_info.get("bookmaker", "").upper()

            # 모델 예측을 스프레드 형식으로 변환 (우세팀에 음수)
            # predicted_margin > 0: 홈팀 우세 → 홈팀 스프레드 음수
            model_spread = -predicted_margin
            if predicted_margin > 0:
                model_text = f"{home_team} {model_spread:+.1f}"
            else:
                model_text = f"{away_team} {-model_spread:+.1f}"

            # 시장 라인 텍스트 (예: HOU -12.5)
            if spread_home < 0:
                market_text = f"{home_team} {spread_home:+.1f}"
            else:
                market_text = f"{away_team} {-spread_home:+.1f}"

            # Edge 계산
            # spread_home: +면 홈팀 언더독, -면 홈팀 페이버릿
            # 시장 예측 마진 ≈ -spread_home이므로
            # edge = |predicted_margin - (-spread_home)| = |predicted_margin + spread_home|
            edge = abs(predicted_margin + spread_home)

            if edge < 1.0:
                edge_color = "#9ca3af"
                edge_text = "일치"
            else:
                edge_color = EDGE_POSITIVE_COLOR
                edge_text = f"{edge:.1f}"

            comparison_html = f'''
                <div style="
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 10px 12px;
                    background: #111827;
                    border-radius: 8px;
                ">
                    <div style="text-align: center; flex: 1;">
                        <div style="color: #6b7280; font-size: 0.65rem;">모델</div>
                        <div style="color: #e5e7eb; font-weight: 600; font-size: 0.85rem;">
                            {model_text}
                        </div>
                    </div>
                    <div style="text-align: center; flex: 0 0 70px;">
                        <div style="color: #6b7280; font-size: 0.65rem;">Edge</div>
                        <div style="color: {edge_color}; font-weight: 700; font-size: 1rem;">
                            {edge_text}
                        </div>
                    </div>
                    <div style="text-align: center; flex: 1;">
                        <div style="color: #6b7280; font-size: 0.65rem;">시장<span style="color:#4b5563; font-size:0.55rem;"> ({bookmaker})</span></div>
                        <div style="color: #9ca3af; font-weight: 600; font-size: 0.85rem;">
                            {market_text}
                        </div>
                    </div>
                </div>
            '''
        else:
            # 시장 라인 없으면 예측만 표시
            comparison_html = f'''
                <div style="
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    padding: 10px 12px;
                    background: #111827;
                    border-radius: 8px;
                ">
                    <div style="text-align: center;">
                        <div style="color: #6b7280; font-size: 0.65rem; margin-bottom: 2px;">예측 점수차</div>
                        <div style="color: #e5e7eb; font-weight: 700; font-size: 1.1rem;">
                            {margin_team} {margin_sign}{predicted_margin:.1f}pt
                        </div>
                    </div>
                </div>
            '''

    # 전체 상세 섹션 조합
    return f'''
        <div style="
            margin-top: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        ">
            {comparison_html}
            {injury_html}
        </div>
    '''


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
    home_injuries: Optional[List[Dict]] = None,
    away_injuries: Optional[List[Dict]] = None,
    home_injury_impact: float = 0.0,
    away_injury_impact: float = 0.0,
    # 경기 결과 (종료된 경기)
    game_status: int = 1,  # 1=예정, 2=진행중, 3=종료
    home_score: Optional[int] = None,
    away_score: Optional[int] = None,
    # B2B 정보
    home_b2b: bool = False,
    away_b2b: bool = False,
    # 적중 여부 숨기기 (오늘 경기용)
    hide_result: bool = False,
    # 배당 정보
    odds_info: Optional[Dict] = None,
) -> None:
    """
    경기 카드 렌더링.
    """
    home_injuries = home_injuries or []
    away_injuries = away_injuries or []

    # 최종 승률 결정 (부상 조정이 있으면 적용)
    final_home_prob = adjusted_win_prob if adjusted_win_prob is not None else home_win_prob
    final_away_prob = 1 - final_home_prob

    # 퍼센트 값
    home_pct = final_home_prob * 100
    away_pct = final_away_prob * 100

    # 예측 승자
    predicted_home_win = final_home_prob >= 0.5

    # 경기 종료 여부 및 적중 여부
    # hide_result=True면 적중 여부를 숨김 (오늘/라이브 경기용)
    is_finished = game_status == 3 and home_score is not None and away_score is not None
    show_result = is_finished and not hide_result  # 적중 배지 표시 여부
    actual_home_win = None
    is_correct = None

    if is_finished:
        actual_home_win = home_score > away_score
        is_correct = predicted_home_win == actual_home_win

    # 결과 배지 HTML (show_result가 True일 때만 표시)
    if show_result:
        if is_correct:
            result_badge = f'''
            <div style="
                position: absolute;
                top: 12px;
                right: 12px;
                background: {SUCCESS_COLOR};
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
            ">✓ 적중</div>
            '''
        else:
            result_badge = f'''
            <div style="
                position: absolute;
                top: 12px;
                right: 12px;
                background: {FAIL_COLOR};
                color: white;
                padding: 4px 12px;
                border-radius: 12px;
                font-size: 0.75rem;
                font-weight: 700;
            ">✗ 실패</div>
            '''
    else:
        result_badge = ""

    # Live 경기 여부
    is_live = game_status == 2 and home_score is not None and away_score is not None

    # 점수 표시 (종료/진행중 경기)
    if is_finished:
        home_score_html = f'''
        <div style="font-size: 1.8rem; font-weight: 800; color: {'#ffffff' if actual_home_win else '#6b7280'}; margin-top: 8px;">
            {home_score}
        </div>
        '''
        away_score_html = f'''
        <div style="font-size: 1.8rem; font-weight: 800; color: {'#ffffff' if not actual_home_win else '#6b7280'}; margin-top: 8px;">
            {away_score}
        </div>
        '''
        prob_label = "예측"
    elif is_live:
        # Live 경기: 현재 점수 표시 (깜빡임 효과)
        home_leading = home_score > away_score
        home_score_html = f'''
        <div style="font-size: 1.6rem; font-weight: 700; color: {'#ef4444' if home_leading else '#9ca3af'}; margin-top: 8px;">
            {home_score}
        </div>
        '''
        away_score_html = f'''
        <div style="font-size: 1.6rem; font-weight: 700; color: {'#ef4444' if not home_leading else '#9ca3af'}; margin-top: 8px;">
            {away_score}
        </div>
        '''
        prob_label = "예측"
    else:
        home_score_html = ""
        away_score_html = ""
        prob_label = ""

    # 경기 상태 표시
    if game_status == 3:
        status_text = "Final"
        status_color = "#6b7280"
    elif game_status == 2:
        status_text = "Live"
        status_color = "#ef4444"
    else:
        status_text = game_time
        status_color = "#9ca3af"

    # 카드 스타일 (경기 상태/결과에 따라 다르게)
    if show_result:
        if is_correct:
            # 적중: 녹색 테두리 + 어두운 녹색 배경
            border_color = SUCCESS_COLOR
            bg_gradient = "linear-gradient(145deg, #1a2e1a 0%, #142014 100%)"
            box_shadow = f"0 4px 20px rgba(34, 197, 94, 0.15), inset 0 0 0 1px {SUCCESS_COLOR}33"
        else:
            # 실패: 빨간 테두리 + 어두운 빨간 배경
            border_color = FAIL_COLOR
            bg_gradient = "linear-gradient(145deg, #2e1a1a 0%, #201414 100%)"
            box_shadow = f"0 4px 20px rgba(239, 68, 68, 0.15), inset 0 0 0 1px {FAIL_COLOR}33"
    elif game_status == 2:
        # Live: 빨간 펄스 테두리
        border_color = "#ef4444"
        bg_gradient = "linear-gradient(145deg, #1e2433 0%, #161b26 100%)"
        box_shadow = "0 4px 20px rgba(239, 68, 68, 0.2)"
    else:
        # 예정: 기본 스타일
        border_color = "#2d3748"
        bg_gradient = "linear-gradient(145deg, #1e2433 0%, #161b26 100%)"
        box_shadow = "0 4px 20px rgba(0,0,0,0.3)"

    # 카드 HTML
    card_html = f"""
    <div style="
        position: relative;
        border: 2px solid {border_color};
        border-radius: 16px;
        padding: 28px;
        margin: 16px 0;
        background: {bg_gradient};
        box-shadow: {box_shadow};
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        {result_badge}

        <!-- 경기 시간/상태 -->
        <div style="text-align: center; margin-bottom: 20px;">
            <span style="
                background: #374151;
                color: {status_color};
                padding: 6px 16px;
                border-radius: 20px;
                font-size: 0.85rem;
                font-weight: {'700' if game_status == 2 else '400'};
            ">{status_text}</span>
        </div>

        <!-- 팀 정보 + 점수/승률 -->
        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px;">
            <!-- 홈팀 -->
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 800; color: {HOME_COLOR};">{home_team}</div>
                <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">{home_name}</div>
                <div style="
                    display: inline-block;
                    background: {HOME_COLOR}22;
                    color: {HOME_COLOR};
                    padding: 2px 10px;
                    border-radius: 12px;
                    font-size: 0.7rem;
                    font-weight: 600;
                    margin-top: 6px;
                ">HOME</div>
                {'<div style="display: inline-block; background: #f59e0b33; color: #f59e0b; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 700; margin-left: 4px;" title="Back-to-Back">B2B</div>' if home_b2b else ''}
                {home_score_html}
                <div style="font-size: {'1.2rem' if (is_finished or is_live) else '2rem'}; font-weight: 700; color: {HOME_COLOR}; margin-top: {'4px' if (is_finished or is_live) else '12px'}; opacity: {'0.7' if is_finished else '1'};">
                    {prob_label} {final_home_prob:.1%}
                </div>
            </div>

            <!-- VS -->
            <div style="flex: 0 0 60px; text-align: center;">
                <div style="font-size: 1.1rem; color: #4b5563; font-weight: 600;">VS</div>
            </div>

            <!-- 원정팀 -->
            <div style="flex: 1; text-align: center;">
                <div style="font-size: 2.2rem; font-weight: 800; color: {AWAY_COLOR};">{away_team}</div>
                <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 4px;">{away_name}</div>
                <div style="
                    display: inline-block;
                    background: {AWAY_COLOR}22;
                    color: {AWAY_COLOR};
                    padding: 2px 10px;
                    border-radius: 12px;
                    font-size: 0.7rem;
                    font-weight: 600;
                    margin-top: 6px;
                ">AWAY</div>
                {'<div style="display: inline-block; background: #f59e0b33; color: #f59e0b; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 700; margin-left: 4px;" title="Back-to-Back">B2B</div>' if away_b2b else ''}
                {away_score_html}
                <div style="font-size: {'1.2rem' if (is_finished or is_live) else '2rem'}; font-weight: 700; color: {AWAY_COLOR}; margin-top: {'4px' if (is_finished or is_live) else '12px'}; opacity: {'0.7' if is_finished else '1'};">
                    {prob_label} {final_away_prob:.1%}
                </div>
            </div>
        </div>

        <!-- 확률 바 -->
        <div style="
            width: 100%;
            height: 12px;
            background: #1f2937;
            border-radius: 6px;
            overflow: hidden;
            display: flex;
        ">
            <div style="
                width: {home_pct:.1f}%;
                height: 100%;
                background: linear-gradient(90deg, {HOME_COLOR}cc, {HOME_COLOR});
            "></div>
            <div style="
                width: {away_pct:.1f}%;
                height: 100%;
                background: linear-gradient(90deg, {AWAY_COLOR}, {AWAY_COLOR}cc);
            "></div>
        </div>
        {_render_prediction_detail(predicted_margin, adjusted_margin, home_team, away_team, home_injuries, away_injuries, home_injury_impact, away_injury_impact, home_score, away_score, is_finished, show_result, odds_info)}
    </div>
    """

    # HTML 컴포넌트로 렌더링
    # 높이 계산: 기본 + 상세 섹션
    has_injury = (home_injuries or away_injuries) and adjusted_margin is not None
    injury_height = 60 if has_injury else 0

    if is_finished:
        # 종료: 점수 + 오차 비교
        card_height = 400 + injury_height
    elif is_live:
        # Live: 점수 표시 + 예측 점수차
        card_height = 380 + injury_height
    else:
        # 예정: 예측 점수차 + 시장 라인 (통합)
        card_height = 340 + injury_height

    html(card_html, height=card_height)


def render_no_games() -> None:
    """경기 없음 표시"""
    st.info("📅 해당 날짜에 예정된 경기가 없습니다.")


def render_day_summary(total: int, correct: int, mae: Optional[float] = None) -> None:
    """일별 예측 요약 (적중률 + MAE)"""
    if total == 0:
        return

    accuracy = correct / total * 100

    # MAE 표시 (있을 경우)
    mae_html = ""
    if mae is not None:
        # MAE 수준에 따른 색상
        if mae <= 10:
            mae_color = "#22c55e"  # 녹색
        elif mae <= 13:
            mae_color = "#eab308"  # 노란색
        else:
            mae_color = "#ef4444"  # 빨간색

        mae_html = f'''
            <div style="
                display: inline-block;
                background: #1e293b;
                border-radius: 8px;
                padding: 8px 16px;
                margin-left: 16px;
            ">
                <div style="font-size: 0.7rem; color: #64748b;">평균 오차</div>
                <div style="font-size: 1.2rem; font-weight: 700; color: {mae_color};">{mae:.1f}pt</div>
            </div>
        '''

    summary_html = f"""
    <div style="
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
        border: 1px solid #2d4a6f;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    ">
        <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;">오늘의 예측 성과</div>
        <div style="display: flex; justify-content: center; align-items: center;">
            <div style="
                display: inline-block;
                background: #1e293b;
                border-radius: 8px;
                padding: 8px 16px;
            ">
                <div style="font-size: 0.7rem; color: #64748b;">적중률</div>
                <div style="font-size: 1.8rem; font-weight: 800; color: {'#22c55e' if accuracy >= 50 else '#ef4444'};">
                    {accuracy:.1f}%
                </div>
            </div>
            {mae_html}
        </div>
        <div style="font-size: 0.85rem; color: #64748b; margin-top: 12px;">
            {total}경기 중 {correct}경기 적중
        </div>
    </div>
    """
    html(summary_html, height=160)


def render_loading() -> None:
    """로딩 표시"""
    st.spinner("데이터를 불러오는 중...")
