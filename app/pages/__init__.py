"""
페이지 렌더러 모듈.

리팩토링 Phase 4: main.py에서 페이지 렌더링 로직 추출.
"""

from app.pages.predictions_page import render_predictions_page
from app.pages.paper_betting_page import render_paper_betting_page

__all__ = [
    'render_predictions_page',
    'render_paper_betting_page',
]
