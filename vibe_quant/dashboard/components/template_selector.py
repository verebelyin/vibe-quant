"""Strategy template selector component.

Card grid showing categorized templates with:
- Category tabs
- Template cards with difficulty badges, descriptions
- One-click "Use Template" that populates the editor
"""

from __future__ import annotations

import streamlit as st

from vibe_quant.strategies.templates._metadata import (
    TemplateMeta,
    get_templates_by_category,
)

_DIFFICULTY_COLORS = {
    "Beginner": "green",
    "Intermediate": "orange",
    "Advanced": "red",
}

# Short labels to prevent badge text wrapping in narrow columns
_DIFFICULTY_SHORT = {
    "Beginner": "Beginner",
    "Intermediate": "Mid",
    "Advanced": "Advanced",
}

_CATEGORY_ICONS = {
    "Momentum": ":material/speed:",
    "Trend": ":material/trending_up:",
    "Volatility": ":material/bolt:",
    "Multi-Timeframe": ":material/layers:",
    "Volume": ":material/bar_chart:",
}


def render_template_selector(key_prefix: str = "tmpl") -> str | None:
    """Render the template selector and return the YAML content if selected.

    Returns None if no template was selected, or the YAML string if one was.
    Returns empty string "" if Cancel/Back was clicked (caller should handle).
    """
    col_title, col_cancel = st.columns([4, 1])
    with col_title:
        st.markdown("#### Start from a template")
    with col_cancel:
        if st.button("Cancel", key=f"{key_prefix}_cancel", type="secondary"):
            return ""

    st.caption("Choose a proven strategy template and customize it, or start from scratch.")

    by_category = get_templates_by_category()

    # Create tabs for each category
    tab_names = [
        f"{_CATEGORY_ICONS.get(cat, '')} {cat}" for cat in by_category
    ]
    tabs = st.tabs(tab_names)

    for tab, (_cat, templates) in zip(tabs, by_category.items(), strict=False):
        with tab:
            # Render templates in a 2-column grid
            for i in range(0, len(templates), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx >= len(templates):
                        break
                    tmpl = templates[idx]
                    with col:
                        result = _render_template_card(tmpl, f"{key_prefix}_{_cat}_{idx}")
                        if result is not None:
                            return result

    return None


def _render_template_card(tmpl: TemplateMeta, key: str) -> str | None:
    """Render a single template card. Returns YAML if selected."""
    diff_color = _DIFFICULTY_COLORS.get(tmpl.difficulty, "gray")

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{tmpl.display_name}**")
        with col2:
            short_label = _DIFFICULTY_SHORT.get(tmpl.difficulty, tmpl.difficulty)
            st.markdown(f":{diff_color}[{short_label}]")

        st.caption(tmpl.description)

        detail_col, btn_col = st.columns([3, 1])
        with detail_col:
            st.caption(f"**Markets:** {tmpl.market_conditions[:60]}")
            st.caption(f"**Instruments:** {tmpl.instruments}")
        with btn_col:
            if st.button("Use", key=f"{key}_use", type="primary", width="stretch"):
                try:
                    return tmpl.load_yaml()
                except FileNotFoundError:
                    st.error(f"Template file not found: {tmpl.file_name}")
                    return None

    return None
