import streamlit as st


def load_custom_css():
    """カスタムCSS（iPhone SE向けモバイル最適化）を読み込み"""
    st.markdown("""
    <style>
    /* ========== 基本スタイル ========== */
    .main-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
    }

    .progress-simple {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 500;
        color: #1a1a1a;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        letter-spacing: 0.02em;
    }

    .vocab-highlight {
        background: linear-gradient(180deg, transparent 60%, #ffd54f 60%) !important;
        color: #000 !important;
        font-weight: 500 !important;
        padding: 0 2px !important;
        border-radius: 0 !important;
        border: none !important;
    }

    .japanese-highlight {
        background: linear-gradient(180deg, transparent 60%, #a5d6a7 60%) !important;
        color: #000 !important;
        font-weight: 500 !important;
        padding: 0 2px !important;
        border-radius: 0 !important;
        border: none !important;
    }

    .word-chip {
        background-color: #1a1a1a;
        color: #fafafa;
        padding: 0.25rem 0.6rem;
        border-radius: 2px;
        font-size: 0.8rem;
        margin: 0.15rem;
        display: inline-block;
        font-family: 'Source Serif 4', Georgia, serif;
        letter-spacing: 0.01em;
    }

    .big-nav-button {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2rem;
        padding: 1rem;
        min-height: 60px;
        border-radius: 4px;
        cursor: pointer;
        user-select: none;
        transition: transform 0.1s, background-color 0.2s;
    }

    .big-nav-button:active { transform: scale(0.95); }

    .understanding-row {
        display: flex;
        justify-content: space-around;
        gap: 0.5rem;
        padding: 0.5rem 0;
    }

    .understanding-btn {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem 0.5rem;
        border-radius: 4px;
        cursor: pointer;
        min-height: 70px;
        font-size: 1.8rem;
        transition: transform 0.1s;
    }

    .understanding-btn:active { transform: scale(0.95); }
    .understanding-btn .label { font-size: 0.8rem; margin-top: 0.3rem; }

    .audio-button-center {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2.5rem;
        padding: 1rem;
        cursor: pointer;
    }

    @media (max-width: 400px) {
        [data-testid="stHorizontalBlock"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 0.5rem !important;
        }
        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 !important;
            min-width: 0 !important;
            width: auto !important;
        }
        [data-testid="stHorizontalBlock"] .stButton > button {
            min-height: 48px !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 0.25rem !important;
            white-space: nowrap !important;
        }
    }

    .stButton > button {
        min-height: 48px !important;
        font-size: 1rem !important;
        border-radius: 4px !important;
        padding: 0.75rem 1rem !important;
        font-family: -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
        border: 1px solid #e0e0e0 !important;
        background: #fafafa !important;
        color: #1a1a1a !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        background: #f0f0f0 !important;
        border-color: #ccc !important;
    }

    .stButton > button:active {
        background: #e8e8e8 !important;
        transform: scale(0.98);
    }

    [data-testid="stHorizontalBlock"] .stButton > button {
        min-height: 56px !important;
        font-size: 1.1rem !important;
    }

    .sentence-card {
        background-color: #fafafa !important;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 2px solid #1a1a1a;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }

    .sentence-card p, .sentence-card h3, .sentence-card h4 { color: #1a1a1a !important; }

    .translation-card {
        background-color: #f5f5f5 !important;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        color: #333 !important;
    }

    .translation-card p, .translation-card h4 { color: #333 !important; }

    [data-theme="dark"] .sentence-card {
        background-color: #2a2a2a;
        color: #fafafa;
        border-left: 2px solid #fafafa;
    }

    [data-theme="dark"] .translation-card {
        background-color: #333;
        color: #fafafa;
        border: 1px solid #444;
    }

    [data-theme="dark"] .progress-simple { color: #fafafa; }
    [data-theme="dark"] .word-chip {
        background-color: #fafafa;
        color: #1a1a1a;
    }

    [data-theme="dark"] .stButton > button {
        background: #2a2a2a !important;
        color: #fafafa !important;
        border-color: #444 !important;
    }

    [data-theme="dark"] .stButton > button:hover { background: #333 !important; }

    .progress-text { font-size: 1rem; font-weight: 500; }

    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)
