"""
enVocab - 英語学習アプリ
エントリポイント。TTS・スタイル・データ・タブは各モジュールに分割済み。
"""
import os
import streamlit as st
from dotenv import load_dotenv

from styles import load_custom_css
from tts import show_available_voices
from gemini_client import initialize_gemini
from data_loader import load_all_csv_data, load_word_master
from tabs import word_learning_tab, shadowing_tab, progress_tab, create_sample_data

load_dotenv()

st.set_page_config(
    page_title="英語学習アプリ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def initialize_session_state():
    """セッション状態の初期化"""
    if "current_sentence_idx" not in st.session_state:
        st.session_state.current_sentence_idx = 0
    if "learning_progress" not in st.session_state:
        st.session_state.learning_progress = {}
    if "show_translation" not in st.session_state:
        st.session_state.show_translation = False
    if "studied_today" not in st.session_state:
        st.session_state.studied_today = 0
    if "generated_content" not in st.session_state:
        st.session_state.generated_content = []
    if "current_shadowing_idx" not in st.session_state:
        st.session_state.current_shadowing_idx = 0
    if "show_shadowing_translation" not in st.session_state:
        st.session_state.show_shadowing_translation = False
    if "mobile_mode" not in st.session_state:
        st.session_state.mobile_mode = False
    if "audio_mode" not in st.session_state:
        st.session_state.audio_mode = "full"


def main():
    load_custom_css()
    initialize_session_state()
    initialize_gemini()

    with st.sidebar:
        st.markdown("## ⚙️ 詳細設定")

        with st.expander("🔑 API設定", expanded=False):
            env_api_key = os.getenv("GOOGLE_API_KEY", "")
            session_api_key = st.session_state.get("gemini_api_key", "")

            if env_api_key:
                st.success(f"✅ 環境変数: {env_api_key[:8]}...")
            else:
                st.warning("⚠️ 環境変数なし")

            if session_api_key:
                st.success(f"✅ セッション: {session_api_key[:8]}...")
            else:
                st.warning("⚠️ セッションなし")

            if st.button("🔄 API再読み込み"):
                initialize_gemini()
                st.rerun()

        with st.expander("🔊 音声設定", expanded=False):
            if st.button("🎤 利用可能な音声を確認"):
                show_available_voices()

            st.markdown("""
            **音声機能:**
            - 高品質サーバー生成音声（Piper 英国男性 / gTTS フォールバック）
            - 3段階の速度調整
            """)

        st.markdown("## 📊 データ状況")

    df = load_all_csv_data()
    word_master = load_word_master()

    if df.empty:
        st.error("📁 CSVファイルが見つかりません。'data'フォルダにgroup*.csvファイルを配置してください。")
        if st.button("🔧 サンプルデータを作成"):
            create_sample_data()
            st.rerun()
        return

    with st.sidebar:
        st.markdown(f"**📈 統計:** {len(df)}文 / {df['group_id'].nunique()}グループ")
        st.markdown(f"**📚 今日:** {st.session_state.studied_today}文章学習")

    tab1, tab2, tab3 = st.tabs(["📚 学習", "🎯 シャドーイング", "📊 記録"])

    with tab1:
        word_learning_tab(df, word_master)
    with tab2:
        shadowing_tab()
    with tab3:
        progress_tab(df)


if __name__ == "__main__":
    main()
