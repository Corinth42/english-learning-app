import os
import random
import pandas as pd
import streamlit as st

from config import GENRE_PROMPTS
from data_loader import load_all_csv_data, load_word_master, parse_words_dict
from tts import play_server_generated_audio, show_available_voices
from components import create_flip_card
from gemini_client import initialize_gemini, generate_content_with_gemini, parse_generated_content


def word_learning_tab(df, word_master):
    """単語学習タブ - iPhone SE向けフリップカードUI"""
    with st.expander("⚙️ 設定", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            learning_mode = st.selectbox(
                "学習モード",
                ["順番通り", "ランダム", "特定グループ"],
                help="学習する順序を選択",
            )
        with col2:
            if learning_mode == "特定グループ":
                selected_group = st.selectbox(
                    "グループ選択",
                    options=sorted(df["group_id"].unique()),
                )
                filtered_df = df[df["group_id"] == selected_group].reset_index(drop=True)
            else:
                filtered_df = df.copy()

        jump_to = st.number_input(
            "文章番号へジャンプ",
            min_value=1,
            max_value=len(filtered_df),
            value=st.session_state.current_sentence_idx + 1,
            step=1,
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("移動", key="jump_btn", use_container_width=True):
                st.session_state.current_sentence_idx = jump_to - 1
                st.rerun()
        with btn_col2:
            if st.button("リセット", key="reset_btn", use_container_width=True):
                st.session_state.current_sentence_idx = 0
                st.session_state.show_translation = False
                if "shuffled_indices" in st.session_state:
                    del st.session_state.shuffled_indices
                st.rerun()

    if learning_mode == "特定グループ":
        pass
    else:
        filtered_df = df.copy()

    if learning_mode == "ランダム":
        if "shuffled_indices" not in st.session_state or len(st.session_state.shuffled_indices) != len(filtered_df):
            st.session_state.shuffled_indices = list(range(len(filtered_df)))
            random.shuffle(st.session_state.shuffled_indices)
        current_idx = st.session_state.shuffled_indices[
            st.session_state.current_sentence_idx % len(st.session_state.shuffled_indices)
        ]
    else:
        current_idx = st.session_state.current_sentence_idx % len(filtered_df)

    current_sentence = filtered_df.iloc[current_idx]

    current_pos = st.session_state.current_sentence_idx + 1
    total_sentences = len(filtered_df)
    st.markdown(f'<div class="progress-simple">{current_pos} / {total_sentences}</div>', unsafe_allow_html=True)

    english_text = current_sentence["sentence_content_en"]
    japanese_text = current_sentence["translated_sentence"]
    card_id = f"card_{current_idx}"

    words_dict = parse_words_dict(current_sentence.get("words_contained_dict", "{}"))
    highlight_words = list(words_dict.values()) if words_dict else None

    flip_card_html = create_flip_card(english_text, japanese_text, card_id, highlight_words=highlight_words)
    st.components.v1.html(flip_card_html, height=280, scrolling=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])

    with col1:
        if st.button("⬅️", key="nav_prev_main", use_container_width=True, disabled=(st.session_state.current_sentence_idx == 0)):
            st.session_state.current_sentence_idx -= 1
            st.rerun()

    with col2:
        if "audio_speed" not in st.session_state:
            st.session_state.audio_speed = 1.0
        if st.button("🔊", key="play_audio_main", use_container_width=True):
            play_server_generated_audio(english_text, rate=st.session_state.audio_speed)

    with col3:
        if st.button("➡️", key="nav_next_main", use_container_width=True):
            st.session_state.current_sentence_idx += 1
            st.rerun()

    speed_options = {"🐌": 0.7, "🎵": 1.0, "🚀": 1.3}
    speed_cols = st.columns(3)
    for i, (icon, rate) in enumerate(speed_options.items()):
        with speed_cols[i]:
            selected = st.session_state.audio_speed == rate
            btn_label = f"{'●' if selected else '○'} {icon}"
            if st.button(btn_label, key=f"speed_{rate}", use_container_width=True):
                st.session_state.audio_speed = rate
                st.rerun()

    if words_dict:
        words_html = " ".join([f'<span class="word-chip">{word}</span>' for word in words_dict.values()])
        st.markdown(f'<div style="text-align:center; padding:0.5rem 0;">{words_html}</div>', unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    understanding_level = None

    with col1:
        if st.button("😕\n難しい", key="understand_difficult", use_container_width=True):
            understanding_level = "difficult"
    with col2:
        if st.button("😐\n普通", key="understand_normal", use_container_width=True):
            understanding_level = "normal"
    with col3:
        if st.button("😊\n簡単", key="understand_easy", use_container_width=True):
            understanding_level = "easy"

    if understanding_level:
        sentence_key = f"{current_sentence['group_id']}_{current_sentence['sentence_id']}"
        st.session_state.learning_progress[sentence_key] = understanding_level
        st.session_state.studied_today += 1
        st.session_state.current_sentence_idx += 1
        st.rerun()


def shadowing_tab():
    """シャドーイングタブ"""
    st.markdown("## 🎯 AI生成文章でシャドーイング")

    if not st.session_state.get("gemini_api_key") or not initialize_gemini():
        st.warning("🔑 Gemini APIキーを設定してください")
        env_key = os.getenv("GOOGLE_API_KEY", "")
        if env_key:
            st.info(f"💡 環境変数から検出: {env_key[:10]}...")
        api_key = st.text_input(
            "Google AI APIキー",
            value=st.session_state.get("gemini_api_key", ""),
            type="password",
            help="Google AI Studioで取得したAPIキーを入力。.envファイルのGOOGLE_API_KEYでも設定可能",
        )
        if st.button("APIキーを設定"):
            if api_key:
                st.session_state.gemini_api_key = api_key
                if initialize_gemini():
                    st.success("✅ APIキーが正常に設定されました！")
                    st.rerun()
                else:
                    st.error("❌ APIキーが無効です。確認してください。")
            else:
                st.error("APIキーを入力してください。")
        return

    if not st.session_state.generated_content:
        st.markdown("### 📝 新しい記事を生成")
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_genre = st.selectbox(
                "ジャンル選択",
                options=list(GENRE_PROMPTS.keys()),
                format_func=lambda x: GENRE_PROMPTS[x]["name"],
            )
        with col2:
            topic = st.text_input(
                "詳細テーマ",
                placeholder="例: NVIDIA, 再生可能エネルギー, イギリス産業革命, 海洋汚染",
                help="分析したい具体的な企業名、技術、歴史的事件、環境問題などを入力",
            )
        if st.button("🚀 記事を生成", disabled=not topic):
            with st.spinner("✨ Geminiで高品質な記事を生成中..."):
                content = generate_content_with_gemini(selected_genre, topic)
                if content:
                    parsed_content = parse_generated_content(content)
                    if parsed_content:
                        st.session_state.generated_content = parsed_content
                        st.session_state.current_shadowing_idx = 0
                        st.session_state.show_shadowing_translation = False
                        st.success(f"✅ 記事を生成しました！ ({len(parsed_content)}文)")
                        st.rerun()
                    else:
                        st.error("❌ 記事の解析に失敗しました。再試行してください。")
        return

    content = st.session_state.generated_content
    current_idx = st.session_state.current_shadowing_idx

    with st.expander("⚙️ 設定", expanded=False):
        jump_to = st.number_input(
            "文番号へジャンプ",
            min_value=1,
            max_value=len(content),
            value=current_idx + 1,
            step=1,
            key="shadowing_jump",
        )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("移動", key="shadowing_jump_btn", use_container_width=True):
                st.session_state.current_shadowing_idx = jump_to - 1
                st.rerun()
        with btn_col2:
            if st.button("新規記事", key="new_article", use_container_width=True):
                st.session_state.generated_content = []
                st.session_state.current_shadowing_idx = 0
                st.session_state.show_shadowing_translation = False
                st.rerun()

    current_pos = current_idx + 1
    total_sentences = len(content)
    st.markdown(f'<div class="progress-simple">{current_pos} / {total_sentences}</div>', unsafe_allow_html=True)

    current_sentence = content[current_idx]
    card_id = f"shadow_{current_idx}"

    flip_card_html = create_flip_card(
        current_sentence["english"],
        current_sentence["japanese"],
        card_id,
    )
    st.components.v1.html(flip_card_html, height=280, scrolling=True)

    col1, col2, col3 = st.columns([1.5, 1, 1.5])

    with col1:
        if st.button("⬅️", key="shadowing_prev", use_container_width=True, disabled=(current_idx == 0)):
            st.session_state.current_shadowing_idx -= 1
            st.rerun()

    with col2:
        if "shadowing_audio_speed" not in st.session_state:
            st.session_state.shadowing_audio_speed = 1.0
        if st.button("🔊", key="shadowing_play_audio", use_container_width=True):
            play_server_generated_audio(current_sentence["english"], rate=st.session_state.shadowing_audio_speed)

    with col3:
        if st.button("➡️", key="shadowing_next", use_container_width=True):
            if current_idx < len(content) - 1:
                st.session_state.current_shadowing_idx += 1
                st.rerun()
            else:
                st.success("🎉 記事の最後まで完了しました！")

    speed_options = {"🐌": 0.7, "🎵": 1.0, "🚀": 1.3}
    speed_cols = st.columns(3)
    for i, (icon, rate) in enumerate(speed_options.items()):
        with speed_cols[i]:
            selected = st.session_state.shadowing_audio_speed == rate
            btn_label = f"{'●' if selected else '○'} {icon}"
            if st.button(btn_label, key=f"shadowing_speed_{rate}", use_container_width=True):
                st.session_state.shadowing_audio_speed = rate
                st.rerun()

    with st.expander("📄 記事全体を表示"):
        for i, sentence_pair in enumerate(content):
            st.markdown(f"**{i+1}.** {sentence_pair['english']}")
            st.markdown(f"_{sentence_pair['japanese']}_")
            st.markdown("---")


def progress_tab(df):
    """学習記録タブ"""
    st.markdown("## 📊 学習記録・進捗")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("総学習文章数", len(st.session_state.learning_progress))
    with col2:
        st.metric("今日の学習数", st.session_state.studied_today)
    with col3:
        if st.session_state.learning_progress:
            avg_difficulty = sum(
                1 if level == "easy" else 2 if level == "normal" else 3
                for level in st.session_state.learning_progress.values()
            ) / len(st.session_state.learning_progress)
            st.metric("平均難易度", f"{avg_difficulty:.1f}/3.0")

    if st.session_state.learning_progress:
        understanding_counts = {"easy": 0, "normal": 0, "difficult": 0}
        for level in st.session_state.learning_progress.values():
            understanding_counts[level] += 1
        st.markdown("### 📈 理解度分布")
        progress_df = pd.DataFrame([
            {"理解度": "簡単", "文章数": understanding_counts["easy"]},
            {"理解度": "普通", "文章数": understanding_counts["normal"]},
            {"理解度": "難しい", "文章数": understanding_counts["difficult"]},
        ])
        st.bar_chart(progress_df.set_index("理解度"))

    if st.session_state.learning_progress:
        st.markdown("### 📝 学習ログ")
        log_data = []
        for sentence_key, level in st.session_state.learning_progress.items():
            group_id, sentence_id = sentence_key.split("_")
            log_data.append({
                "グループ": group_id,
                "文章ID": sentence_id,
                "理解度": {"easy": "😊 簡単", "normal": "😐 普通", "difficult": "😕 難しい"}[level],
            })
        log_df = pd.DataFrame(log_data)
        st.dataframe(log_df, use_container_width=True)


def create_sample_data():
    """サンプルデータ作成（テスト用）"""
    os.makedirs("data", exist_ok=True)

    sample_data1 = pd.DataFrame({
        "group_id": [1, 1, 1],
        "sentence_id": [1, 2, 3],
        "sentence_type": ["academic", "conversation", "free"],
        "sentence_content_en": [
            "The rapid advancement of artificial intelligence has revolutionized various industries.",
            "Could you please explain how machine learning algorithms work in simple terms?",
            "Data science combines statistics, programming, and domain expertise to extract insights.",
        ],
        "translated_sentence": [
            "人工知能の急速な発展は、様々な産業に革命をもたらしました。",
            "機械学習アルゴリズムがどのように動作するかを簡単に説明していただけますか？",
            "データサイエンスは統計学、プログラミング、ドメイン専門知識を組み合わせて洞察を抽出します。",
        ],
        "words_contained_dict": [
            "{'1': 'artificial', '2': 'intelligence', '3': 'revolutionized'}",
            "{'4': 'machine', '5': 'learning', '6': 'algorithms'}",
            "{'7': 'statistics', '8': 'programming', '9': 'expertise'}",
        ],
    })
    sample_data1.to_csv("data/group1.csv", index=False)

    word_master = pd.DataFrame({
        "word_id": range(1, 10),
        "word": [
            "artificial", "intelligence", "revolutionized", "machine", "learning",
            "algorithms", "statistics", "programming", "expertise",
        ],
    })
    word_master.to_csv("data/word_master.csv", index=False)

    st.success("✅ サンプルデータを作成しました！ページを再読み込みしてください。")
