import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import json
from pathlib import Path
import random
import google.generativeai as genai
from dotenv import load_dotenv

# 環境変数読み込み
load_dotenv()

# ジャンル別プロンプト定義
GENRE_PROMPTS = {
    "ビジネス・企業": {
        "name": "🏢 ビジネス・企業分析",
        "prompt": """あなたは、技術・ビジネスに関する専門的なリサーチャー兼ライターです。
指定された「{topic}」について、最新の動向・技術・研究・市場戦略・主要プロダクト・将来の展望を詳しく分析し、論理的かつ一貫性のある英文記事を作成してください。

【記事の使用目的】
1. 英語学習: C1レベルの文章でリーディングとシャドーイング練習
2. 投資判断: 株式投資のための業界・企業理解ツール

【記事要件】
- 企業のビジョン・ミッション、歴史・規模・所在地、最新プロダクト・サービス、研究開発、競争環境、今後の展望を含む
- 論理的に構成された800-1200語の分析記事
- 投資判断に役立つ分析的視点を含む
- British English表現を使用

【出力フォーマット】
英語記事を1文ごとに改行して出力し、その直後に各文の自然な日本語訳を記載してください。

例:
NVIDIA is at the forefront of AI and GPU technology, continuously expanding its influence in gaming, data centers, and autonomous vehicles.
NVIDIAはAIとGPU技術の最前線に立ち、ゲーム、データセンター、自動運転車の分野で影響力を拡大し続けています。

The company's latest innovation, the Blackwell GPU architecture, aims to enhance AI training efficiency by 4x compared to previous models.
同社の最新技術であるBlackwell GPUアーキテクチャは、従来モデルと比較してAIの学習効率を4倍向上させることを目指しています。"""
    },
    
    "科学・テクノロジー": {
        "name": "🔬 科学・テクノロジー",
        "prompt": """あなたは科学・技術分野の専門ライターです。
「{topic}」について、最新の研究動向、技術革新、社会への影響、将来の可能性を包括的に分析した英文記事を作成してください。

【記事要件】
- 科学的根拠に基づいた正確な情報
- 最新の研究成果や技術動向を含む
- 社会への実用化・影響を分析
- 800-1200語程度、C1レベルの英語
- British English表現を使用

【出力フォーマット】
英語記事を1文ごとに改行して出力し、その直後に各文の自然な日本語訳を記載してください。"""
    },
    
    "歴史・文化": {
        "name": "🏛️ 歴史・文化",
        "prompt": """あなたは歴史・文化研究の専門家です。
「{topic}」について、歴史的背景、文化的意義、現代への影響、国際的な視点を織り交ぜた英文記事を作成してください。

【記事要件】
- 歴史的事実の正確性を重視
- 文化的コンテキストの説明を含む
- 現代社会との関連性を分析
- 800-1200語程度、C1レベルの英語
- British English表現を使用

【出力フォーマット】
英語記事を1文ごとに改行して出力し、その直後に各文の自然な日本語訳を記載してください。"""
    },
    
    "自然・環境": {
        "name": "🌍 自然・環境",
        "prompt": """あなたは環境科学・自然保護の専門ライターです。
「{topic}」について、生態系への影響、環境問題、保護活動、持続可能な解決策を分析した英文記事を作成してください。

【記事要件】
- 科学的データに基づいた環境分析
- 生態系や気候変動への影響を含む
- 実践可能な解決策の提案
- 800-1200語程度、C1レベルの英語
- British English表現を使用

【出力フォーマット】
英語記事を1文ごとに改行して出力し、その直後に各文の自然な日本語訳を記載してください。"""
    }
}
st.set_page_config(
    page_title="英語学習アプリ",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（モバイル対応）
st.markdown("""
<style>
.main-header {
    font-size: 2rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}

.sentence-card {
    background-color: #f8f9fa !important;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
    color: #212529 !important;
}

.sentence-card p, .sentence-card h3, .sentence-card h4 {
    color: #212529 !important;
}

.translation-card {
    background-color: #e8f4fd !important;
    padding: 1rem;
    border-radius: 8px;
    margin: 0.5rem 0;
    border: 1px solid #bee5eb;
    color: #495057 !important;
}

.translation-card p, .translation-card h4 {
    color: #495057 !important;
}

/* ダークモード対応 */
[data-theme="dark"] .sentence-card {
    background-color: #343a40;
    color: #f8f9fa;
    border-left: 4px solid #4dabf7;
}

[data-theme="dark"] .translation-card {
    background-color: #495057;
    color: #f8f9fa;
    border: 1px solid #6c757d;
}

.progress-text {
    font-size: 1.1rem;
    font-weight: 500;
}

.word-chip {
    background-color: #007bff;
    color: white;
    padding: 0.2rem 0.6rem;
    border-radius: 15px;
    font-size: 0.8rem;
    margin: 0.2rem;
    display: inline-block;
}

/* モバイル対応 */
@media (max-width: 768px) {
    .sentence-card {
        padding: 1rem;
    }
    .main-header {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# データローダー関数
@st.cache_data
def load_all_csv_data(data_dir="data"):
    """全CSVファイルを読み込んで統合"""
    all_data = []
    csv_files = glob.glob(os.path.join(data_dir, "group*.csv"))
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            all_data.append(df)
            st.sidebar.success(f"✅ {os.path.basename(file_path)} 読み込み完了")
        except Exception as e:
            st.sidebar.error(f"❌ {os.path.basename(file_path)} 読み込みエラー: {str(e)}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()

@st.cache_data
def load_word_master(data_dir="data"):
    """単語マスターデータを読み込み"""
    word_master_path = os.path.join(data_dir, "word_master.csv")
    try:
        if os.path.exists(word_master_path):
            return pd.read_csv(word_master_path)
        else:
            st.sidebar.warning("⚠️ word_master.csv が見つかりません")
            return pd.DataFrame()
    except Exception as e:
        st.sidebar.error(f"❌ word_master.csv 読み込みエラー: {str(e)}")
        return pd.DataFrame()

def parse_words_dict(words_str):
    """words_contained_dict文字列を辞書に変換"""
    try:
        if pd.isna(words_str) or words_str == "":
            return {}
        # 文字列が辞書形式の場合
        if isinstance(words_str, str):
            return json.loads(words_str.replace("'", '"'))
        return {}
    except:
        return {}

def show_available_voices():
    """利用可能な音声を表示"""
    html_code = """
    <script>
        function showVoices() {
            const voices = window.speechSynthesis.getVoices();
            const voiceList = voices.map(voice => 
                `${voice.name} (${voice.lang}) - ${voice.localService ? 'Local' : 'Remote'}`
            ).join('<br>');
            
            const britishVoices = voices.filter(voice => 
                voice.lang.includes('en-GB') || 
                voice.name.toLowerCase().includes('british') ||
                voice.name.toLowerCase().includes('uk')
            );
            
            const britishList = britishVoices.map(voice => 
                `✅ ${voice.name} (${voice.lang})`
            ).join('<br>');
            
            document.getElementById('voice-info').innerHTML = `
                <div style="background: #f0f8ff; padding: 15px; border-radius: 8px; margin: 10px 0;">
                    <h4>🇬🇧 British English音声:</h4>
                    ${britishList || '❌ British English音声が見つかりません'}
                    
                    <h4>📋 全ての利用可能音声:</h4>
                    <div style="max-height: 200px; overflow-y: auto; font-size: 12px;">
                        ${voiceList}
                    </div>
                </div>
            `;
        }
        
        if (window.speechSynthesis.getVoices().length === 0) {
            window.speechSynthesis.onvoiceschanged = showVoices;
        } else {
            showVoices();
        }
    </script>
    <div id="voice-info">音声情報を読み込み中...</div>
    """
    
    st.components.v1.html(html_code, height=300)

def play_text_to_speech(text, rate=1.0):
    """ブラウザ標準TTSで音声再生"""
    # 特殊文字をエスケープ
    escaped_text = text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    
    # British English音声設定のJavaScript
    html_code = f"""
    <script>
        function playTTS() {{
            // 既存の音声を停止
            window.speechSynthesis.cancel();
            
            const utterance = new SpeechSynthesisUtterance("{escaped_text}");
            
            // British English設定
            utterance.lang = 'en-GB';
            utterance.rate = {rate};
            utterance.pitch = 1.0;
            utterance.volume = 1.0;
            
            // 利用可能な音声を取得してBritish Englishを優先選択
            const voices = window.speechSynthesis.getVoices();
            const britishVoice = voices.find(voice => 
                voice.lang.includes('en-GB') || 
                voice.name.toLowerCase().includes('british') ||
                voice.name.toLowerCase().includes('uk')
            );
            
            if (britishVoice) {{
                utterance.voice = britishVoice;
                console.log('Using British voice:', britishVoice.name);
            }} else {{
                console.log('British voice not found, using default');
            }}
            
            // エラーハンドリング
            utterance.onerror = function(event) {{
                console.error('Speech synthesis error:', event.error);
            }};
            
            utterance.onend = function() {{
                console.log('Speech finished');
            }};
            
            // 音声再生
            window.speechSynthesis.speak(utterance);
        }}
        
        // 音声リストが読み込まれるまで少し待つ
        if (window.speechSynthesis.getVoices().length === 0) {{
            window.speechSynthesis.onvoiceschanged = function() {{
                playTTS();
            }};
        }} else {{
            playTTS();
        }}
    </script>
    """
    
    # Streamlitで実行
    st.components.v1.html(html_code, height=0)

def initialize_gemini():
    """Gemini APIの初期化"""
    # 環境変数から読み込み、なければセッション状態から
    env_api_key = os.getenv('GOOGLE_API_KEY', '')
    
    if env_api_key and 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = env_api_key
    elif 'gemini_api_key' not in st.session_state:
        st.session_state.gemini_api_key = ""
    
    if st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            return True
        except Exception as e:
            st.error(f"Gemini API初期化エラー: {str(e)}")
            return False
    return False

def initialize_session_state():
    """セッション状態の初期化"""
    if 'current_sentence_idx' not in st.session_state:
        st.session_state.current_sentence_idx = 0
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {}
    if 'show_translation' not in st.session_state:
        st.session_state.show_translation = False
    if 'studied_today' not in st.session_state:
        st.session_state.studied_today = 0
    if 'generated_content' not in st.session_state:
        st.session_state.generated_content = []
    if 'current_shadowing_idx' not in st.session_state:
        st.session_state.current_shadowing_idx = 0
    if 'show_shadowing_translation' not in st.session_state:
        st.session_state.show_shadowing_translation = False
    if 'mobile_mode' not in st.session_state:
        st.session_state.mobile_mode = False
    if 'audio_mode' not in st.session_state:
        st.session_state.audio_mode = 'full'  # 'full' or 'sentence'

def generate_content_with_gemini(genre, topic):
    """Gemini APIでコンテンツ生成"""
    try:
        # 新しいモデル名に変更
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = GENRE_PROMPTS[genre]["prompt"].format(topic=topic)
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"コンテンツ生成エラー: {str(e)}")
        return None

def parse_generated_content(content):
    """生成されたコンテンツを英文と日本語訳に分割"""
    lines = content.strip().split('\n')
    parsed_content = []
    
    current_en = ""
    current_jp = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 英文の判定（アルファベットで始まり、英語っぽい）
        if line[0].isupper() and any(c.isalpha() for c in line):
            if current_en and current_jp:
                parsed_content.append({"english": current_en, "japanese": current_jp})
            current_en = line
            current_jp = ""
        else:
            # 日本語訳として扱う
            current_jp = line
    
    # 最後のペアを追加
    if current_en and current_jp:
        parsed_content.append({"english": current_en, "japanese": current_jp})
    
    return parsed_content

def highlight_words_in_sentence(sentence, words_dict, word_master):
    """文章内の学習対象単語をハイライト"""
    if not words_dict:
        return sentence
    
    highlighted_sentence = sentence
    target_words = list(words_dict.values())
    
    # 単語を長い順にソート（部分マッチを避けるため）
    target_words.sort(key=len, reverse=True)
    
    for word in target_words:
        # 大文字小文字を無視して置換
        import re
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_sentence = pattern.sub(
            f'<span style="background-color: #ffeb3b; font-weight: bold; padding: 2px 4px; border-radius: 3px;">{word}</span>',
            highlighted_sentence
        )
    
    return highlighted_sentence

def highlight_words_in_japanese(japanese_sentence, words_dict, word_master):
    """日本語訳内の対応する単語をハイライト"""
    if not words_dict or word_master.empty:
        return japanese_sentence
    
    highlighted_sentence = japanese_sentence
    
    # word_masterから日本語の意味を取得してハイライト
    for word_id, english_word in words_dict.items():
        try:
            word_id_int = int(word_id)
            word_info = word_master[word_master['word_id'] == word_id_int]
            if not word_info.empty and 'japanese_meaning' in word_info.columns:
                japanese_meaning = word_info.iloc[0]['japanese_meaning']
                if japanese_meaning in highlighted_sentence:
                    highlighted_sentence = highlighted_sentence.replace(
                        japanese_meaning,
                        f'<span style="background-color: #c8e6c9; font-weight: bold; padding: 2px 4px; border-radius: 3px;">{japanese_meaning}</span>'
                    )
        except:
            continue
    
    return highlighted_sentence
    """生成されたコンテンツを英文と日本語訳に分割"""
    lines = content.strip().split('\n')
    parsed_content = []
    
    current_en = ""
    current_jp = ""
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # 英文の判定（アルファベットで始まり、英語っぽい）
        if line[0].isupper() and any(c.isalpha() for c in line):
            if current_en and current_jp:
                parsed_content.append({"english": current_en, "japanese": current_jp})
            current_en = line
            current_jp = ""
        else:
            # 日本語訳として扱う
            current_jp = line
    
    # 最後のペアを追加
    if current_en and current_jp:
        parsed_content.append({"english": current_en, "japanese": current_jp})
    
    return parsed_content
    """セッション状態の初期化"""
    if 'current_sentence_idx' not in st.session_state:
        st.session_state.current_sentence_idx = 0
    if 'learning_progress' not in st.session_state:
        st.session_state.learning_progress = {}
    if 'show_translation' not in st.session_state:
        st.session_state.show_translation = False
    if 'studied_today' not in st.session_state:
        st.session_state.studied_today = 0

def main():
    # セッション状態初期化
    initialize_session_state()
    
    # Gemini API初期化を最初に実行
    initialize_gemini()
    
    # ヘッダー
    st.markdown('<h1 class="main-header">📚 英語学習アプリ</h1>', unsafe_allow_html=True)
    
    # サイドバー - API設定状況
    st.sidebar.markdown("## 🔑 API設定状況")
    env_api_key = os.getenv('GOOGLE_API_KEY', '')
    session_api_key = st.session_state.get('gemini_api_key', '')
    
    if env_api_key:
        st.sidebar.success(f"✅ 環境変数: {env_api_key[:8]}...")
    else:
        st.sidebar.warning("⚠️ 環境変数なし")
    
    if session_api_key:
        st.sidebar.success(f"✅ セッション: {session_api_key[:8]}...")
    else:
        st.sidebar.warning("⚠️ セッションなし")
    
    if st.sidebar.button("🔄 API再読み込み"):
        initialize_gemini()
        st.rerun()
    
    # サイドバー - 音声設定
    st.sidebar.markdown("## 🔊 音声設定")
    
    # 利用可能な音声を表示
    if st.sidebar.button("🎤 利用可能な音声を確認"):
        show_available_voices()
    
    # 音声設定のヘルプ
    with st.sidebar.expander("📖 音声機能について"):
        st.markdown("""
        **現在の音声機能:**
        - ブラウザ標準のText-to-Speech
        - British English (en-GB) 優先
        - 3段階の速度調整
        
        **デバイス別対応:**
        - 🍎 **Mac**: 高品質なDaniel (British)
        - 🪟 **Windows**: Microsoft系音声
        - 📱 **iOS**: 内蔵British音声
        
        **次回アップデート予定:**
        - Google Cloud TTS (高品質)
        - 単語別再生機能
        """)
    
    # サイドバー - データ読み込み状況
    st.sidebar.markdown("## 📊 データ読み込み状況")
    
    # データ読み込み
    df = load_all_csv_data()
    word_master = load_word_master()
    
    if df.empty:
        st.error("📁 CSVファイルが見つかりません。'data'フォルダにgroup*.csvファイルを配置してください。")
        
        # サンプルデータ作成ボタン
        if st.button("🔧 サンプルデータを作成"):
            create_sample_data()
            st.rerun()
        
        return
    
    # データ概要表示
    st.sidebar.markdown(f"**📈 データ統計**")
    st.sidebar.write(f"• 総文章数: {len(df)}")
    st.sidebar.write(f"• グループ数: {df['group_id'].nunique()}")
    st.sidebar.write(f"• 今日の学習: {st.session_state.studied_today}文章")
    
    # メインナビゲーション
    tab1, tab2, tab3 = st.tabs(["📚 単語学習", "🎯 シャドーイング", "📊 学習記録"])
    
    with tab1:
        word_learning_tab(df, word_master)
    
    with tab2:
        shadowing_tab()
    
    with tab3:
        progress_tab(df)

def word_learning_tab(df, word_master):
    """単語学習タブ"""
    st.markdown("## 📖 文章ベース単語学習")
    
    # 学習オプション
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        learning_mode = st.selectbox(
            "学習モード",
            ["順番通り", "ランダム", "特定グループ"],
            help="学習する順序を選択"
        )
    
    with col2:
        if learning_mode == "特定グループ":
            selected_group = st.selectbox(
                "グループ選択",
                options=sorted(df['group_id'].unique())
            )
            filtered_df = df[df['group_id'] == selected_group].reset_index(drop=True)
        else:
            filtered_df = df.copy()
    
    with col3:
        if st.button("🔄 リセット"):
            st.session_state.current_sentence_idx = 0
            st.session_state.show_translation = False
            st.rerun()
    
    # 学習対象データフレーム
    if learning_mode == "ランダム":
        if 'shuffled_indices' not in st.session_state:
            st.session_state.shuffled_indices = list(range(len(filtered_df)))
            random.shuffle(st.session_state.shuffled_indices)
        current_idx = st.session_state.shuffled_indices[st.session_state.current_sentence_idx % len(st.session_state.shuffled_indices)]
    else:
        current_idx = st.session_state.current_sentence_idx % len(filtered_df)
    
    current_sentence = filtered_df.iloc[current_idx]
    
    # 進捗表示
    progress = (st.session_state.current_sentence_idx + 1) / len(filtered_df)
    st.progress(progress)
    st.markdown(f'<p class="progress-text">進捗: {st.session_state.current_sentence_idx + 1} / {len(filtered_df)} 文章</p>', unsafe_allow_html=True)
    
    # 文章表示カード（ハイライト付き）
    words_dict = parse_words_dict(current_sentence.get('words_contained_dict', '{}'))
    highlighted_sentence = highlight_words_in_sentence(
        current_sentence['sentence_content_en'], 
        words_dict, 
        word_master
    )
    
    st.markdown(f'''
    <div class="sentence-card">
        <div class="swipe-indicator left">😕</div>
        <div class="swipe-indicator right">😊</div>
        <h3>📝 Group {current_sentence['group_id']} - Sentence {current_sentence['sentence_id']}</h3>
        <h4>英文:</h4>
        <p style="font-size: 1.2rem; line-height: 1.6;">{highlighted_sentence}</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # モバイル検出とモード切り替え
    is_mobile = st.checkbox("📱 モバイルモード", value=st.session_state.mobile_mode)
    if is_mobile != st.session_state.mobile_mode:
        st.session_state.mobile_mode = is_mobile
        st.rerun()
    
    # 音声再生コントロール
    st.markdown("### 🔊 音声再生")
    
    # 音声モード選択
    audio_mode = st.radio(
        "再生モード",
        ["📄 全文一括再生", "📝 1文ずつ再生"],
        index=0 if st.session_state.audio_mode == 'full' else 1,
        horizontal=True
    )
    st.session_state.audio_mode = 'full' if audio_mode.startswith("📄") else 'sentence'
    
    if st.session_state.mobile_mode:
        # モバイルモード：縦並び大きなボタン
        if st.button("🔊 通常速度で再生", key="mobile_normal", use_container_width=True):
            play_text_to_speech(current_sentence['sentence_content_en'], rate=1.0)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🐌 ゆっくり", key="mobile_slow", use_container_width=True):
                play_text_to_speech(current_sentence['sentence_content_en'], rate=0.7)
        with col2:
            if st.button("🚀 早め", key="mobile_fast", use_container_width=True):
                play_text_to_speech(current_sentence['sentence_content_en'], rate=1.3)
        
        if st.button("⏹️ 停止", key="mobile_stop", use_container_width=True):
            st.components.v1.html("""
                <script>window.speechSynthesis.cancel();</script>
            """, height=0)
    else:
        # デスクトップモード：横並び
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("🔊 通常速度"):
                play_text_to_speech(current_sentence['sentence_content_en'], rate=1.0)
        with col2:
            if st.button("🐌 ゆっくり"):
                play_text_to_speech(current_sentence['sentence_content_en'], rate=0.7)
        with col3:
            if st.button("🚀 早め"):
                play_text_to_speech(current_sentence['sentence_content_en'], rate=1.3)
        with col4:
            if st.button("⏹️ 停止"):
                st.components.v1.html("""
                    <script>window.speechSynthesis.cancel();</script>
                """, height=0)
    
    # 翻訳表示/非表示
    if st.session_state.mobile_mode:
        # モバイル：トグルボタン
        if st.button(
            "👀 日本語訳を表示" if not st.session_state.show_translation else "🙈 日本語訳を隠す",
            key="mobile_translation_toggle",
            use_container_width=True
        ):
            st.session_state.show_translation = not st.session_state.show_translation
    else:
        # デスクトップ：中央ボタン
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("👀 日本語訳を表示" if not st.session_state.show_translation else "🙈 日本語訳を隠す"):
                st.session_state.show_translation = not st.session_state.show_translation
    
    # 翻訳表示
    if st.session_state.show_translation:
        highlighted_translation = highlight_words_in_japanese(
            current_sentence['translated_sentence'], 
            words_dict, 
            word_master
        )
        st.markdown(f'''
        <div class="translation-card">
            <h4>🇯🇵 日本語訳:</h4>
            <p style="font-size: 1.1rem;">{highlighted_translation}</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # 含有単語表示
    if words_dict:
        st.markdown("**📚 学習対象単語:**")
        words_html = "".join([f'<span class="word-chip">{word}</span>' for word in words_dict.values()])
        st.markdown(words_html, unsafe_allow_html=True)
    
    # 理解度チェック
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    understanding_level = None
    with col1:
        if st.button("😕 難しい"):
            understanding_level = "difficult"
    with col2:
        if st.button("😐 普通"):
            understanding_level = "normal"
    with col3:
        if st.button("😊 簡単"):
            understanding_level = "easy"
    
    # 理解度記録と次の文章へ
    if understanding_level:
        sentence_key = f"{current_sentence['group_id']}_{current_sentence['sentence_id']}"
        st.session_state.learning_progress[sentence_key] = understanding_level
        st.session_state.studied_today += 1
        st.session_state.current_sentence_idx += 1
        st.session_state.show_translation = False
        
        level_names = {'difficult': '難しい', 'normal': '普通', 'easy': '簡単'}
        st.success(f"理解度を記録しました: {level_names[understanding_level]}")
        st.rerun()
    
    # ナビゲーションボタン
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ 前の文章") and st.session_state.current_sentence_idx > 0:
            st.session_state.current_sentence_idx -= 1
            st.session_state.show_translation = False
            st.rerun()
    
    with col3:
        if st.button("次の文章 ➡️"):
            st.session_state.current_sentence_idx += 1
            st.session_state.show_translation = False
            st.rerun()

def shadowing_tab():
    """シャドーイングタブ"""
    st.markdown("## 🎯 AI生成文章でシャドーイング")
    
    # API Key設定
    if not st.session_state.get('gemini_api_key') or not initialize_gemini():
        st.warning("🔑 Gemini APIキーを設定してください")
        
        # 現在の設定状況を表示
        env_key = os.getenv('GOOGLE_API_KEY', '')
        if env_key:
            st.info(f"💡 環境変数から検出: {env_key[:10]}...")
        
        api_key = st.text_input(
            "Google AI APIキー", 
            value=st.session_state.get('gemini_api_key', ''),
            type="password", 
            help="Google AI Studioで取得したAPIキーを入力。.envファイルのGOOGLE_API_KEYでも設定可能"
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
    
    # コンテンツ生成セクション
    if not st.session_state.generated_content:
        st.markdown("### 📝 新しい記事を生成")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_genre = st.selectbox(
                "ジャンル選択",
                options=list(GENRE_PROMPTS.keys()),
                format_func=lambda x: GENRE_PROMPTS[x]["name"]
            )
        
        with col2:
            topic = st.text_input(
                "詳細テーマ",
                placeholder="例: NVIDIA, 再生可能エネルギー, イギリス産業革命, 海洋汚染",
                help="分析したい具体的な企業名、技術、歴史的事件、環境問題などを入力"
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
    
    # 生成されたコンテンツでシャドーイング学習
    else:
        content = st.session_state.generated_content
        current_idx = st.session_state.current_shadowing_idx
        
        # 進捗表示
        progress = (current_idx + 1) / len(content)
        st.progress(progress)
        st.markdown(f'<p class="progress-text">進捗: {current_idx + 1} / {len(content)} 文</p>', unsafe_allow_html=True)
        
        # 現在の文章表示
        current_sentence = content[current_idx]
        
        st.markdown(f'''
        <div class="sentence-card">
            <h3>🎯 シャドーイング練習 - 文 {current_idx + 1}</h3>
            <h4>英文:</h4>
            <p style="font-size: 1.2rem; line-height: 1.6;">{current_sentence["english"]}</p>
        </div>
        ''', unsafe_allow_html=True)
        
        # 音声再生コントロール
        st.markdown("### 🔊 音声再生")
        
        # 音声モード選択
        audio_mode_shadowing = st.radio(
            "再生モード",
            ["📄 全文一括再生", "📝 1文ずつ再生"],
            index=0 if st.session_state.audio_mode == 'full' else 1,
            horizontal=True,
            key="shadowing_audio_mode"
        )
        current_audio_mode = 'full' if audio_mode_shadowing.startswith("📄") else 'sentence'
        
        if current_audio_mode == 'full':
            # 全文一括再生モード
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if st.button("🔊 全文通常速度", key="shadowing_full_normal"):
                    full_text = " ".join([item["english"] for item in content])
                    play_text_to_speech(full_text, rate=1.0)
            
            with col2:
                if st.button("🐌 全文ゆっくり", key="shadowing_full_slow"):
                    full_text = " ".join([item["english"] for item in content])
                    play_text_to_speech(full_text, rate=0.7)
            
            with col3:
                if st.button("🚀 全文早め", key="shadowing_full_fast"):
                    full_text = " ".join([item["english"] for item in content])
                    play_text_to_speech(full_text, rate=1.3)
            
            with col4:
                if st.button("⏹️ 停止", key="shadowing_full_stop"):
                    st.components.v1.html("""
                        <script>window.speechSynthesis.cancel();</script>
                    """, height=0)
        else:
            # 1文ずつ再生モード
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                if st.button("🔊 通常速度", key="shadowing_normal"):
                    play_text_to_speech(current_sentence["english"], rate=1.0)
            
            with col2:
                if st.button("🐌 ゆっくり", key="shadowing_slow"):
                    play_text_to_speech(current_sentence["english"], rate=0.7)
            
            with col3:
                if st.button("🚀 早め", key="shadowing_fast"):
                    play_text_to_speech(current_sentence["english"], rate=1.3)
            
            with col4:
                if st.button("⏹️ 停止", key="shadowing_stop"):
                    st.components.v1.html("""
                        <script>window.speechSynthesis.cancel();</script>
                    """, height=0)
        
        # 翻訳表示/非表示
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("👀 日本語訳を表示" if not st.session_state.show_shadowing_translation else "🙈 日本語訳を隠す", key="shadowing_translation"):
                st.session_state.show_shadowing_translation = not st.session_state.show_shadowing_translation
        
        # 翻訳表示
        if st.session_state.show_shadowing_translation:
            st.markdown(f'''
            <div class="translation-card">
                <h4>🇯🇵 日本語訳:</h4>
                <p style="font-size: 1.1rem;">{current_sentence["japanese"]}</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # ナビゲーションボタン
        st.markdown("---")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            if st.button("⬅️ 前の文", key="shadowing_prev") and current_idx > 0:
                st.session_state.current_shadowing_idx -= 1
                st.session_state.show_shadowing_translation = False
                st.rerun()
        
        with col2:
            if st.button("🔄 リピート", key="shadowing_repeat"):
                play_text_to_speech(current_sentence["english"], rate=1.0)
        
        with col3:
            if st.button("次の文 ➡️", key="shadowing_next"):
                if current_idx < len(content) - 1:
                    st.session_state.current_shadowing_idx += 1
                    st.session_state.show_shadowing_translation = False
                    st.rerun()
                else:
                    st.success("🎉 記事の最後まで完了しました！")
        
        with col4:
            if st.button("🆕 新しい記事", key="new_article"):
                st.session_state.generated_content = []
                st.session_state.current_shadowing_idx = 0
                st.session_state.show_shadowing_translation = False
                st.rerun()
        
        # 記事全体表示オプション
        with st.expander("📄 記事全体を表示"):
            for i, sentence_pair in enumerate(content):
                st.markdown(f"**{i+1}.** {sentence_pair['english']}")
                st.markdown(f"_{sentence_pair['japanese']}_")
                st.markdown("---")

def progress_tab(df):
    """学習記録タブ"""
    st.markdown("## 📊 学習記録・進捗")
    
    # 基本統計
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("総学習文章数", len(st.session_state.learning_progress))
    
    with col2:
        st.metric("今日の学習数", st.session_state.studied_today)
    
    with col3:
        if st.session_state.learning_progress:
            avg_difficulty = sum([1 if level == "easy" else 2 if level == "normal" else 3 
                                for level in st.session_state.learning_progress.values()]) / len(st.session_state.learning_progress)
            st.metric("平均難易度", f"{avg_difficulty:.1f}/3.0")
    
    # 理解度分布
    if st.session_state.learning_progress:
        understanding_counts = {"easy": 0, "normal": 0, "difficult": 0}
        for level in st.session_state.learning_progress.values():
            understanding_counts[level] += 1
        
        st.markdown("### 📈 理解度分布")
        progress_df = pd.DataFrame([
            {"理解度": "簡単", "文章数": understanding_counts["easy"]},
            {"理解度": "普通", "文章数": understanding_counts["normal"]},
            {"理解度": "難しい", "文章数": understanding_counts["difficult"]}
        ])
        st.bar_chart(progress_df.set_index("理解度"))
    
    # 詳細ログ
    if st.session_state.learning_progress:
        st.markdown("### 📝 学習ログ")
        log_data = []
        for sentence_key, level in st.session_state.learning_progress.items():
            group_id, sentence_id = sentence_key.split("_")
            log_data.append({
                "グループ": group_id,
                "文章ID": sentence_id,
                "理解度": {"easy": "😊 簡単", "normal": "😐 普通", "difficult": "😕 難しい"}[level]
            })
        
        log_df = pd.DataFrame(log_data)
        st.dataframe(log_df, use_container_width=True)

def create_sample_data():
    """サンプルデータ作成（テスト用）"""
    os.makedirs("data", exist_ok=True)
    
    # サンプルCSVデータ
    sample_data1 = pd.DataFrame({
        'group_id': [1, 1, 1],
        'sentence_id': [1, 2, 3],
        'sentence_type': ['academic', 'conversation', 'free'],
        'sentence_content_en': [
            "The rapid advancement of artificial intelligence has revolutionized various industries.",
            "Could you please explain how machine learning algorithms work in simple terms?",
            "Data science combines statistics, programming, and domain expertise to extract insights."
        ],
        'translated_sentence': [
            "人工知能の急速な発展は、様々な産業に革命をもたらしました。",
            "機械学習アルゴリズムがどのように動作するかを簡単に説明していただけますか？",
            "データサイエンスは統計学、プログラミング、ドメイン専門知識を組み合わせて洞察を抽出します。"
        ],
        'words_contained_dict': [
            "{'1': 'artificial', '2': 'intelligence', '3': 'revolutionized'}",
            "{'4': 'machine', '5': 'learning', '6': 'algorithms'}",
            "{'7': 'statistics', '8': 'programming', '9': 'expertise'}"
        ]
    })
    
    sample_data1.to_csv("data/group1.csv", index=False)
    
    # word_master.csv
    word_master = pd.DataFrame({
        'word_id': range(1, 10),
        'word': ['artificial', 'intelligence', 'revolutionized', 'machine', 'learning', 
                'algorithms', 'statistics', 'programming', 'expertise']
    })
    word_master.to_csv("data/word_master.csv", index=False)
    
    st.success("✅ サンプルデータを作成しました！ページを再読み込みしてください。")

if __name__ == "__main__":
    main()