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
import html
import re

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
    initial_sidebar_state="collapsed"  # サイドバーは初期状態で折りたたみ
)

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

    /* シンプルな進捗表示 */
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

    /* 安全なハイライト用CSS */
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

    /* ========== 大きなナビゲーションボタン ========== */
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

    .big-nav-button:active {
        transform: scale(0.95);
    }

    /* ========== 大きな理解度ボタン ========== */
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

    .understanding-btn:active {
        transform: scale(0.95);
    }

    .understanding-btn .label {
        font-size: 0.8rem;
        margin-top: 0.3rem;
    }

    /* ========== 音声ボタン ========== */
    .audio-button-center {
        display: flex;
        justify-content: center;
        align-items: center;
        font-size: 2.5rem;
        padding: 1rem;
        cursor: pointer;
    }

    /* ========== iPhone SE向けカラム強制横並び ========== */
    @media (max-width: 400px) {
        /* Streamlit columnsコンテナを強制的にflexで横並びに */
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

        /* ボタンサイズ調整 */
        [data-testid="stHorizontalBlock"] .stButton > button {
            min-height: 48px !important;
            font-size: 0.9rem !important;
            padding: 0.5rem 0.25rem !important;
            white-space: nowrap !important;
        }
    }

    /* ========== Streamlitボタン上書き ========== */
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

    /* 理解度ボタン専用 */
    [data-testid="stHorizontalBlock"] .stButton > button {
        min-height: 56px !important;
        font-size: 1.1rem !important;
    }

    /* ========== 旧スタイル（後方互換性） ========== */
    .sentence-card {
        background-color: #fafafa !important;
        padding: 1.5rem;
        border-radius: 4px;
        border-left: 2px solid #1a1a1a;
        margin: 1rem 0;
        color: #1a1a1a !important;
    }

    .sentence-card p, .sentence-card h3, .sentence-card h4 {
        color: #1a1a1a !important;
    }

    .translation-card {
        background-color: #f5f5f5 !important;
        padding: 1rem;
        border-radius: 4px;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        color: #333 !important;
    }

    .translation-card p, .translation-card h4 {
        color: #333 !important;
    }

    /* ダークモード対応 */
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

    [data-theme="dark"] .progress-simple {
        color: #fafafa;
    }

    [data-theme="dark"] .word-chip {
        background-color: #fafafa;
        color: #1a1a1a;
    }

    [data-theme="dark"] .stButton > button {
        background: #2a2a2a !important;
        color: #fafafa !important;
        border-color: #444 !important;
    }

    [data-theme="dark"] .stButton > button:hover {
        background: #333 !important;
    }

    .progress-text {
        font-size: 1rem;
        font-weight: 500;
    }

    /* ========== expander内のボタン調整 ========== */
    [data-testid="stExpander"] [data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
    }
    </style>
    """, unsafe_allow_html=True)

def create_flip_card(english_text, japanese_text, card_id, show_tap_hint=True, highlight_words=None):
    """フリップカード用HTML/CSS/JSを生成（タップで英文↔和訳を切り替え）

    Args:
        highlight_words: ハイライトする単語のリスト（学習対象単語）
    """
    import html as html_module
    import re

    # テキストをエスケープしつつ、学習対象単語をハイライト
    def escape_and_highlight(text, words_to_highlight=None):
        escaped = html_module.escape(text)
        if words_to_highlight:
            for word in words_to_highlight:
                pattern = re.compile(re.escape(html_module.escape(word)), re.IGNORECASE)
                escaped = pattern.sub(
                    f'<span class="target-word">{html_module.escape(word)}</span>',
                    escaped
                )
        return escaped

    escaped_en = escape_and_highlight(english_text, highlight_words)
    escaped_jp = html_module.escape(japanese_text)

    tap_hint = "tap to translate" if show_tap_hint else ""

    flip_card_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500&family=Noto+Sans+JP:wght@400;500&display=swap');

    .flip-container-{card_id} {{
        perspective: 1000px;
        width: 100%;
        margin: 0.5rem 0;
        touch-action: manipulation;
    }}

    .flip-card-{card_id} {{
        position: relative;
        width: 100%;
        height: 240px;
        transition: transform 0.5s cubic-bezier(0.4, 0, 0.2, 1);
        transform-style: preserve-3d;
        cursor: pointer;
    }}

    .flip-card-{card_id}.flipped {{
        transform: rotateY(180deg);
    }}

    .flip-card-front-{card_id}, .flip-card-back-{card_id} {{
        position: absolute;
        width: 100%;
        height: 240px;
        backface-visibility: hidden;
        border-radius: 4px;
        padding: 1rem 1rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        box-sizing: border-box;
        overflow: visible;
    }}

    .flip-card-front-{card_id} {{
        background: #fafafa;
        color: #1a1a1a;
        border: 1px solid #e0e0e0;
    }}

    .flip-card-back-{card_id} {{
        background: #1a1a1a;
        color: #fafafa;
        transform: rotateY(180deg);
        border: 1px solid #333;
    }}

    .flip-card-scroll-container {{
        flex: 1;
        width: 100%;
        overflow-y: auto;
        overflow-x: hidden;
        -webkit-overflow-scrolling: touch;
        overscroll-behavior-y: contain;
        touch-action: pan-y pinch-zoom;
        padding: 0.25rem 0.5rem;
        text-align: center;
    }}

    .flip-card-text {{
        font-family: 'Source Serif 4', Georgia, serif;
        font-size: 1.15rem;
        line-height: 1.9;
        font-weight: 400;
        padding: 0.5rem 0;
        letter-spacing: 0.01em;
        max-width: 100%;
    }}

    .flip-card-back-{card_id} .flip-card-text {{
        font-family: 'Noto Sans JP', 'Hiragino Kaku Gothic ProN', sans-serif;
        font-size: 1.05rem;
        line-height: 1.8;
    }}

    .flip-card-hint {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.7rem;
        opacity: 0.4;
        margin-top: 1rem;
        text-transform: lowercase;
        letter-spacing: 0.05em;
    }}

    .flip-card-label {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 0.65rem;
        opacity: 0.35;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        font-weight: 500;
    }}

    /* 学習対象単語のハイライト */
    .target-word {{
        background: linear-gradient(180deg, transparent 60%, #ffd54f 60%);
        padding: 0 2px;
        font-weight: 500;
    }}

    .flip-card-back-{card_id} .target-word {{
        background: linear-gradient(180deg, transparent 60%, #5c6bc0 60%);
        color: #fff;
    }}

    /* iPhone SE向け調整 */
    @media (max-width: 400px) {{
        .flip-card-{card_id} {{
            height: 220px;
        }}
        .flip-card-text {{
            font-size: 1.0rem;
            line-height: 1.7;
        }}
        .flip-card-front-{card_id}, .flip-card-back-{card_id} {{
            height: 220px;
            padding: 0.75rem;
        }}
        .flip-card-back-{card_id} .flip-card-text {{
            font-size: 0.9rem;
        }}
        .flip-card-scroll-container {{
            padding: 0.15rem 0;
        }}
    }}
    </style>

    <div class="flip-container-{card_id}">
        <div class="flip-card-{card_id}" id="flipCard{card_id}">
            <div class="flip-card-front-{card_id}">
                <div class="flip-card-label">English</div>
                <div class="flip-card-scroll-container">
                    <div class="flip-card-text">{escaped_en}</div>
                </div>
                <div class="flip-card-hint">{tap_hint}</div>
            </div>
            <div class="flip-card-back-{card_id}">
                <div class="flip-card-label">日本語</div>
                <div class="flip-card-scroll-container">
                    <div class="flip-card-text">{escaped_jp}</div>
                </div>
                <div class="flip-card-hint">tap to return</div>
            </div>
        </div>
    </div>

    <script>
    (function() {{
        let isTouchMoving = false;
        let touchStartInScrollContainer = false;

        // フリップ処理（タップでのみ発火、スクロール中は無視）
        window.toggleFlip{card_id} = function(e) {{
            if (isTouchMoving || touchStartInScrollContainer) {{
                return;
            }}
            const card = document.getElementById('flipCard{card_id}');
            card.classList.toggle('flipped');
        }};

        // カード内部スクロール制御
        const scrollContainers = document.querySelectorAll('.flip-card-scroll-container');

        scrollContainers.forEach(function(container) {{
            container.addEventListener('touchstart', function(e) {{
                touchStartInScrollContainer = true;
                isTouchMoving = false;
            }}, {{ passive: true }});

            container.addEventListener('touchmove', function(e) {{
                isTouchMoving = true;
            }}, {{ passive: true }});

            container.addEventListener('touchend', function(e) {{
                // スクロールコンテナ内でタッチ終了時、少し遅延してリセット
                setTimeout(function() {{
                    touchStartInScrollContainer = false;
                    isTouchMoving = false;
                }}, 100);
            }}, {{ passive: true }});
        }});

        // カード全体のタッチイベント
        const flipCard = document.getElementById('flipCard{card_id}');
        flipCard.addEventListener('touchstart', function(e) {{
            if (!e.target.closest('.flip-card-scroll-container')) {{
                touchStartInScrollContainer = false;
            }}
            isTouchMoving = false;
        }}, {{ passive: true }});

        flipCard.addEventListener('touchmove', function(e) {{
            isTouchMoving = true;
        }}, {{ passive: true }});

        flipCard.addEventListener('touchend', function(e) {{
            if (!isTouchMoving && !touchStartInScrollContainer) {{
                toggleFlip{card_id}();
            }}
            setTimeout(function() {{
                isTouchMoving = false;
                touchStartInScrollContainer = false;
            }}, 100);
        }}, {{ passive: true }});

        // クリックイベント（デスクトップ用）
        flipCard.addEventListener('click', function(e) {{
            if (!e.target.closest('.flip-card-scroll-container')) {{
                toggleFlip{card_id}();
            }}
        }});

        // スワイプジェスチャー検出（カード外のみ）
        let touchStartX = 0;
        let touchStartY = 0;
        let touchEndX = 0;
        const flipContainer = document.querySelector('.flip-container-{card_id}');

        flipContainer.addEventListener('touchstart', function(e) {{
            touchStartX = e.changedTouches[0].screenX;
            touchStartY = e.changedTouches[0].screenY;
        }}, false);

        flipContainer.addEventListener('touchend', function(e) {{
            touchEndX = e.changedTouches[0].screenX;
            const touchEndY = e.changedTouches[0].screenY;

            // 縦スクロールが主な場合はスワイプ検出しない
            const deltaX = Math.abs(touchEndX - touchStartX);
            const deltaY = Math.abs(touchEndY - touchStartY);

            if (deltaX > deltaY) {{
                handleSwipe();
            }}
        }}, false);

        function handleSwipe() {{
            const swipeThreshold = 50;
            const diff = touchEndX - touchStartX;

            if (Math.abs(diff) > swipeThreshold) {{
                if (diff > 0) {{
                    window.parent.postMessage({{type: 'swipe', direction: 'prev'}}, '*');
                }} else {{
                    window.parent.postMessage({{type: 'swipe', direction: 'next'}}, '*');
                }}
            }}
        }}
    }})();
    </script>
    """

    return flip_card_html


def create_swipe_handler():
    """スワイプジェスチャーのハンドラーJS（Streamlit側で受信）"""
    swipe_js = """
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'swipe') {
            // Streamlitにスワイプイベントを通知
            const direction = e.data.direction;
            // セッション状態を更新するためのワークアラウンド
            // 実際にはボタンクリックをシミュレート
            console.log('Swipe detected:', direction);
        }
    });
    </script>
    """
    return swipe_js


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

def generate_audio_file(text, rate=1.0, lang='en'):
    """サーバーサイドで音声ファイル生成（iOS Chrome用代替案）"""
    try:
        from gtts import gTTS
        import io
        import base64
        
        # British English設定
        tts_lang = 'en-uk' if lang == 'en-GB' else 'en'
        
        # gTTSで音声生成
        tts = gTTS(text=text, lang=tts_lang, slow=(rate < 0.8))
        
        # バイトストリームに保存
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        # Base64エンコード
        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
        
        return audio_base64
        
    except ImportError:
        st.error("GTTSライブラリがインストールされていません")
        return None
    except Exception as e:
        st.error(f"音声生成エラー: {str(e)}")
        return None

def play_server_generated_audio(text, rate=1.0):
    """サーバー生成音声の再生"""
    
    # 音声ファイルを生成
    with st.spinner("🎵 音声を生成中..."):
        audio_base64 = generate_audio_file(text, rate, 'en-uk')
    
    if not audio_base64:
        st.error("音声生成に失敗しました")
        return
    
    # HTML5 Audio要素で再生
    audio_html = f"""
    <div style="margin: 10px 0;">
        <audio controls autoplay style="width: 100%;">
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        <p style="font-size: 12px; color: #666; margin-top: 5px;">
            🎵 サーバー生成音声 (British English)
        </p>
    </div>
    """
    
    st.components.v1.html(audio_html, height=80)
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
    """モバイル最適化音声再生（iframe環境対応）"""
    import html
    escaped_text = html.escape(text).replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')
    
    # iframe環境対応のJavaScript
    html_code = f"""
    <script>
        function playIframeTTS() {{
            console.log('🎯 Starting iframe-optimized TTS...');
            
            // より正確なデバイス・ブラウザ判定
            const ua = navigator.userAgent;
            const isIOS = /iPad|iPhone|iPod/.test(ua);
            const isChromeIOS = /CriOS/.test(ua);  // iOS Chrome専用判定
            const isSafariIOS = /Safari/.test(ua) && !/Chrome/.test(ua) && isIOS;
            const isAndroid = /Android/.test(ua);
            
            console.log(`Device Info:
                iOS: ${{isIOS}}
                iOS Chrome: ${{isChromeIOS}}
                iOS Safari: ${{isSafariIOS}}
                Android: ${{isAndroid}}
                URL: ${{window.location.href}}
            `);
            
            // 既存音声を停止
            try {{
                window.speechSynthesis.cancel();
                if (isIOS) {{
                    // iOS特別処理：複数回cancel
                    setTimeout(() => window.speechSynthesis.cancel(), 50);
                    setTimeout(() => window.speechSynthesis.cancel(), 100);
                }}
            }} catch(e) {{
                console.warn('Cancel failed:', e);
            }}
            
            const utterance = new SpeechSynthesisUtterance("{escaped_text}");
            
            // iOS専用設定（ChromeでもSafariでも同じ処理）
            if (isIOS) {{
                utterance.lang = 'en-US';  // iOS では en-US が最も安定
                utterance.rate = Math.max(0.1, Math.min(2.0, {rate * 0.85})); // iOS向け速度制限
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                
                // iOS向け追加設定
                utterance.voiceURI = 'native';
            }} else if (isAndroid) {{
                utterance.lang = 'en-GB';
                utterance.rate = {rate};
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
            }} else {{
                // Desktop
                utterance.lang = 'en-GB';
                utterance.rate = {rate};
                utterance.pitch = 0.9;
                utterance.volume = 1.0;
            }}
            
            // 音声選択（iOS専用最適化）
            function selectIOSVoice() {{
                const voices = window.speechSynthesis.getVoices();
                console.log(`Available voices: ${{voices.length}}`);
                
                if (voices.length === 0) return null;
                
                if (isIOS) {{
                    // iOS向け音声優先順位（英語系のみ）
                    const preferredNames = [
                        'Samantha',    // US English - 高品質
                        'Alex',        // US English - 標準
                        'Victoria',    // US English - 女性
                        'Daniel',      // UK English - 男性
                        'Kate',        // UK English - 女性
                        'Moira',       // Irish English
                        'Karen',       // Australian English
                    ];
                    
                    // 名前による検索
                    for (const name of preferredNames) {{
                        const voice = voices.find(v => v.name === name);
                        if (voice) {{
                            console.log(`Selected iOS voice by name: ${{voice.name}} (${{voice.lang}})`);
                            return voice;
                        }}
                    }}
                    
                    // 言語による検索
                    const langPriority = ['en-US', 'en-GB', 'en-AU', 'en'];
                    for (const lang of langPriority) {{
                        const voice = voices.find(v => v.lang === lang || v.lang.startsWith(lang));
                        if (voice && voice.localService) {{
                            console.log(`Selected iOS voice by lang: ${{voice.name}} (${{voice.lang}})`);
                            return voice;
                        }}
                    }}
                    
                    // フォールバック：最初の英語音声
                    const enVoice = voices.find(v => v.lang.startsWith('en'));
                    if (enVoice) {{
                        console.log(`iOS fallback voice: ${{enVoice.name}} (${{enVoice.lang}})`);
                        return enVoice;
                    }}
                }}
                
                return null;
            }}
            
            const selectedVoice = selectIOSVoice();
            if (selectedVoice) {{
                utterance.voice = selectedVoice;
            }}
            
            // 成功/失敗フィードバック用関数
            function showFeedback(message, isSuccess = true) {{
                const feedback = document.createElement('div');
                feedback.innerHTML = message;
                feedback.style.cssText = `
                    position: fixed; 
                    top: 20px; 
                    left: 50%; 
                    transform: translateX(-50%);
                    z-index: 999999;
                    background: ${{isSuccess ? '#4CAF50' : '#f44336'}}; 
                    color: white; 
                    padding: 15px 20px;
                    border-radius: 25px; 
                    font-size: 14px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
                    font-family: Arial, sans-serif;
                    text-align: center;
                    max-width: 300px;
                `;
                
                document.body.appendChild(feedback);
                setTimeout(() => {{
                    if (document.body.contains(feedback)) {{
                        document.body.removeChild(feedback);
                    }}
                }}, isSuccess ? 2000 : 5000);
            }}
            
            // イベントハンドラー設定
            utterance.onstart = function() {{
                console.log('✅ Speech started successfully');
                showFeedback('🔊 音声再生開始', true);
            }};
            
            utterance.onend = function() {{
                console.log('✅ Speech completed');
            }};
            
            utterance.onerror = function(event) {{
                console.error('❌ Speech error:', event.error, event);
                
                const errorMessages = {{
                    'not-allowed': '🚫 音声が許可されていません<br>設定 → サイト設定 → 音声を許可',
                    'network': '🌐 ネットワークエラー<br>WiFi接続を確認してください',
                    'synthesis-failed': '🎵 音声合成に失敗<br>別の速度で試してください',
                    'synthesis-unavailable': '❌ 音声機能が利用できません<br>デバイス設定を確認してください',
                    'audio-hardware': '🎧 オーディオハードウェアエラー<br>イヤホン接続を確認してください',
                    'language-unavailable': '🗣️ 指定言語が利用できません<br>デフォルト音声を使用します'
                }};
                
                const message = errorMessages[event.error] || `音声エラー: ${{event.error}}`;
                showFeedback(message, false);
            }};
            
            // iOS専用実行処理
            function executeIOSSpeech() {{
                console.log('🚀 Executing iOS speech...');
                
                try {{
                    // iframe環境での特別処理
                    if (window.location.href.includes('srcdoc')) {{
                        console.log('🔧 iframe environment detected');
                        
                        // トップウィンドウからのアクセスを試行
                        if (window.parent && window.parent.speechSynthesis) {{
                            console.log('📡 Using parent window speechSynthesis');
                            window.parent.speechSynthesis.speak(utterance);
                            return;
                        }}
                    }}
                    
                    // 通常の実行
                    window.speechSynthesis.speak(utterance);
                    console.log('📢 speechSynthesis.speak() executed');
                    
                }} catch (error) {{
                    console.error('❌ Speech execution failed:', error);
                    showFeedback(`実行エラー: ${{error.message}}`, false);
                }}
                
                // 3秒後にタイムアウトチェック
                setTimeout(() => {{
                    if (!window.speechSynthesis.speaking) {{
                        console.warn('⏰ Speech timeout - not speaking after 3 seconds');
                        showFeedback('⏰ 音声開始タイムアウト<br>もう一度お試しください', false);
                    }}
                }}, 3000);
            }}
            
            // 実行（iOS向け遅延）
            if (isIOS) {{
                setTimeout(executeIOSSpeech, 150);
            }} else {{
                executeIOSSpeech();
            }}
        }}
        
        // 音声リスト準備完了後に実行
        function initIframeTTS() {{
            const voices = window.speechSynthesis.getVoices();
            
            if (voices.length === 0) {{
                console.log('⏳ Waiting for voices to load...');
                window.speechSynthesis.onvoiceschanged = function() {{
                    console.log('🔄 Voices loaded, starting TTS');
                    playIframeTTS();
                }};
                
                // 2秒でタイムアウト
                setTimeout(() => {{
                    console.log('⚠️ Voice loading timeout, attempting anyway');
                    playIframeTTS();
                }}, 2000);
            }} else {{
                playIframeTTS();
            }}
        }}
        
        // 初期化
        initIframeTTS();
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
        model = genai.GenerativeModel('gemini-2.5-flash')
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

def safe_html_display(text, highlight_spans=None):
    """安全なHTML表示（XSS対策＋ハイライト機能）"""
    import html
    
    if not highlight_spans:
        # ハイライト対象がない場合は、そのままエスケープして表示
        return html.escape(text)
    
    # テキストを安全に処理してハイライトを適用
    result = ""
    last_end = 0
    
    # ハイライト位置をソート
    spans = sorted(highlight_spans, key=lambda x: x['start'])
    
    for span in spans:
        start, end = span['start'], span['end']
        word = span['word']
        style_class = span.get('class', 'highlight-word')
        
        # 前の部分（エスケープ）
        if start > last_end:
            result += html.escape(text[last_end:start])
        
        # ハイライト部分（安全なスタイル適用）
        escaped_word = html.escape(word)
        if style_class == 'highlight-word':
            result += f'<mark class="vocab-highlight">{escaped_word}</mark>'
        else:
            result += f'<mark class="japanese-highlight">{escaped_word}</mark>'
        
        last_end = end
    
    # 残りの部分（エスケープ）
    if last_end < len(text):
        result += html.escape(text[last_end:])
    
    return result

def find_word_positions(sentence, target_words):
    """文章内の単語位置を検出"""
    import re
    positions = []
    
    for word in target_words:
        # 大文字小文字を無視して検索
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        for match in pattern.finditer(sentence):
            positions.append({
                'start': match.start(),
                'end': match.end(),
                'word': sentence[match.start():match.end()],  # 元の文字ケースを保持
                'class': 'highlight-word'
            })
    
    return positions

def highlight_words_in_sentence(sentence, words_dict, word_master):
    """文章内の学習対象単語をハイライト（完全修正版）"""
    if not words_dict:
        return safe_html_display(sentence)
    
    target_words = list(words_dict.values())
    if not target_words:
        return safe_html_display(sentence)
    
    # 単語位置を検出
    word_positions = find_word_positions(sentence, target_words)
    
    # 重複除去（同じ位置の場合）
    unique_positions = []
    for pos in word_positions:
        if not any(p['start'] == pos['start'] and p['end'] == pos['end'] for p in unique_positions):
            unique_positions.append(pos)
    
    return safe_html_display(sentence, unique_positions)

def highlight_words_in_japanese(japanese_sentence, words_dict, word_master):
    """日本語訳内の対応する単語をハイライト（完全修正版）"""
    if not words_dict or word_master.empty:
        return safe_html_display(japanese_sentence)
    
    japanese_words = []
    
    # word_masterから日本語の意味を取得
    for word_id, english_word in words_dict.items():
        try:
            word_id_int = int(word_id)
            word_info = word_master[word_master['word_id'] == word_id_int]
            if not word_info.empty and 'japanese_meaning' in word_info.columns:
                japanese_meaning = word_info.iloc[0]['japanese_meaning']
                if japanese_meaning and japanese_meaning.strip():
                    japanese_words.append(japanese_meaning.strip())
        except:
            continue
    
    if not japanese_words:
        return safe_html_display(japanese_sentence)
    
    # 日本語単語の位置を検出
    word_positions = find_word_positions(japanese_sentence, japanese_words)
    for pos in word_positions:
        pos['class'] = 'japanese-highlight'
    
    return safe_html_display(japanese_sentence, word_positions)


def main():
    # カスタムCSS読み込み
    load_custom_css()

    # セッション状態初期化
    initialize_session_state()

    # Gemini API初期化を最初に実行
    initialize_gemini()

    # サイドバーに詳細設定を移動（折りたたみ状態）
    with st.sidebar:
        st.markdown("## ⚙️ 詳細設定")

        # API設定状況
        with st.expander("🔑 API設定", expanded=False):
            env_api_key = os.getenv('GOOGLE_API_KEY', '')
            session_api_key = st.session_state.get('gemini_api_key', '')

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

        # 音声設定
        with st.expander("🔊 音声設定", expanded=False):
            if st.button("🎤 利用可能な音声を確認"):
                show_available_voices()

            st.markdown("""
            **音声機能:**
            - 高品質サーバー生成音声
            - 3段階の速度調整
            """)

        # データ読み込み状況
        st.markdown("## 📊 データ状況")
    
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

    # サイドバーにデータ概要（コンパクト表示）
    with st.sidebar:
        st.markdown(f"**📈 統計:** {len(df)}文 / {df['group_id'].nunique()}グループ")
        st.markdown(f"**📚 今日:** {st.session_state.studied_today}文章学習")

    # メインナビゲーション（タブを大きく）
    tab1, tab2, tab3 = st.tabs(["📚 学習", "🎯 シャドーイング", "📊 記録"])
    
    with tab1:
        word_learning_tab(df, word_master)
    
    with tab2:
        shadowing_tab()
    
    with tab3:
        progress_tab(df)

def word_learning_tab(df, word_master):
    """単語学習タブ - iPhone SE向けフリップカードUI"""

    # 設定メニュー（歯車アイコンからアクセス）
    with st.expander("⚙️ 設定", expanded=False):
        col1, col2 = st.columns(2)
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

        # ジャンプ機能
        jump_to = st.number_input(
            "文章番号へジャンプ",
            min_value=1,
            max_value=len(filtered_df) if learning_mode != "特定グループ" else len(filtered_df),
            value=st.session_state.current_sentence_idx + 1,
            step=1
        )
        # 「移動」と「リセット」ボタンを横並び
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("移動", key="jump_btn", use_container_width=True):
                st.session_state.current_sentence_idx = jump_to - 1
                st.rerun()
        with btn_col2:
            if st.button("リセット", key="reset_btn", use_container_width=True):
                st.session_state.current_sentence_idx = 0
                st.session_state.show_translation = False
                if 'shuffled_indices' in st.session_state:
                    del st.session_state.shuffled_indices
                st.rerun()

    # データ準備
    if learning_mode == "特定グループ":
        pass  # already filtered above
    else:
        filtered_df = df.copy()

    if learning_mode == "ランダム":
        if 'shuffled_indices' not in st.session_state or len(st.session_state.shuffled_indices) != len(filtered_df):
            st.session_state.shuffled_indices = list(range(len(filtered_df)))
            random.shuffle(st.session_state.shuffled_indices)
        current_idx = st.session_state.shuffled_indices[st.session_state.current_sentence_idx % len(st.session_state.shuffled_indices)]
    else:
        current_idx = st.session_state.current_sentence_idx % len(filtered_df)

    current_sentence = filtered_df.iloc[current_idx]

    # ========== シンプルな進捗表示 ==========
    current_pos = st.session_state.current_sentence_idx + 1
    total_sentences = len(filtered_df)
    st.markdown(f'<div class="progress-simple">{current_pos} / {total_sentences}</div>', unsafe_allow_html=True)

    # ========== フリップカード ==========
    english_text = current_sentence['sentence_content_en']
    japanese_text = current_sentence['translated_sentence']
    card_id = f"card_{current_idx}"

    # 学習対象単語を取得
    words_dict = parse_words_dict(current_sentence.get('words_contained_dict', '{}'))
    highlight_words = list(words_dict.values()) if words_dict else None

    flip_card_html = create_flip_card(english_text, japanese_text, card_id, highlight_words=highlight_words)
    st.components.v1.html(flip_card_html, height=260)

    # ========== ナビゲーション + 音声ボタン ==========
    col1, col2, col3 = st.columns([1.5, 1, 1.5])

    with col1:
        if st.button("⬅️", key="nav_prev_main", use_container_width=True,
                     disabled=(st.session_state.current_sentence_idx == 0)):
            st.session_state.current_sentence_idx -= 1
            st.rerun()

    with col2:
        # 音声ボタン（速度選択付き）
        if 'audio_speed' not in st.session_state:
            st.session_state.audio_speed = 1.0

        if st.button("🔊", key="play_audio_main", use_container_width=True):
            play_server_generated_audio(english_text, rate=st.session_state.audio_speed)

    with col3:
        if st.button("➡️", key="nav_next_main", use_container_width=True):
            st.session_state.current_sentence_idx += 1
            st.rerun()

    # 音声速度選択（コンパクト）
    speed_options = {"🐌": 0.7, "🎵": 1.0, "🚀": 1.3}
    speed_cols = st.columns(3)
    for i, (icon, rate) in enumerate(speed_options.items()):
        with speed_cols[i]:
            selected = st.session_state.audio_speed == rate
            btn_label = f"{'●' if selected else '○'} {icon}"
            if st.button(btn_label, key=f"speed_{rate}", use_container_width=True):
                st.session_state.audio_speed = rate
                st.rerun()

    # ========== 学習対象単語（コンパクト表示） ==========
    if words_dict:
        words_html = " ".join([f'<span class="word-chip">{word}</span>' for word in words_dict.values()])
        st.markdown(f'<div style="text-align:center; padding:0.5rem 0;">{words_html}</div>', unsafe_allow_html=True)

    # ========== 大きな理解度ボタン ==========
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

    # 理解度記録と次の文章へ
    if understanding_level:
        sentence_key = f"{current_sentence['group_id']}_{current_sentence['sentence_id']}"
        st.session_state.learning_progress[sentence_key] = understanding_level
        st.session_state.studied_today += 1
        st.session_state.current_sentence_idx += 1

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

        # 設定（折りたたみ）
        with st.expander("⚙️ 設定", expanded=False):
            jump_to = st.number_input(
                "文番号へジャンプ",
                min_value=1,
                max_value=len(content),
                value=current_idx + 1,
                step=1,
                key="shadowing_jump"
            )
            # 「移動」と「新しい記事を生成」ボタンを横並び
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

        # ========== シンプルな進捗表示 ==========
        current_pos = current_idx + 1
        total_sentences = len(content)
        st.markdown(f'<div class="progress-simple">{current_pos} / {total_sentences}</div>', unsafe_allow_html=True)

        # ========== フリップカード ==========
        current_sentence = content[current_idx]
        card_id = f"shadow_{current_idx}"

        flip_card_html = create_flip_card(
            current_sentence["english"],
            current_sentence["japanese"],
            card_id
        )
        st.components.v1.html(flip_card_html, height=260)

        # ========== ナビゲーション + 音声ボタン ==========
        col1, col2, col3 = st.columns([1.5, 1, 1.5])

        with col1:
            if st.button("⬅️", key="shadowing_prev", use_container_width=True,
                         disabled=(current_idx == 0)):
                st.session_state.current_shadowing_idx -= 1
                st.rerun()

        with col2:
            # 音声ボタン
            if 'shadowing_audio_speed' not in st.session_state:
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

        # 音声速度選択（コンパクト）
        speed_options = {"🐌": 0.7, "🎵": 1.0, "🚀": 1.3}
        speed_cols = st.columns(3)
        for i, (icon, rate) in enumerate(speed_options.items()):
            with speed_cols[i]:
                selected = st.session_state.shadowing_audio_speed == rate
                btn_label = f"{'●' if selected else '○'} {icon}"
                if st.button(btn_label, key=f"shadowing_speed_{rate}", use_container_width=True):
                    st.session_state.shadowing_audio_speed = rate
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