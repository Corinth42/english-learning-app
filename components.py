import html as html_module
import re


def create_flip_card(english_text, japanese_text, card_id, show_tap_hint=True, highlight_words=None):
    """フリップカード用HTML/CSS/JSを生成（タップで英文↔和訳を切り替え）"""
    def escape_and_highlight(text, words_to_highlight=None):
        escaped = html_module.escape(text)
        if words_to_highlight:
            for word in words_to_highlight:
                pattern = re.compile(re.escape(html_module.escape(word)), re.IGNORECASE)
                escaped = pattern.sub(
                    f'<span class="target-word">{html_module.escape(word)}</span>',
                    escaped,
                )
        return escaped

    escaped_en = escape_and_highlight(english_text, highlight_words)
    escaped_jp = html_module.escape(japanese_text)
    tap_hint = "tap to translate" if show_tap_hint else ""

    flip_card_html = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@400;500&family=Noto+Sans+JP:wght@400;500&display=swap');

    .flip-container-{card_id} {{ width: 100%; margin: 0.5rem 0; position: relative; }}
    .flip-card-{card_id} {{ position: relative; width: 100%; min-height: 200px; cursor: pointer; -webkit-tap-highlight-color: transparent; }}
    .flip-card-face-{card_id} {{
        width: 100%; min-height: 200px; border-radius: 8px; padding: 1rem;
        display: flex; flex-direction: column; align-items: center; text-align: center; box-sizing: border-box;
        transition: opacity 0.3s ease, transform 0.2s ease;
    }}
    .flip-card-front-{card_id} {{ background: #fafafa; color: #1a1a1a; border: 2px solid #e0e0e0; }}
    .flip-card-back-{card_id} {{ background: #1a1a1a; color: #fafafa; border: 2px solid #333; display: none; }}
    .flip-card-{card_id}.flipped .flip-card-front-{card_id} {{ display: none; }}
    .flip-card-{card_id}.flipped .flip-card-back-{card_id} {{ display: flex; }}
    .flip-card-scroll-container-{card_id} {{
        flex: 1; width: 100%; max-height: 180px; overflow-y: auto; overflow-x: hidden;
        -webkit-overflow-scrolling: touch; padding: 0.5rem; text-align: center;
    }}
    .flip-card-text-{card_id} {{
        font-family: 'Source Serif 4', Georgia, serif; font-size: 1.15rem; line-height: 1.9;
        font-weight: 400; padding: 0.5rem 0; letter-spacing: 0.01em; max-width: 100%; word-wrap: break-word;
    }}
    .flip-card-back-{card_id} .flip-card-text-{card_id} {{
        font-family: 'Noto Sans JP', 'Hiragino Kaku Gothic ProN', sans-serif; font-size: 1.05rem; line-height: 1.8;
    }}
    .flip-card-hint-{card_id} {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.75rem; opacity: 0.5;
        margin-top: 0.75rem; text-transform: lowercase; letter-spacing: 0.05em;
        padding: 0.25rem 0.75rem; background: rgba(0,0,0,0.05); border-radius: 12px;
    }}
    .flip-card-back-{card_id} .flip-card-hint-{card_id} {{ background: rgba(255,255,255,0.1); }}
    .flip-card-label-{card_id} {{
        font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 0.7rem; opacity: 0.4;
        margin-bottom: 0.5rem; text-transform: uppercase; letter-spacing: 0.15em; font-weight: 600;
    }}
    .target-word {{ background: linear-gradient(180deg, transparent 60%, #ffd54f 60%); padding: 0 2px; font-weight: 500; }}
    .flip-card-back-{card_id} .target-word {{ background: linear-gradient(180deg, transparent 60%, #5c6bc0 60%); color: #fff; }}
    .flip-card-{card_id}:active .flip-card-face-{card_id} {{ transform: scale(0.98); }}
    @media (max-width: 400px) {{
        .flip-card-face-{card_id} {{ min-height: 180px; padding: 0.75rem; }}
        .flip-card-text-{card_id} {{ font-size: 1.0rem; line-height: 1.7; }}
        .flip-card-back-{card_id} .flip-card-text-{card_id} {{ font-size: 0.95rem; }}
        .flip-card-scroll-container-{card_id} {{ max-height: 150px; }}
    }}
    </style>
    <div class="flip-container-{card_id}">
        <div class="flip-card-{card_id}" id="flipCard{card_id}" onclick="toggleFlipCard{card_id}(event)">
            <div class="flip-card-face-{card_id} flip-card-front-{card_id}">
                <div class="flip-card-label-{card_id}">English</div>
                <div class="flip-card-scroll-container-{card_id}" onclick="event.stopPropagation()">
                    <div class="flip-card-text-{card_id}">{escaped_en}</div>
                </div>
                <div class="flip-card-hint-{card_id}">{tap_hint}</div>
            </div>
            <div class="flip-card-face-{card_id} flip-card-back-{card_id}">
                <div class="flip-card-label-{card_id}">日本語</div>
                <div class="flip-card-scroll-container-{card_id}" onclick="event.stopPropagation()">
                    <div class="flip-card-text-{card_id}">{escaped_jp}</div>
                </div>
                <div class="flip-card-hint-{card_id}">tap to return</div>
            </div>
        </div>
    </div>
    <script>
    function toggleFlipCard{card_id}(event) {{
        if (event.target.closest('.flip-card-scroll-container-{card_id}')) return;
        document.getElementById('flipCard{card_id}').classList.toggle('flipped');
    }}
    </script>
    """
    return flip_card_html


def create_swipe_handler():
    """スワイプジェスチャーのハンドラーJS（Streamlit側で受信）"""
    return """
    <script>
    window.addEventListener('message', function(e) {
        if (e.data && e.data.type === 'swipe') {
            const direction = e.data.direction;
            console.log('Swipe detected:', direction);
        }
    });
    </script>
    """
