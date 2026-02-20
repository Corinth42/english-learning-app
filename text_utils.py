import html
import re


def safe_html_display(text, highlight_spans=None):
    """安全なHTML表示（XSS対策＋ハイライト機能）"""
    if not highlight_spans:
        return html.escape(text)

    result = ""
    last_end = 0
    spans = sorted(highlight_spans, key=lambda x: x["start"])

    for span in spans:
        start, end = span["start"], span["end"]
        word = span["word"]
        style_class = span.get("class", "highlight-word")

        if start > last_end:
            result += html.escape(text[last_end:start])

        escaped_word = html.escape(word)
        if style_class == "highlight-word":
            result += f'<mark class="vocab-highlight">{escaped_word}</mark>'
        else:
            result += f'<mark class="japanese-highlight">{escaped_word}</mark>'

        last_end = end

    if last_end < len(text):
        result += html.escape(text[last_end:])

    return result


def find_word_positions(sentence, target_words):
    """文章内の単語位置を検出"""
    positions = []
    for word in target_words:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        for match in pattern.finditer(sentence):
            positions.append({
                "start": match.start(),
                "end": match.end(),
                "word": sentence[match.start() : match.end()],
                "class": "highlight-word",
            })
    return positions


def highlight_words_in_sentence(sentence, words_dict, word_master):
    """文章内の学習対象単語をハイライト"""
    if not words_dict:
        return safe_html_display(sentence)

    target_words = list(words_dict.values())
    if not target_words:
        return safe_html_display(sentence)

    word_positions = find_word_positions(sentence, target_words)
    unique_positions = []
    for pos in word_positions:
        if not any(p["start"] == pos["start"] and p["end"] == pos["end"] for p in unique_positions):
            unique_positions.append(pos)

    return safe_html_display(sentence, unique_positions)


def highlight_words_in_japanese(japanese_sentence, words_dict, word_master):
    """日本語訳内の対応する単語をハイライト"""
    if not words_dict or word_master.empty:
        return safe_html_display(japanese_sentence)

    japanese_words = []
    for word_id, english_word in words_dict.items():
        try:
            word_id_int = int(word_id)
            word_info = word_master[word_master["word_id"] == word_id_int]
            if not word_info.empty and "japanese_meaning" in word_info.columns:
                japanese_meaning = word_info.iloc[0]["japanese_meaning"]
                if japanese_meaning and str(japanese_meaning).strip():
                    japanese_words.append(str(japanese_meaning).strip())
        except Exception:
            continue

    if not japanese_words:
        return safe_html_display(japanese_sentence)

    word_positions = find_word_positions(japanese_sentence, japanese_words)
    for pos in word_positions:
        pos["class"] = "japanese-highlight"

    return safe_html_display(japanese_sentence, word_positions)
