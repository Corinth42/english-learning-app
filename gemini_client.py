import os
import streamlit as st
import google.generativeai as genai

from config import GENRE_PROMPTS


def initialize_gemini():
    """Gemini APIの初期化"""
    env_api_key = os.getenv("GOOGLE_API_KEY", "")

    if env_api_key and "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = env_api_key
    elif "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = ""

    if st.session_state.gemini_api_key:
        try:
            genai.configure(api_key=st.session_state.gemini_api_key)
            return True
        except Exception as e:
            st.error(f"Gemini API初期化エラー: {str(e)}")
            return False
    return False


def generate_content_with_gemini(genre, topic):
    """Gemini APIでコンテンツ生成"""
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = GENRE_PROMPTS[genre]["prompt"].format(topic=topic)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        st.error(f"コンテンツ生成エラー: {str(e)}")
        return None


def parse_generated_content(content):
    """生成されたコンテンツを英文と日本語訳に分割"""
    lines = content.strip().split("\n")
    parsed_content = []
    current_en = ""
    current_jp = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line[0].isupper() and any(c.isalpha() for c in line):
            if current_en and current_jp:
                parsed_content.append({"english": current_en, "japanese": current_jp})
            current_en = line
            current_jp = ""
        else:
            current_jp = line

    if current_en and current_jp:
        parsed_content.append({"english": current_en, "japanese": current_jp})

    return parsed_content
