import glob
import json
import os
import pandas as pd
import streamlit as st


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
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


@st.cache_data
def load_word_master(data_dir="data"):
    """単語マスターデータを読み込み"""
    word_master_path = os.path.join(data_dir, "word_master.csv")
    try:
        if os.path.exists(word_master_path):
            return pd.read_csv(word_master_path)
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
        if isinstance(words_str, str):
            return json.loads(words_str.replace("'", '"'))
        return {}
    except Exception:
        return {}
