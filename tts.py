# enVocab TTS: Piper (英国男性・高品質) + gTTS フォールバック（無償）
#
# 【英国男性・高品質にするには】
# 1. pip install piper-tts
# 2. Hugging Face から en_GB 男性モデルをダウンロード:
#    https://huggingface.co/rhasspy/piper-voices/tree/v1.0.0/en/en_GB/northern_english_male
#    (.onnx と .onnx.json の2ファイル)
# 3. プロジェクト直下に voices/ を作り、その中に配置するか、
#    環境変数 PIPER_VOICE_PATH で .onnx のパスを指定
# Piper が使えない場合は自動で gTTS (British English) にフォールバックします。

import io
import base64
import os
from pathlib import Path

# Streamlit は tts 内で import（循環回避）
def _st():
    import streamlit as st
    return st


def _piper_available():
    try:
        import piper
        return True
    except ImportError:
        return False


def _get_piper_voice_path():
    """PIPER_VOICE_PATH または voices/ 内の en_GB 男性モデルを返す。"""
    env_path = os.getenv("PIPER_VOICE_PATH", "").strip()
    if env_path and Path(env_path).with_suffix(".onnx").exists():
        return str(Path(env_path).with_suffix(".onnx"))
    if env_path and Path(env_path).exists():
        return env_path
    voices_dir = Path(__file__).resolve().parent / "voices"
    if not voices_dir.exists():
        return None
    # en_GB の男性候補: northern_english_male, alan, aru
    for name in ("northern_english_male", "alan", "aru"):
        for onnx in voices_dir.rglob(f"*{name}*.onnx"):
            if onnx.with_suffix(onnx.suffix + ".json").exists():
                return str(onnx)
            json_alt = Path(str(onnx) + ".json")
            if json_alt.exists():
                return str(onnx)
    return None


def generate_audio_file(text: str, rate: float = 1.0, lang: str = "en") -> tuple[str | None, str]:
    """
    サーバーで音声を生成し (Base64文字列, MIMEタイプ) で返す。
    Piper（英国男性）が利用可能なら優先、否则 gTTS（British）にフォールバック。
    """
    if not text or not text.strip():
        return None, "audio/mp3"

    # Piper: 英国男性、rate は 1.0 に近いときのみ使用（Piper は length_scale 未対応のため）
    use_piper = _piper_available() and (0.85 <= rate <= 1.15)
    voice_path = _get_piper_voice_path() if use_piper else None

    if voice_path:
        try:
            from piper import PiperVoice
            import wave

            config_path = Path(voice_path).with_suffix(Path(voice_path).suffix + ".json")
            if not config_path.exists():
                config_path = Path(voice_path + ".json")
            if not config_path.exists():
                raise FileNotFoundError(f"Piper config not found: {config_path}")

            voice = PiperVoice.load(voice_path, config_path=str(config_path))
            buf = io.BytesIO()
            chunks = list(voice.synthesize_stream_raw(text.strip(), sentence_silence=0.0))
            if not chunks:
                raise ValueError("Piper produced no audio")
            wav_bytes = b"".join(chunks)
            sample_rate = getattr(voice.config, "sample_rate", 22050)
            with wave.open(buf, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(sample_rate)
                wav.writeframes(wav_bytes)
            buf.seek(0)
            return base64.b64encode(buf.getvalue()).decode(), "audio/wav"
        except Exception:
            pass

    # gTTS フォールバック（British English）
    try:
        from gtts import gTTS

        tts_lang = "en-uk" if lang in ("en-GB", "en-uk") else "en"
        tts = gTTS(text=text, lang=tts_lang, slow=(rate < 0.8))
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return base64.b64encode(audio_buffer.getvalue()).decode(), "audio/mp3"
    except ImportError:
        _st().error("gTTS がインストールされていません。pip install gtts")
        return None, "audio/mp3"
    except Exception as e:
        _st().error(f"音声生成エラー: {e}")
        return None, "audio/mp3"


def play_server_generated_audio(text: str, rate: float = 1.0) -> None:
    """サーバー生成音声を再生（British English）。"""
    st = _st()
    with st.spinner("🎵 音声を生成中..."):
        audio_base64, mime = generate_audio_file(text, rate, "en-uk")
    if not audio_base64:
        st.error("音声生成に失敗しました")
        return
    audio_html = f"""
    <div style="margin: 10px 0;">
        <audio controls autoplay style="width: 100%;">
            <source src="data:{mime};base64,{audio_base64}" type="{mime}">
            Your browser does not support the audio element.
        </audio>
        <p style="font-size: 12px; color: #666; margin-top: 5px;">
            🎵 サーバー生成音声 (British English)
        </p>
    </div>
    """
    st.components.v1.html(audio_html, height=80)


def show_available_voices() -> None:
    """利用可能なブラウザ音声一覧を表示。"""
    st = _st()
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


def play_text_to_speech(text: str, rate: float = 1.0) -> None:
    """モバイル最適化のブラウザ TTS（iframe 対応）。"""
    import html as html_module
    st = _st()
    escaped_text = (
        html_module.escape(text).replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    )

    html_code = f"""
    <script>
        function playIframeTTS() {{
            console.log('🎯 Starting iframe-optimized TTS...');

            const ua = navigator.userAgent;
            const isIOS = /iPad|iPhone|iPod/.test(ua);
            const isChromeIOS = /CriOS/.test(ua);
            const isSafariIOS = /Safari/.test(ua) && !/Chrome/.test(ua) && isIOS;
            const isAndroid = /Android/.test(ua);

            try {{
                window.speechSynthesis.cancel();
                if (isIOS) {{
                    setTimeout(() => window.speechSynthesis.cancel(), 50);
                    setTimeout(() => window.speechSynthesis.cancel(), 100);
                }}
            }} catch(e) {{ console.warn('Cancel failed:', e); }}

            const utterance = new SpeechSynthesisUtterance("{escaped_text}");

            if (isIOS) {{
                utterance.lang = 'en-US';
                utterance.rate = Math.max(0.1, Math.min(2.0, {rate * 0.85}));
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
                utterance.voiceURI = 'native';
            }} else if (isAndroid) {{
                utterance.lang = 'en-GB';
                utterance.rate = {rate};
                utterance.pitch = 1.0;
                utterance.volume = 1.0;
            }} else {{
                utterance.lang = 'en-GB';
                utterance.rate = {rate};
                utterance.pitch = 0.9;
                utterance.volume = 1.0;
            }}

            function selectIOSVoice() {{
                const voices = window.speechSynthesis.getVoices();
                if (voices.length === 0) return null;
                if (isIOS) {{
                    const preferredNames = ['Samantha', 'Alex', 'Victoria', 'Daniel', 'Kate', 'Moira', 'Karen'];
                    for (const name of preferredNames) {{
                        const voice = voices.find(v => v.name === name);
                        if (voice) return voice;
                    }}
                    const langPriority = ['en-US', 'en-GB', 'en-AU', 'en'];
                    for (const lang of langPriority) {{
                        const voice = voices.find(v => v.lang === lang || v.lang.startsWith(lang));
                        if (voice && voice.localService) return voice;
                    }}
                    return voices.find(v => v.lang.startsWith('en')) || null;
                }}
                return null;
            }}

            const selectedVoice = selectIOSVoice();
            if (selectedVoice) utterance.voice = selectedVoice;

            function showFeedback(message, isSuccess = true) {{
                const feedback = document.createElement('div');
                feedback.innerHTML = message;
                feedback.style.cssText = `
                    position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
                    z-index: 999999; background: ${{isSuccess ? '#4CAF50' : '#f44336'}};
                    color: white; padding: 15px 20px; border-radius: 25px; font-size: 14px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.3); font-family: Arial, sans-serif;
                    text-align: center; max-width: 300px;
                `;
                document.body.appendChild(feedback);
                setTimeout(() => {{ if (document.body.contains(feedback)) document.body.removeChild(feedback); }}, isSuccess ? 2000 : 5000);
            }}

            utterance.onstart = () => {{ console.log('✅ Speech started'); showFeedback('🔊 音声再生開始', true); }};
            utterance.onend = () => console.log('✅ Speech completed');
            utterance.onerror = function(event) {{
                console.error('❌ Speech error:', event.error);
                const messages = {{
                    'not-allowed': '🚫 音声が許可されていません',
                    'network': '🌐 ネットワークエラー',
                    'synthesis-failed': '🎵 音声合成に失敗',
                    'synthesis-unavailable': '❌ 音声機能が利用できません',
                    'audio-hardware': '🎧 オーディオエラー',
                    'language-unavailable': '🗣️ 指定言語が利用できません'
                }};
                showFeedback(messages[event.error] || '音声エラー', false);
            }};

            function executeIOSSpeech() {{
                try {{
                    if (window.location.href.includes('srcdoc') && window.parent && window.parent.speechSynthesis) {{
                        window.parent.speechSynthesis.speak(utterance);
                        return;
                    }}
                    window.speechSynthesis.speak(utterance);
                }} catch (error) {{
                    showFeedback('実行エラー: ' + error.message, false);
                }}
                setTimeout(() => {{
                    if (!window.speechSynthesis.speaking)
                        showFeedback('⏰ 音声開始タイムアウト', false);
                }}, 3000);
            }}

            if (isIOS) setTimeout(executeIOSSpeech, 150);
            else executeIOSSpeech();
        }}

        function initIframeTTS() {{
            if (window.speechSynthesis.getVoices().length === 0) {{
                window.speechSynthesis.onvoiceschanged = () => playIframeTTS();
                setTimeout(() => playIframeTTS(), 2000);
            }} else playIframeTTS();
        }}
        initIframeTTS();
    </script>
    """
    st.components.v1.html(html_code, height=0)
