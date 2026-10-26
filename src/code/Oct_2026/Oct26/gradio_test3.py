import whisper
from gtts import gTTS
import gradio as gr
import os
import tempfile

# Whisper 모델 로드 (다국어 자동 번역)
model = whisper.load_model("base")

# 감정 분석 / TTS 는 무거워서 함수 실행 시 따로 로드(lazy load)
_sentiment = None
_tts = None

def get_sentiment():
    global _sentiment
    if _sentiment is None:
        from transformers import pipeline
        _sentiment = pipeline("sentiment-analysis")
    return _sentiment

def get_tts():
    global _tts
    if _tts is None:
        from TTS.api import TTS
        _tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    return _tts

# 종합 처리 함수
def full_pipeline(audio_path):
    if audio_path is None:
        return "녹음된 오디오가 없어요. 녹음을 끝낸 뒤 Submit을 눌러주세요.", None

    # 1. Whisper로 영어로 자동 번역
    result = model.transcribe(audio_path, task="translate")
    translated_text = result["text"].strip()
    if not translated_text:
        return "인식/번역된 내용이 없어요", None

    # 2. 감정 분석
    s = get_sentiment()(translated_text)[0]
    label = s["label"]
    score = round(s["score"], 2)

    # 3. TTS 음성 합성 (Coqui, 영어)
    tts = get_tts()
    fd, tts_out_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    tts.tts_to_file(translated_text, file_path=tts_out_path)

    analysis = f"번역된 문장: {translated_text}\n감정: {label} (score: {score})"
    return analysis, tts_out_path

def transcribe_and_tts(audio_path):
    if audio_path is None:
        return "녹음된 오디오가 없어요. 녹음을 끝낸 뒤 Submit을 눌러주세요.", None
    # Whisper STT 수행
    result = model.transcribe(audio_path, language="en", fp16=False)
    transcript = result["text"].strip()
    if not transcript:
        return "인식된 내용이 없어요", None

    fd, tts_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    # gTTS로 음성 생성
    gTTS(text=transcript, lang="ko").save(tts_path)

    return transcript, tts_path

demo1 = gr.Interface(
    fn=transcribe_and_tts,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="말해보세요"),
    outputs=[
        gr.Textbox(label="인식된 텍스트"),
        gr.Audio(label="다시 음성으로", autoplay=True),
    ],
    title="Whisper + gTTS 무료 음성 인터페이스",
    description="마이크로 말하면 텍스트로 인식하고 다시 음성으로 들려 줍니다.( 완전 로컬, 완전 무료)",
)

demo2 = gr.Interface(
    fn=full_pipeline,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="한국어로 말하세요"),
    outputs=[
        gr.Textbox(label="분석 결과 (번역 + 감정 분석)"),
        gr.Audio(label="영어 음성 재생", autoplay=True),
    ],
    title="자동 번역 + 감정 분석 + 고급 음성 합성",
    description="Whisper로 한국어를 영어로 번역하고, 감정 분석 후 고급 TTS로 음성 출력까지 해주는 AI 파이프라인",
)

# 두 개의 앱을 탭으로 묶어서 실행
app = gr.TabbedInterface([demo1, demo2], ["STT+TTS", "번역+감정+TTS"])

if __name__ == "__main__":
    app.launch()
