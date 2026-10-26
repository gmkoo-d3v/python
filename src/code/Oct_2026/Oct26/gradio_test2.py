import whisper
from gtts import gTTS
import gradio as gr
import os
import tempfile

model = whisper.load_model("base")



def transcribe_and_tts(audio_path):
    if audio_path is None:
        return '녹음된 오디오가 없어요. 녹음을 끝낸 뒤 Submit을 눌러주세요.', None
    # Whisper STT 수행
    result = model.transcribe(audio_path, language='en', fp16=False)
    transcript = result["text"].strip()
    if not transcript:
        return "인식된 내용이 없어요", None

    fd,tts_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)

    # gTTS로 음성 생성
    gTTS(text=transcript, lang='ko').save(tts_path)

    return transcript, tts_path

app = gr.Interface(
    fn=transcribe_and_tts,
    inputs=gr.Audio(sources=["microphone"], type="filepath", label="말해보세요"),
    outputs=[
        gr.Textbox(label="인식된 텍스트"),
        gr.Audio(label='다시 음성으로', autoplay=True),
    ],
    title="Whisper + gTTS 무료 음성 인터페이스",
    description="마이크로 말하면 텍스트로 인식하고 다시 음성으로 들려 줍니다.( 완전 로컬, 완전 무료)"
)

# 앱 실행
app.launch()
