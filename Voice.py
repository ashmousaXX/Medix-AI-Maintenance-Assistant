import os
import torch

try:
    import sounddevice as sd
    import soundfile as sf
    _MIC_AVAILABLE = True
except OSError:
    _MIC_AVAILABLE = False

from dotenv import load_dotenv
from groq import Groq
from transformers import pipeline
from rag import answer_query

STT_MODEL = "openai/whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
RECORDINGS_DIR = "voice_recordings"

os.makedirs(RECORDINGS_DIR,exist_ok=True,)

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")

groq_client = Groq(
    api_key=GROQ_API_KEY
)

print(f"Loading STT model: {STT_MODEL}")
stt = pipeline(
    "automatic-speech-recognition",
    model=STT_MODEL,
    device=-1,
    dtype=torch.float32,
)

print("STT model loaded.")

def record_audio(duration=RECORD_SECONDS,):
    if not _MIC_AVAILABLE:
        raise RuntimeError(
            "Microphone recording isn't available in this environment. "
            "Use the Streamlit st.audio_input widget instead when "
            "running as a web app."
        )
    print(
        f"Recording for {duration} seconds..."
    )

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    path = os.path.join(RECORDINGS_DIR,"input.wav")
    sf.write(
        path,
        audio,
        SAMPLE_RATE,
    )
    return path

def transcribe_audio(audio_path,):
    result = stt(
        audio_path,
        generate_kwargs={
            "language": "english",
            "task": "transcribe",
        },
    )
    return result["text"].strip()

def generate_speech(text,):
    text = text.strip()
    if not text:
        return None
    output_path = os.path.join(RECORDINGS_DIR,"answer.wav")
    response = groq_client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    response.write_to_file( output_path)

    print(
        f"Audio saved: {output_path}"
    )
    return output_path

def voice_query():
    audio_path = record_audio()
    print(
        "Transcribing..."
    )
    text = transcribe_audio(audio_path)

    print()
    print(
        "User:",
        text
    )
    if not text:
        print("No speech detected.")
        return

    print()
    print(
        "Running RAG..."
    )
    result = answer_query(query=text,top_k=5)
    answer = result.get(
        "answer",
        ""
    )
    speech_text = result.get(
        "speech_answer",
        ""
    )

    detected_device = result.get(
        "detected_device",
        "unknown"
    )
    print()
    print(
        "Detected device:",
        detected_device
    )
    print()
    print(
        "Answer:"
    )
    print(
        answer
    )

    # TTS
    if speech_text:
        print()
        print(
            "Speech:"
        )
        print(
            speech_text
        )
        print()
        print(
            "Generating speech..."
        )
        generate_speech(
            speech_text
        )
if __name__ == "__main__":
    voice_query()