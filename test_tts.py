import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise RuntimeError("GROQ_API_KEY was not found in .env")
client = Groq(api_key=api_key)

TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"
OUTPUT_PATH = "test_tts.wav"


def generate_speech(text,output_path=OUTPUT_PATH,):
    text = text.strip()
    if not text:
        raise ValueError("Text cannot be empty.")
    if len(text) > 200:
        raise ValueError("Text must be 200 characters or less.")
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice=TTS_VOICE,
        input=text,
        response_format="wav",
    )
    response.write_to_file(output_path)
    print(
        f"Saved: "
        f"{os.path.abspath(output_path)}"
    )

if __name__ == "__main__":
    print("Generating speech...")
    generate_speech("The monitor is showing a warning.")
    print("TTS test completed.")