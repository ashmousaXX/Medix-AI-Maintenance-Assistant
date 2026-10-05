import os
import time
import uuid
import re
import struct
import tempfile
import asyncio
import edge_tts
import hashlib

from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, send_file, after_this_request
from dotenv import load_dotenv
from groq import Groq
from rag import answer_query
from llm import generate_answer
from preprocessing import extract_pdf_text, clean_text
EDGE_VOICE = "en-US-AndrewNeural"   

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STT_MODEL = "whisper-large-v3-turbo"
TTS_MODEL = "canopylabs/orpheus-v1-english"
TTS_VOICE = "troy"
RECORDINGS_DIR = os.path.join(BASE_DIR, "voice_recordings")
os.makedirs(RECORDINGS_DIR, exist_ok=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found in .env")
groq_client = Groq(api_key=GROQ_API_KEY)

app = Flask(__name__)

def fix_wav_header(path):
    with open(path, "r+b") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(4)
        f.write(struct.pack("<I", file_size - 8))
        f.seek(12)
        while True:
            chunk_id = f.read(4)
            if len(chunk_id) < 4:
                break
            chunk_size_bytes = f.read(4)
            chunk_size = struct.unpack("<I", chunk_size_bytes)[0]
            if chunk_id == b"data":
                data_start = f.tell()
                real_data_size = file_size - data_start
                f.seek(data_start - 4)
                f.write(struct.pack("<I", real_data_size))
                break
            f.seek(chunk_size, os.SEEK_CUR)

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "Medix-ui.html")

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})

CLOSING_RE = re.compile(
    r"^\s*(ok(ay)?[,. ]*)?(thanks|thank you|thanks a lot|thank you so much|that'?s all|bye|goodbye)[\s.!,]*$",
    re.I,
)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"error": "query is required"}), 400

    if len(query) > 1000:
        return jsonify({"error": "Query is too long."}), 400
    
    if CLOSING_RE.match(query):
        msg = "You're welcome! Good luck with the repair."
        return jsonify({
            "answer": msg,
            "speech_answer": msg,
            "device": None,
            "retrieval_type": "closing",
        })
    try:
        t0 = time.time()
        result = answer_query(
            query=query,
            top_k=8,
        )
        print(f"[timing] /api/ask (retrieval + LLM) took {time.time() - t0:.2f}s")

    except Exception as error:
        print(f"[ERROR] /api/ask failed: {error}")
        return jsonify({
            "error": "The assistant couldn't process this question right now. Please try again in a moment."
        }), 502
    return jsonify(
        {
            "answer": result.get("answer", ""),
            "speech_answer": result.get("speech_answer", ""),
            "device": result.get("detected_device"),
            "retrieval_type": result.get("retrieval_type"),
        }
    )

EDGE_VOICE = "en-US-GuyNeural"

async def _edge_save(text, path):
    await edge_tts.Communicate(
        text, EDGE_VOICE, rate="+8%", pitch="+0Hz"
    ).save(path)

CACHE_DIR = os.path.join(BASE_DIR, "tts_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

@app.route("/api/speech", methods=["POST"])
def speech():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "text is required"}), 400

    key = hashlib.md5(f"{EDGE_VOICE}:{text}".encode()).hexdigest()
    final_path = os.path.join(CACHE_DIR, key + ".mp3")

    if not os.path.exists(final_path):
        tmp_path = final_path + f".{uuid.uuid4().hex}.tmp"
        try:
            t0 = time.time()
            asyncio.run(_edge_save(text, tmp_path))
            os.replace(tmp_path, final_path)
            print(f"[timing] /api/speech (edge-tts, {len(text)} chars) took {time.time() - t0:.2f}s")
        except Exception as error:
            print(f"[ERROR] /api/speech failed: {error}")
            try:
                os.remove(tmp_path)
            except OSError:
                pass
            return jsonify({"error": "Text-to-speech generation failed. Please try again."}), 502
    else:
        print("[timing] /api/speech served from cache")
    return send_file(final_path, mimetype="audio/mpeg")

@app.route("/api/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_file = request.files["audio"]
    filename = audio_file.filename or "input.webm"

    try:
        t0 = time.time()
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, audio_file.read()),
            model=STT_MODEL,
            language="en",
            prompt="Technical maintenance question about a ventilator or patient monitor.",
            temperature=0,
        )
        text = (transcription.text or "").strip()
        text = re.sub(r"\s*thanks for watching[.!]?\s*$", "", text, flags=re.I).strip()
        print(f"[timing] /api/transcribe (Groq STT) took {time.time() - t0:.2f}s")
        print(f"[transcript] {text}")
    except Exception as error:
        print(f"[ERROR] /api/transcribe failed: {error}")
        return jsonify({"error": "Couldn't transcribe the audio. Please try again."}), 502
    return jsonify({"text": text})

MAX_FILE_CONTEXT_CHARS = 12000
MAX_FILE_PAGES = 8
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "on", "for", "and", "or", "with", "if", "has", "have", "this",
    "that", "from", "by", "as", "be", "it", "may", "can", "what",
    "should", "do", "i", "my", "me",
}
FAULT_SECTION_MARKERS = (
    "troubleshooting", "malfunction", "symptom or condition",
    "possible cause", "fault", "error code",
)

def _score_page(page_text, query_words):
    page_lower = page_text.lower()
    meaningful_words = query_words - STOPWORDS
    if not meaningful_words:
        meaningful_words = query_words  

    score = sum(page_lower.count(w) for w in meaningful_words)
    if any(marker in page_lower for marker in FAULT_SECTION_MARKERS):
        score += 20  
    return score

def _extract_text_from_upload(file_storage):
    filename = file_storage.filename or ""
    suffix = Path(filename).suffix.lower()

    if suffix == ".txt":
        raw = file_storage.read().decode("utf-8", errors="ignore")
        return [{"page": 1, "text": clean_text(raw)}]

    if suffix == ".docx":
        try:
            import docx
        except ImportError:
            raise RuntimeError(
                "Reading .docx files needs the 'python-docx' package. "
                "Install it with: pip install python-docx"
            )
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
            tmp.write(file_storage.read())
            tmp_path = tmp.name
        try:
            document = docx.Document(tmp_path)
            full_text = "\n".join(p.text for p in document.paragraphs)
        finally:
            os.remove(tmp_path)
        return [{"page": 1, "text": clean_text(full_text)}]

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(file_storage.read())
        tmp_path = tmp.name
    try:
        pages = extract_pdf_text(Path(tmp_path))
    finally:
        os.remove(tmp_path)
    return pages

def _build_context_from_pages(pages, query, device_label):
    query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
    scored = [
        (page.get("page", i + 1), page.get("text", ""), _score_page(page.get("text", ""), query_words))
        for i, page in enumerate(pages)
        if page.get("text", "").strip()
    ]
    scored = [item for item in scored if item[2] > 0]
    scored.sort(key=lambda item: item[2], reverse=True)
    selected = []
    total_chars = 0
    for page_number, text, _score in scored:
        if len(selected) >= MAX_FILE_PAGES or total_chars >= MAX_FILE_CONTEXT_CHARS:
            break
        selected.append((page_number, text))
        total_chars += len(text)

    if not selected:
        return ""
    
    selected.sort(key=lambda item: item[0])
    parts = []
    for index, (page_number, text) in enumerate(selected, start=1):
        parts.append(
            f"SOURCE {index}\n"
            f"Device: {device_label}\n"
            f"Page: {page_number}\n"
            f"Section: Uploaded document\n"
            f"Manual evidence:\n"
            f"{text[:3000]}"
        )
    return "\n\n".join(parts)

@app.route("/api/ask_file", methods=["POST"])
def ask_file():
    query = (request.form.get("query") or "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    if len(query) > 1000:
        return jsonify({"error": "Query is too long."}), 400
    if "file" not in request.files:
        return jsonify({"error": "file is required"}), 400

    file_storage = request.files["file"]
    device_label = Path(file_storage.filename or "uploaded file").stem
    try:
        pages = _extract_text_from_upload(file_storage)
    except RuntimeError as error:
        return jsonify({"error": str(error)}), 400
    except Exception as error:
        print(f"[ERROR] /api/ask_file extraction failed: {error}")
        return jsonify({"error": "Couldn't read this file. Try a different PDF or TXT file."}), 400

    context = _build_context_from_pages(pages, query, device_label)
    if not context:
        message = "No readable text was found in the uploaded file."
        return jsonify({
            "answer": message,
            "speech_answer": message,
            "device": device_label,
            "retrieval_type": "not_found",
        })

    try:
        t0 = time.time()
        generated = generate_answer(query=query, context=context, device=device_label)
        print(f"[timing] /api/ask_file (LLM) took {time.time() - t0:.2f}s")
    except Exception as error:
        print(f"[ERROR] /api/ask_file generation failed: {error}")
        return jsonify({"error": "The assistant couldn't process this file right now."}), 502

    return jsonify({
        "answer": generated.get("display_answer", ""),
        "speech_answer": generated.get("speech_answer", ""),
        "device": device_label,
        "retrieval_type": "uploaded_file",
    })

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)