import json
import re
from config import PROCESSED_DIR

CHUNKS_FILE = PROCESSED_DIR / "maintai_chunks.json"
SC6002XL_PATTERN = re.compile(
    r"Symptom or condition:\s*(?P<symptom>.*?)\s*"
    r"Possible cause:\s*(?P<cause>.*?)\s*"
    r"Troubleshooting and remedial action:\s*(?P<action>.*)",
    re.IGNORECASE | re.DOTALL,
)

def normalize_text(text):
    match = SC6002XL_PATTERN.search(text)
    if not match:
        return text, False
    symptom = match.group("symptom").strip()
    cause = match.group("cause").strip()
    action = match.group("action").strip()
    normalized = (
        f"Malfunction: {symptom} "
        f"Possible cause: {cause} "
        f"Action: {action}"
    )
    prefix = text[: match.start()].strip()
    if prefix:
        normalized = f"{prefix} {normalized}"
    return normalized, True

def main():
    with open(CHUNKS_FILE, "r", encoding="utf-8") as file:
        chunks = json.load(file)
    changed = 0
    for chunk in chunks:
        text = chunk.get("text", "")
        new_text, was_changed = normalize_text(text)
        if was_changed:
            chunk["text"] = new_text
            changed += 1
    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)
    print(f"Normalized {changed} chunk(s) out of {len(chunks)} total.")
    print("Now rebuild the vector database:")
    print(
        '  python -c "from retrieval import create_vector_database; '
        'create_vector_database()"'
    )
if __name__ == "__main__":
    main()