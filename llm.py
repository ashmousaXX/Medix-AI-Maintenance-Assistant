import json
import re
from dotenv import load_dotenv
from groq_client import chat, ANSWER_MODELS, LLMUnavailable
from prompts import SYSTEM_PROMPT, build_user_prompt

load_dotenv(override=True)
MAX_COMPLETION_TOKENS = 1200
MODELS_TO_TRY = list(ANSWER_MODELS)
UNVERIFIED_NOTE = (
    "\n\n*Note: part of this answer could not be fully "
    "verified against the retrieved manual pages and may "
    "need manual double-checking.*"
)

def clean_speech_text(text):
    """
    Strip/replace symbols that Orpheus TTS would read aloud
    literally (e.g. "equals", "dash", "open parenthesis").
    """
    text = text.replace("=", " means ")
    text = re.sub(r"[–—-]", ", ", text)      
    text = re.sub(r"[()]", "", text)          
    text = re.sub(r"[*_#|/\\]", "", text)     
    text = re.sub(r"\s+", " ", text).strip()
    return text

def truncate_speech_text(text, limit=180):

    if len(text) <= limit:
        return text
    truncated = text[:limit]
    last_period = truncated.rfind(".")
    if last_period > 40:
        return truncated[: last_period + 1]
    weak_endings = {
        "a", "an", "the", "to", "for", "of", "in", "on", "at", "by",
        "with", "and", "or", "but", "is", "are", "as", "from", "that",
        "this", "it", "its", "if",
    }
    words = truncated.rstrip(" ,").split(" ")
    while words and words[-1].lower().strip(",.") in weak_endings:
        words.pop()
    if not words:
        return truncated.rsplit(" ", 1)[0].rstrip(",") + "."
    return " ".join(words).rstrip(",") + "."

def call_groq_with_retry(messages):
    try:
        raw, used_model = chat(
            messages,
            MODELS_TO_TRY,
            max_tokens=MAX_COMPLETION_TOKENS,
            temperature=0.0,
            label="answer",
        )
    except LLMUnavailable as error:
        raise RuntimeError(f"Groq call failed: {error}")
    return raw

def _strip_fences(raw):
    """Remove <think> blocks and accidental ```json fences."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()
    return raw


MAX_CONTEXT_CHARS = 6000
def _truncate_context(context, limit=MAX_CONTEXT_CHARS):
    if len(context) <= limit:
        return context
    truncated = context[:limit]
    last_source = truncated.rfind("\n\nSOURCE ")
    if last_source > 0:
        truncated = truncated[:last_source]
    return truncated + (
        "\n\n[Additional matching sources were omitted to stay within "
        "the model's request size limit.]"
    )

def _context_pages(context):
    header_pages = set(int(n) for n in re.findall(r"^Page:\s*(\d+)", context, re.MULTILINE))
    body_pages = set(int(n) for n in re.findall(r"[Pp]age\s*:?\s*(\d+)", context))
    return header_pages | body_pages

def _answer_pages(display_answer):
    """Every page number the model's answer claims to cite."""
    return set(int(n) for n in re.findall(r"[Pp]age\s*:?\s*(\d+)", display_answer))

def _find_unsupported_pages(display_answer, context):
    """
    Page numbers cited in the answer that do not appear anywhere in
    the retrieved context -- a strong signal the model pulled a
    detail from outside knowledge instead of the provided evidence,
    even when that detail happens to be factually accurate.
    """
    return _answer_pages(display_answer) - _context_pages(context)

def _flag_if_unsupported(display_answer, context):
    unsupported = _find_unsupported_pages(display_answer, context)
    if unsupported:
        print(
            f"[WARN] Non-JSON answer cited page(s) not in context: "
            f"{sorted(unsupported)}"
        )
        return display_answer + UNVERIFIED_NOTE
    return display_answer

FALLBACK_TEXT = (
    "The assistant did not return a response for this query. "
    "Please try rephrasing the question or asking again."
)

def generate_answer(query,context,device=None,):
    context = _truncate_context(context)
    system_prompt = SYSTEM_PROMPT
    user_prompt = build_user_prompt(
        query=query,
        device=device,
        context=context,
    )
    try:
        raw = call_groq_with_retry(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
    except RuntimeError as error:
        print(f"[DEBUG] generate_answer giving up: {error}")
        return {
            "display_answer": FALLBACK_TEXT,
            "speech_answer": FALLBACK_TEXT,
            "llm_failed": True,
        }

    if not raw:
        return {
            "display_answer": FALLBACK_TEXT,
            "speech_answer": FALLBACK_TEXT,
            "llm_failed": True,
        }
    raw = _strip_fences(raw)

    try:
        parsed = json.loads(raw, strict=False)
        display_answer = str(parsed.get("display_answer", "")).strip()
        speech_answer = str(parsed.get("speech_answer", "")).strip()

        if not display_answer:
            display_answer = raw
        unsupported = _find_unsupported_pages(display_answer, context)
        if unsupported:
            print(f"[WARN] Answer cited page(s) not in context: {sorted(unsupported)} - retrying once")
            corrective_prompt = (
                user_prompt
                + "\n\nYour previous answer cited page number(s) "
                + ", ".join(str(p) for p in sorted(unsupported))
                + ", which do NOT appear in the SERVICE MANUAL EVIDENCE above. "
                + "This means you used information from outside the provided "
                + "evidence. Rewrite your answer using ONLY the pages that "
                + "actually appear above: "
                + ", ".join(str(p) for p in sorted(_context_pages(context)))
                + ". Remove any claim tied to a page not in that list."
            )
            try:
                retry_raw = _strip_fences(
                    call_groq_with_retry(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": corrective_prompt},
                        ]
                    )
                )
                retry_parsed = json.loads(retry_raw, strict=False)
                retry_display = str(retry_parsed.get("display_answer", "")).strip()
                retry_speech = str(retry_parsed.get("speech_answer", "")).strip()
                if retry_display and not _find_unsupported_pages(retry_display, context):
                    display_answer = retry_display
                    speech_answer = retry_speech or retry_display
                else:
                    print("[WARN] Retry still cites unsupported page(s) - flagging answer")
                    display_answer += UNVERIFIED_NOTE
            except Exception as error:
                print(f"[WARN] Corrective retry failed: {error} - flagging original answer")
                display_answer += UNVERIFIED_NOTE
        if not speech_answer:
            speech_answer = display_answer
        speech_answer = clean_speech_text(speech_answer)
        speech_answer = truncate_speech_text(speech_answer)
        return {
            "display_answer": display_answer,
            "speech_answer": speech_answer,
        }

    except (json.JSONDecodeError, AttributeError):
        display_match = re.search(
            r'"display_answer"\s*:\s*"(.*?)"\s*,\s*"speech_answer"',
            raw, re.DOTALL,
        )
        speech_match = re.search(
            r'"speech_answer"\s*:\s*"?(.*?)"?\s*\}?\s*$',
            raw, re.DOTALL,
        )
        if display_match:
            display_answer = display_match.group(1).replace('\\"', '"').replace("\\n", "\n")
            display_answer = _flag_if_unsupported(display_answer, context)
            speech_answer = (
                speech_match.group(1).strip().rstrip('"')
                if speech_match else display_answer
            )
            speech_answer = clean_speech_text(speech_answer)
            speech_answer = truncate_speech_text(speech_answer)
            return {
                "display_answer": display_answer,
                "speech_answer": speech_answer,
            }
        display_answer = raw
        speech_source = raw
        m = re.search(r"(?im)^[\s*#]*speech_answer[\s*#]*:\s*(.+)$", raw)
        if m:
            speech_source = m.group(1).strip()
            display_answer = (raw[:m.start()] + raw[m.end():]).strip()
        display_answer = _flag_if_unsupported(display_answer, context)
        speech_answer = truncate_speech_text(clean_speech_text(speech_source))
        return {
            "display_answer": display_answer,
            "speech_answer": speech_answer,
        }