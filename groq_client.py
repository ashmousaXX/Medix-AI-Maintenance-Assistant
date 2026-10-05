import os
import time
from dotenv import load_dotenv
from groq import Groq

load_dotenv(override=True)

client = Groq( api_key=os.getenv("GROQ_API_KEY"),max_retries=0,timeout=30.0,)

ANSWER_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
HYDE_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]

_forced_model = os.getenv("Medix_MODEL")
if _forced_model:
    ANSWER_MODELS = [_forced_model]
    print(f"[LLM] Medix_MODEL set: answers use ONLY {_forced_model}")
MAX_SHORT_WAIT = 15         
DEFAULT_COOLDOWN_SECONDS = 60

_cooldown = {}
_available = None           
class LLMUnavailable(Exception):
    """Raised when no model could produce an answer."""
def _available_models():
    global _available
    if _available is None:
        try:
            _available = {m.id for m in client.models.list().data}
            print(f"[LLM] models available on this account: {sorted(_available)}")
        except Exception as error:
            print(f"[LLM] could not list models ({error}); not filtering")
            _available = set()
    return _available

def _retry_after_seconds(error, default=DEFAULT_COOLDOWN_SECONDS):
    try:
        header = error.response.headers.get("retry-after")
        if header:
            return max(1.0, float(header))
    except Exception:
        pass
    return float(default)

def _try_models(messages, models, max_tokens, temperature, label):
    last_error = None
    now = time.time()
    all_cooling = True
    for model in models:
        if now < _cooldown.get(model, 0):
            continue
        all_cooling = False
        kwargs = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
        if model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = "low"  
        try:
            response = client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            text = (choice.message.content or "").strip()
            if choice.finish_reason == "length":
                print(
                    f"[LLM] {label} {model}: reply truncated "
                    f"(finish_reason=length) - trying next model"
                )
                last_error = LLMUnavailable(f"{model} reply was truncated")
                continue
            if not text:
                print(
                    f"[LLM] {label} {model}: empty response "
                    f"(finish_reason={choice.finish_reason}) - trying next model"
                )
                last_error = LLMUnavailable(f"{model} returned an empty response")
                continue
            if model != models[0]:
                print(f"[LLM] {label}: answered by fallback model {model}")
            return text, model
        except Exception as error:  
            name = type(error).__name__
            if name == "RateLimitError":
                wait = _retry_after_seconds(error)
                _cooldown[model] = time.time() + wait
                print(
                    f"[LLM] {label} {model} RATE LIMITED "
                    f"(cooldown {wait:.0f}s): {error}"
                )
            elif name == "NotFoundError":
                _cooldown[model] = time.time() + 3600
                print(
                    f"[LLM] {label} {model} NOT AVAILABLE on this account "
                    f"- skipping it. ({error})"
                )
            else:
                print(f"[LLM] {label} {model} failed ({name}): {error}")
            last_error = error
    if last_error is None and all_cooling:
        soonest = min(
            max(0.0, _cooldown.get(m, 0) - now) for m in models
        )
        raise LLMUnavailable(
            f"all models are rate-limited, try again in about {soonest:.0f}s"
        )
    raise LLMUnavailable(
        "all models failed; last error: "
        f"{type(last_error).__name__ if last_error else 'none'}"
    )

def chat(
    messages,
    models,
    max_tokens=1800,
    temperature=0.2,
    label="",
):

    available = _available_models()
    if available:
        usable = [m for m in models if m in available]
        if not usable:
            raise LLMUnavailable(
                "none of the configured models exist on this account; "
                f"available: {sorted(available)}"
            )
    else:
        usable = list(models)

    for attempt in range(2):
        try:
            return _try_models(messages, usable, max_tokens, temperature, label)
        except LLMUnavailable as error:
            if attempt == 1:
                raise
            now = time.time()
            waits = [
                _cooldown[m] - now
                for m in usable
                if 0 < _cooldown.get(m, 0) - now <= MAX_SHORT_WAIT
            ]
            if not waits:
                raise
            pause = min(waits) + 0.5
            print(f"[LLM] {label}: waiting {pause:.1f}s for the rate limit, then retrying once")
            time.sleep(pause)