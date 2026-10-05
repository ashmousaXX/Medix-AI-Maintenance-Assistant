import re
from retrieval import retrieve
from llm import generate_answer, truncate_speech_text

_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|hiya|yo|good\s*(morning|afternoon|evening))\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_THANKS_RE = re.compile(
    r"^\s*(thanks|thank you|thx|much appreciated)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see you|see ya|take care)\s*[!.?]*\s*$",
    re.IGNORECASE,
)
_NO_MORE_RE = re.compile(
    r"^\s*(no|nope|nah|no\s*thanks|not\s*now|that'?s\s*all|nothing\s*else|"
    r"i'?m\s*(all\s*)?done|im\s*(all\s*)?done)\s*[!.?]*\s*$",
    re.IGNORECASE,
)

_SMALL_TALK_RESPONSES = {
    "greeting": (
        "Hi! I'm Medix. Tell me the device and the symptom you're "
        "seeing, and I'll pull up what the service manual says."
    ),
    "thanks": "You're welcome! Let me know if there's another fault you'd like to check.",
    "bye": "Take care! Come back anytime you run into another issue.",
    "closing": "Glad I could help! Feel free to come back anytime you have another issue.",
}

def _detect_small_talk(query):
    text = (query or "").strip()
    if not text:
        return None
    if _GREETING_RE.match(text):
        return "greeting"
    if _THANKS_RE.match(text):
        return "thanks"
    if _BYE_RE.match(text):
        return "bye"
    if _NO_MORE_RE.match(text):
        return "closing"
    return None

def format_exact_results(results):
    if not results.get("ids"):
        return ""

    context_parts = []
    for index, (document, metadata) in enumerate(
        zip(
            results["documents"],
            results["metadatas"],
        ),
        start=1,
    ):
        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata.get('device', '')}\n"
            f"Page: {metadata.get('page', '')}\n"
            f"Section: {metadata.get('section', '')}\n"
            f"Error code: {metadata.get('error_code', '')}\n"
            f"Manual evidence:\n"
            f"{document}"
        )
        context_parts.append(source)
    return "\n\n".join(context_parts)

def format_semantic_results(results):
    documents = results.get("documents")
    if not documents:
        return ""
    if not documents[0]:
        return ""
    documents = documents[0]
    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]
    distances = results.get(
        "distances",
        [[]]
    )[0]
    context_parts = []

    for index, (
        document,
        metadata,
        distance,
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        source = (
            f"SOURCE {index}\n"
            f"Device: {metadata.get('device', '')}\n"
            f"Page: {metadata.get('page', '')}\n"
            f"Section: {metadata.get('section', '')}\n"
            f"Distance: {distance:.4f}\n"
            f"Manual evidence:\n"
            f"{document}"
        )

        context_parts.append(source)
    return "\n\n".join(context_parts)

def build_rag_context(
    query,
    device_id=None,
    top_k=8,
):
    retrieval_output = retrieve(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )
    retrieval_type = retrieval_output["retrieval_type"]
    results = retrieval_output["results"]

    if retrieval_type == "exact_error":
        context = format_exact_results(results)

    elif retrieval_type == "not_found":
        context = ""

    else:
        context = format_semantic_results(results)
    return {
        "retrieval_type": retrieval_type,
        "detected_error_code": retrieval_output.get(
            "detected_error_code"
        ),
        "detected_device": retrieval_output.get(
            "detected_device"
        ),
        "context": context,
        "raw_results": results,
    }

def answer_query(
    query,
    top_k=8,
    device_id=None,
):
    small_talk = _detect_small_talk(query)
    if small_talk:
        message = _SMALL_TALK_RESPONSES[small_talk]
        retrieval_type = "closing" if small_talk in ("bye", "closing") else "small_talk"
        return {
            "answer": message,
            "speech_answer": message,
            "detected_device": None,
            "retrieval_type": retrieval_type,
            "context": "",
        }
    rag_result = build_rag_context(
        query=query,
        device_id=device_id,
        top_k=top_k,
    )
    context = rag_result["context"]

    if not context:
        fallback = (
            "No relevant information was found "
            "in the service manuals. Is there anything else I can "
            "help you with?"
        )
        return {
            "answer": fallback,
            "speech_answer": fallback,
            "detected_device": rag_result.get(
                "detected_device"
            ),
            "retrieval_type": rag_result["retrieval_type"],
            "context": "",
        }
    generated = generate_answer(
        query=query,
        context=context,
        device=rag_result.get(
            "detected_device"
        ),
    )
    display_answer = (
        generated.get("display_answer", "")
        + "\n\n*Is there anything else I can help you with?*"
    )
    speech_answer = truncate_speech_text(
        generated.get("speech_answer", "").rstrip(". ")
        + ". Anything else I can help with?",
        limit=200,
    )
    return {
        "answer": display_answer,
        "speech_answer": speech_answer,
        "detected_device": rag_result.get(
            "detected_device"
        ),
        "retrieval_type": rag_result[
            "retrieval_type"
        ],
        "context": context,
    }

def test_full_rag():
    query = (
        "The ventilator has a gas supply problem"
    )
    result = answer_query(
        query=query,
        top_k=8,
    )
    print("=" * 70)
    print(
        "DEVICE:",
        result["detected_device"]
    )
    print(
        "TYPE:",
        result["retrieval_type"]
    )
    print()
    print(
        result["answer"]
    )
    print("=" * 70)

if __name__ == "__main__":
    test_full_rag()