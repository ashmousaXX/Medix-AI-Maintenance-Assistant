import re
import time
from retrieval import retrieve

LLM_CALL_DELAY_SECONDS = 15
LLM_FAILURE_PREFIX = "The assistant did not return a response"
STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in",
    "on", "for", "and", "or", "with", "if", "has", "have", "this",
    "that", "from", "by", "as", "be", "it", "may", "can",
}

TEST_CASES = [
    
    {
        "name": "Servo - Transducer malfunction",
        "query": "The inspiratory flow transducer is defective",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": ["transducer"],
    },
    {
        "name": "Servo - Gas supply",
        "query": "The ventilator has a gas supply problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": ["gas supply"],
    },
    {
        "name": "Servo - Power supply",
        "query": "The ventilator has a power supply problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": ["voltage supply", "power supply"],
    },
    {
        "name": "Servo - Pressure problem",
        "query": "The ventilator has a pressure problem",
        "expected_device": "servo_ventilator",
        "expected_type": "semantic",
        "expected_answer_terms": ["pressure"],
    },

    {
        "name": "Philips - Blank screen",
        "query": "The patient monitor screen is blank",
        "expected_device": "philips_v24_v25_agilent_m1205_monitor_service_manual",
        "expected_type": "semantic",
        "expected_answer_terms": ["blank", "screen"],
    },
    {
        "name": "Philips - Power problem",
        "query": "The monitor has a power supply problem",
        "expected_device": "philips_v24_v25_agilent_m1205_monitor_service_manual",
        "expected_type": "semantic",
        "expected_answer_terms": ["power supply", "fuse"],
    },

    {
        "name": "SC6002XL - Display malfunction",
        "query": "Parts of the display are missing or the colors look wrong",
        "expected_device": "sc6002xl",
        "expected_type": "semantic",
        "expected_answer_terms": ["Front Panel PC Board"],
    },

    {
        "name": "Negative - Unrelated problem",
        "query": "How do I repair a home coffee machine?",
        "expected_device": None,
        "expected_type": "not_found",
        "expected_answer_terms": [],
    },
]

def normalize(text):
    """Normalize text for loose comparison."""
    text = str(text or "").lower()
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def compact(text):
    """Stronger normalization used when comparing evidence with answers."""
    text = normalize(text)
    text = text.replace("**", "").replace("__", "")
    return text.strip()

def content_words(text):
    """Extract meaningful words for conservative overlap checks."""
    words = re.findall(r"[a-z0-9]+", normalize(text))
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}

def word_overlap(text_a, text_b):
    """Ratio of meaningful words from text_a that also occur in text_b."""
    words_a = content_words(text_a)
    words_b = content_words(text_b)
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)

def contains_expected_terms(text, expected_terms):
    """
    Check whether important expected terms appear in the answer.
    Multi-word terms are checked as sets of words, since manuals may
    use slightly different word ordering.
    """
    text = normalize(text)
    if not expected_terms:
        return True, []
    missing = []
    for term in expected_terms:
        words = normalize(term).split()
        found = all(w in text for w in words) if len(words) > 1 else normalize(term) in text
        if not found:
            missing.append(term)
    return len(missing) == 0, missing

# SOURCE parsing
def parse_sources(context):
    """
    Parse the retrieved LLM context into {1: "source 1 text", 2: ...}
    so the evaluator can independently verify every SOURCE citation
    used by the LLM.
    """
    sources = {}
    if not context:
        return sources

    pattern = re.compile(r"(?im)^\s*SOURCE\s+(\d+)\s*$")
    matches = list(pattern.finditer(context))
    for i, match in enumerate(matches):
        number = int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(context)
        sources[number] = context[start:end].strip()
    return sources

def extract_source_numbers(text):
    """Extract all 'SOURCE N' references from the generated answer."""
    if not text:
        return []
    matches = re.findall(r"\bSOURCE\s+(\d+)\b", str(text), flags=re.IGNORECASE)
    return sorted({int(n) for n in matches})

def validate_source_citations(answer, context):
    """Detect hallucinated SOURCE numbers (cited but not in context)."""
    available = parse_sources(context)
    cited = extract_source_numbers(answer)
    invalid = [n for n in cited if n not in available]
    return {
        "ok": len(invalid) == 0,
        "available_sources": sorted(available.keys()),
        "cited_sources": cited,
        "invalid_sources": invalid,
    }

def extract_malfunction_action_pairs(source_text):
    """
    Extract explicit "Malfunction: ... Action: ..." pairs from one
    SOURCE. Used to check whether an action actually belongs to the
    malfunction being discussed. Intentionally conservative — if the
    source structure is ambiguous, it does not invent an association.
    """
    pairs = []
    if not source_text:
        return pairs
    chunks = re.split(r"(?i)\bMalfunction\s*:", source_text.strip())

    for chunk in chunks[1:]:
        chunk = chunk.strip()
        action_match = re.search(r"(?i)\bAction\s*:", chunk)

        if not action_match:
            pairs.append({"malfunction": chunk, "action": None})
            continue
        malfunction_text = chunk[:action_match.start()].strip()
        action_text = chunk[action_match.end():].strip()
        next_malfunction = re.search(r"(?i)\bMalfunction\s*:", action_text)
        if next_malfunction:
            action_text = action_text[:next_malfunction.start()].strip()
        pairs.append({"malfunction": malfunction_text, "action": action_text or None})
    return pairs

def action_supported_by_source(action_text, source_text):
    """
    Check whether a generated action is actually supported by the
    cited source, via direct phrase containment or strong word overlap.
    Avoids rejecting harmless paraphrases while catching invented ones.
    """
    if not action_text:
        return False
    action = compact(action_text)
    source = compact(source_text)
    if not action:
        return False
    action_clean = re.sub(r"\(\s*source\s+\d+[^)]*\)", "", action, flags=re.IGNORECASE).strip()
    if action_clean and action_clean in source:
        return True

    action_words = content_words(action_clean)
    source_words = content_words(source)
    if not action_words:
        return False
    return len(action_words & source_words) / len(action_words) >= 0.65

def extract_answer_sections(answer):
    """
    Extract Fixora's structured sections (Matching fault / Manual
    action / Related faults) from the generated answer, tolerating
    markdown formatting.
    """
    text = str(answer or "")
    text = re.sub(r"(?im)^\s*\*?\s*is there anything else i can help you with\??\s*\*?\s*$","", text,)
    text = re.sub(r"(?i)\*\*matching fault:\*\*", "\nMATCHING_FAULT:\n", text)
    text = re.sub(r"(?i)\*\*manual action:\*\*", "\nMANUAL_ACTION:\n", text)
    text = re.sub(r"(?i)\*\*related faults in the manual:\*\*", "\nRELATED_FAULTS:\n", text)
    result = {"matching_fault": "", "manual_action": "", "related_faults": ""}
    match = re.search(
        r"(?is)MATCHING_FAULT:\s*(.*?)(?=\nMANUAL_ACTION:|\nRELATED_FAULTS:|$)", text
    )
    if match:
        result["matching_fault"] = match.group(1).strip()
    match = re.search(r"(?is)MANUAL_ACTION:\s*(.*?)(?=\nRELATED_FAULTS:|$)", text)
    if match:
        result["manual_action"] = match.group(1).strip()
    match = re.search(r"(?is)RELATED_FAULTS:\s*(.*)$", text)
    if match:
        result["related_faults"] = match.group(1).strip()
    return result

def _split_bulleted_segments(text):
    """
    Split an answer section into its bullet-point items when it lists
    multiple faults/actions (Rule 6c format). Falls back to treating
    the whole text as one segment when there's no bullet structure.
    """
    if not text:
        return []
    parts = re.split(r"\n\s*[-*•]\s*", "\n" + text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [text]

def _validate_multi_fault_answer(fault_bullets, action_bullets, sources):
    """
    Validate a multi-fault answer bullet by bullet: each fault bullet
    is checked only against the ONE source it cites, not against the
    combined block — checking the whole block dilutes word overlap
    across unrelated malfunctions and spuriously fails correct
    multi-fault answers.
    """
    problems = []
    any_checked = False
    for fault_bullet in fault_bullets:
        fault_sources = extract_source_numbers(fault_bullet)
        if len(fault_sources) != 1:
            continue  

        source_number = fault_sources[0]
        source_text = sources.get(source_number)
        if source_text is None:
            continue

        any_checked = True
        fault_clean = re.sub(
            r"\(\s*source\s+\d+[^)]*\)", "", fault_bullet, flags=re.IGNORECASE
        )

        pairs = extract_malfunction_action_pairs(source_text)
        fault_supported = any(
            word_overlap(fault_clean, pair["malfunction"]) >= 0.40
            for pair in pairs
        ) or word_overlap(fault_clean, source_text) >= 0.40

        if not fault_supported:
            problems.append(
                f"A listed fault citing SOURCE {source_number} does not "
                f"match that source's content."
            )
            continue

        matching_action_bullet = next(
            (b for b in action_bullets if source_number in extract_source_numbers(b)),
            None,
        )
        if matching_action_bullet:
            if not action_supported_by_source(matching_action_bullet, source_text):
                problems.append(
                    f"The action listed for SOURCE {source_number} is not "
                    f"grounded in that source."
                )

    if not any_checked:
        return None  
    return {"ok": len(problems) == 0, "problems": problems}

NO_MATCH_PHRASES = [
    "no malfunction in the provided manual evidence",
    "no malfunction in the provided evidence",
    "no malfunction in the retrieved",
    "does not directly describe",
    "none of the retrieved sources",
    "none mention",
    "no evidence directly matches",
    "nothing in the evidence directly matches",
    "does not match closely enough",
    "does not contain a single malfunction",
    "does not contain a malfunction",
    "too general",
]

NO_MATCH_REGEXES = [
    r"\bno (specific |single |direct |matching )?(\w+ )?(malfunction|fault|entry|entries|match)",
    r"\b(does|do) not (directly )?(contain|describe|match|specify|provide)",
    r"\b(is|are) not (provided|described|documented|listed|specified)",
    r"\b(too|is) (general|broad|vague)",
    r"\bnot (sufficiently|closely) ",
]

def declares_no_match(text):
    t = compact(text)
    return (
        any(p in t for p in NO_MATCH_PHRASES)
        or any(re.search(p, t) for p in NO_MATCH_REGEXES)
    )
def validate_listed_entries(answer, sources):
    problems = []
    for bullet in re.findall(r"(?m)^\s*[-*•]\s*(.+)$", str(answer or "")):
        nums = extract_source_numbers(bullet)
        if len(nums) != 1:
            continue
        src = sources.get(nums[0], "")
        if re.search(r"(?i)\bmalfunction\s*:", bullet) and not re.search(r"(?i)\bmalfunction\s*:", src):
            problems.append(f"SOURCE {nums[0]} is not a Malfunction/Action entry but was presented as one.")
        clean = re.sub(r"\(\s*source\s+\d+[^)]*\)", "", bullet, flags=re.I)
        for sent in re.split(r"(?<=[.;])\s+", clean):
            sent = sent.strip()
            core = sent.strip("., ")
            if core.startswith("...") or (core.startswith("(") and core.endswith(")")):
                continue
            if len(content_words(sent)) >= 5 and word_overlap(sent, src) < 0.4:
                problems.append(f"Possibly invented sentence (SOURCE {nums[0]}): {sent.strip()[:90]}")
    return problems

def validate_matching_fault_action(query, answer, context):
    """
    Detect three grounding problems:
    1. Matching fault not grounded in the retrieved context.
    2. Action not present in the cited source.
    3. Action that actually belongs to a different malfunction.
    """
    sources = parse_sources(context)
    sections = extract_answer_sections(answer)
    matching_fault = sections["matching_fault"]
    manual_action = sections["manual_action"]
    label_re = re.compile(r"(?i)\b(?:malfunction|action)\s*:\s*")
    matching_fault = label_re.sub("", matching_fault)
    manual_action = label_re.sub("", manual_action)
    problems = []

    if declares_no_match(matching_fault):
        return {"ok": True, "problems": [], "matching_fault_source": None}

    fault_sources_mentioned = set(extract_source_numbers(matching_fault))
    if len(fault_sources_mentioned) > 1:
        fault_bullets = _split_bulleted_segments(matching_fault)
        action_bullets = _split_bulleted_segments(manual_action)
        multi_result = _validate_multi_fault_answer(fault_bullets, action_bullets, sources)
        if multi_result is not None:
            return {
                "ok": multi_result["ok"],
                "problems": multi_result["problems"],
                "matching_fault_source": None,
            }

    if not matching_fault:
        return {"ok": False, "problems": ["No explicit Matching fault section found."]}
    fault_clean = re.sub(r"\(\s*source\s+\d+[^)]*\)", "", matching_fault, flags=re.IGNORECASE)
    action_sources = extract_source_numbers(manual_action)
    matching_fault_supported = False
    matching_fault_source = None

    for source_number, source_text in sources.items():
        for pair in extract_malfunction_action_pairs(source_text):
            if (
                word_overlap(fault_clean, pair["malfunction"]) >= 0.50
                or word_overlap(pair["malfunction"], fault_clean) >= 0.50
            ):
                matching_fault_supported = True
                matching_fault_source = source_number
                break
        if matching_fault_supported:
            break

    if not matching_fault_supported:
        all_context = " ".join(sources.values())
        fault_words = content_words(fault_clean)
        context_words = content_words(all_context)
        if fault_words and len(fault_words & context_words) / len(fault_words) >= 0.60:
            matching_fault_supported = True

    if not matching_fault_supported:
        problems.append("Matching fault is not sufficiently grounded in the retrieved manual evidence.")

    if manual_action:
        missing_action_phrases = [
            "detailed corrective action is not provided",
            "does not provide a detailed corrective action",
            "no detailed corrective action",
            "corrective action is not provided",
            "action is not provided",
        ]
        declares_missing = any(p in compact(manual_action) for p in missing_action_phrases)

        if not declares_missing:
            if not action_sources and matching_fault_source is not None:
                action_sources = [matching_fault_source]
            action_supported = False
            wrong_malfunction_action = False

            for source_number in action_sources:
                source_text = sources.get(source_number)
                if source_text is None:
                    continue
                pairs = extract_malfunction_action_pairs(source_text)

                if not pairs:
                    if action_supported_by_source(manual_action, source_text):
                        action_supported = True
                        break
                    continue

                for pair in pairs:
                    if (
                        word_overlap(fault_clean, pair["malfunction"]) >= 0.50
                        and pair["action"]
                        and action_supported_by_source(manual_action, pair["action"])
                    ):
                        action_supported = True
                        break
                if action_supported:
                    break

                for pair in pairs:
                    if pair["action"] and action_supported_by_source(manual_action, pair["action"]):
                        if word_overlap(fault_clean, pair["malfunction"]) < 0.50:
                            wrong_malfunction_action = True
                            problems.append(
                                "The Manual action appears to belong to a "
                                "different malfunction than the reported "
                                "Matching fault."
                            )
                            break
                if wrong_malfunction_action:
                    break

            if not action_supported and not wrong_malfunction_action:
                all_context = " ".join(sources.values())
                if word_overlap(manual_action, all_context) >= 0.60:
                    action_supported = True

            if not action_supported and not wrong_malfunction_action:
                problems.append(
                    "Manual action is not sufficiently grounded in the "
                    "retrieved evidence for the matching fault."
                )
    return {"ok": len(problems) == 0, "problems": problems, "matching_fault_source": matching_fault_source}

def validate_answer_grounding(query, answer, context):
    """Run all grounding checks: SOURCE hallucination + fault/action grounding."""
    source_check = validate_source_citations(answer, context)
    fault_action_check = validate_matching_fault_action(query, answer, context)

    problems = []
    if not source_check["ok"]:
        cited = ", ".join(f"SOURCE {n}" for n in source_check["invalid_sources"])
        problems.append(f"Hallucinated SOURCE number(s): {cited}")
    problems.extend(fault_action_check["problems"])
    problems.extend(validate_listed_entries(answer, parse_sources(context)))
    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "source_check": source_check,
        "fault_action_check": fault_action_check,
    }

def evaluate_retrieval():
    total = len(TEST_CASES)
    device_correct = 0
    type_correct = 0
    error_code_correct = 0
    error_tests = sum(1 for c in TEST_CASES if c.get("expected_error_code"))
    results = []
    print()
    print("=" * 80)
    print("Medix RETRIEVAL EVALUATION")
    print("=" * 80)

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{index}/{total}] {case['name']}")
        print(f"Query: {case['query']}")
        result = retrieve(query=case["query"], top_k=8)
        actual_type = result.get("retrieval_type")
        actual_device = result.get("detected_device")
        actual_error = result.get("detected_error_code")
        type_ok = actual_type == case["expected_type"]
        device_ok = actual_device == case["expected_device"]
        error_ok = True

        if case.get("expected_error_code"):
            error_ok = str(actual_error) == str(case["expected_error_code"])
            if error_ok:
                error_code_correct += 1
        type_correct += type_ok
        device_correct += device_ok
        print(f"Expected type : {case['expected_type']}")
        print(f"Actual type   : {actual_type}")
        print(f"Expected device: {case['expected_device']}")
        print(f"Actual device  : {actual_device}")

        if case.get("expected_error_code"):
            print(f"Expected error: {case['expected_error_code']}")
            print(f"Actual error  : {actual_error}")
            print(f"Error code: {'PASS' if error_ok else 'FAIL'}")
            print(f"Type: {'PASS' if type_ok else 'FAIL'}")
            print(f"Device: {'PASS' if device_ok else 'FAIL'}")
        results.append({"name": case["name"], "query": case["query"], "result": result})

    print()
    print("=" * 80)
    print("RETRIEVAL SUMMARY")
    print("=" * 80)
    print(f"Total tests: {total}")
    print(f"Device accuracy: {device_correct / total * 100:.1f}%")
    print(f"Retrieval type accuracy: {type_correct / total * 100:.1f}%")
    if error_tests:
        print(f"Error-code accuracy: {error_code_correct / error_tests * 100:.1f}%")
    print("=" * 80)
    return results

def evaluate_answers():
    print()
    print("=" * 80)
    print("FIXORA ANSWER EVALUATION")
    print("=" * 80)

    answer_correct = 0
    answer_total = 0
    grounding_correct = 0
    source_correct = 0
    action_grounding_correct = 0
    skipped = 0

    for index, case in enumerate(TEST_CASES, start=1):
        print(f"\n[{index}/{len(TEST_CASES)}] {case['name']}")

        if case["expected_type"] == "not_found":
            result = retrieve(query=case["query"], top_k=5)
            if result.get("retrieval_type") == "not_found":
                print("Retrieval returned NOT_FOUND.")
                print("Negative test: PASS")
                answer_correct += 1
                answer_total += 1
                grounding_correct += 1
                source_correct += 1
                action_grounding_correct += 1
                continue

        from rag import answer_query
        rag_result = answer_query(query=case["query"], top_k=8)
        time.sleep(LLM_CALL_DELAY_SECONDS)
        context = rag_result.get("context", "")
        answer = rag_result.get("answer", "")
        if answer.startswith(LLM_FAILURE_PREFIX):
            print("LLM call failed (rate limit / empty response) — SKIPPED, not counted.")
            skipped += 1
            continue
        print("\nContext sent to LLM:")
        print("\nContext sent to LLM:")

        import re as _re
        headers = _re.findall(
            r"SOURCE (\d+)\nDevice: (.*?)\nPage: (.*?)\nSection: (.*?)\n",
            context,
        )
        print("All retrieved sources (page/section):")
        for source_num, device, page, section in headers:
            print(f"  SOURCE {source_num}: Page {page}, Section: {section}")
        print(context[:5000])
        print("-" * 80)

        ok, missing = contains_expected_terms(answer, case.get("expected_answer_terms", []))
        answer_total += 1
        answer_correct += ok
        print(f"Expected-term check: {'PASS' if ok else 'FAIL'}")
        if missing:
            print("Missing expected terms:", ", ".join(missing))

        grounding = validate_answer_grounding(query=case["query"], answer=answer, context=context)
        source_check = grounding["source_check"]
        fault_action_check = grounding["fault_action_check"]
        source_correct += source_check["ok"]
        action_grounding_correct += fault_action_check["ok"]
        grounding_correct += grounding["ok"]
        print(f"\nSOURCE grounding: {'PASS' if source_check['ok'] else 'FAIL'}")
        print("Available SOURCES:", source_check["available_sources"] or "NONE")
        print("Cited SOURCES:", source_check["cited_sources"] or "NONE")

        if source_check["invalid_sources"]:
            hallucinated = ", ".join(f"SOURCE {n}" for n in source_check["invalid_sources"])
            print("HALLUCINATED SOURCES:", hallucinated)

        print(f"\nMalfunction/action grounding: {'PASS' if fault_action_check['ok'] else 'FAIL'}")
        for problem in fault_action_check["problems"]:
            print(f"  - {problem}")

        print(f"\nOverall hallucination check: {'PASS' if grounding['ok'] else 'FAIL'}")
        if grounding["problems"]:
            print("\nGrounding problems:")
            for problem in grounding["problems"]:
                print(f"  - {problem}")
                print("\nAnswer:")
                print(answer)
                print("\n" + "=" * 80)
                print()
                print("=" * 80)
                print("ANSWER SUMMARY")
                print("=" * 80)
                print(f"Evaluated answers: {answer_total}")
    if skipped:
        print(f"Skipped (LLM call failed): {skipped}")

    if answer_total:
        print(f"Answer fact-match accuracy: {answer_correct / answer_total * 100:.1f}%")
        print(f"SOURCE grounding accuracy: {source_correct / answer_total * 100:.1f}%")
        print(f"Malfunction/action grounding accuracy: {action_grounding_correct / answer_total * 100:.1f}%")
        print(f"Overall hallucination-free answers: {grounding_correct / answer_total * 100:.1f}%")
        print("=" * 80)

if __name__ == "__main__":
    evaluate_retrieval()
    choice = input("\nRun LLM answer evaluation? (y/n): ").strip().lower()
    if choice == "y":
        evaluate_answers()
    else:
        print("LLM answer evaluation skipped.")