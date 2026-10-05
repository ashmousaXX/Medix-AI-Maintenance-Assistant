import json
import re
from pathlib import Path
import pymupdf
import pdfplumber
import pytesseract
from PIL import Image
from config import (MANUALS, MANUALS_DIR, PROCESSED_DIR,)
import os
import shutil

tesseract_path = shutil.which("tesseract")

if not tesseract_path:
    windows_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(windows_default):
        tesseract_path = windows_default

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    print(
        "WARNING: Tesseract OCR binary was not found. "
        "Scanned/image-only PDFs will fail to extract any text."
    )

def clean_text(text):
    if not text:
        return ""
    text = text.replace("\u00ad", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_cell(value):
    if value is None:
        return ""
    value = str(value).replace("\n", " ")
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\bpage(\d)", r"page \1", value, flags=re.IGNORECASE,)
    return value.strip()

def fix_medical_terms(text):
    if not text:
        return ""
    text = re.sub(r"\bSpO\s+2\b", "SpO2", text, flags=re.IGNORECASE,)
    text = re.sub(r"\betCO\s+2\b", "etCO2", text, flags=re.IGNORECASE,)
    text = re.sub(r"\bCO\s+2\b", "CO2", text, flags=re.IGNORECASE,)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

ERROR_CODE_FALSE_POSITIVE_CONTEXT = {
    "page", "pg", "p", "chapter", "ch", "section", "sec",
    "table", "tbl", "figure", "fig", "step", "item", "note",
    "rev", "revision", "version", "ver", "model", "type",
    "part", "no", "number", "appendix", "volume", "vol",
    "manual", "chart", "diagram", "row", "column", "col",
}

def _is_false_positive_context(text, match_start):
    preceding = text[:match_start].strip().split()
    if not preceding:
        return False
    context_word = preceding[-1].lower().strip(".,:;()[]-")
    return context_word in ERROR_CODE_FALSE_POSITIVE_CONTEXT

def detect_error_code(text):
    patterns = [
        r"\berror\s+code\s*[:#]?\s*(\d{1,5})\b",
        r"\bfault\s+code\s*[:#]?\s*(\d{1,5})\b",
        r"\berror\s+(\d{1,5})\b",
        r"\bfault\s+(\d{1,5})\b",
        r"\bE[-\s]?(\d{1,5})\b",
        r"\bERR[-\s]?(\d{1,5})\b",
    ]
    for pattern in patterns:
        for match in re.finditer(
            pattern,
            text,
            flags=re.IGNORECASE,
        ):
            if _is_false_positive_context(text, match.start()):
                continue
            return match.group(1)
    return None

def detect_chunk_type(text):
    upper_text = text.upper()
    if any(
        term in upper_text
        for term in ["DANGER", "WARNING", "CAUTION", ]
    ):
        return "safety"
    if detect_error_code(text) is not None:
        return "error_code"
    return "text"

def extract_with_pymupdf(pdf_path):
    pages = []
    try:
        document = pymupdf.open(pdf_path)
        for page_number, page in enumerate(
            document,
            start=1,
        ):
            try:
                text = page.get_text(
                    "text"
                ).strip()
            except Exception as error:
                print(
                    f"    PyMuPDF page "
                    f"{page_number} error: {error}"
                )
                text = ""
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )
        document.close()

    except Exception as error:
        print(f" PyMuPDF failed: {error}")
        return []
    return pages

def extract_with_pdfplumber(pdf_path):
    pages = []
    try:
        with pdfplumber.open(
            pdf_path
        ) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1,):
                try:
                    text = (page.extract_text(x_tolerance=2, y_tolerance=2,) or "")
                except Exception as error:
                    print(
                        f"    pdfplumber page "
                        f"{page_number} error: {error}"
                    )
                    text = ""
                pages.append(
                    {
                        "page": page_number,
                        "text": text.strip(),
                    }
                )
    except Exception as error:
        print(f"  pdfplumber failed: {error}")
        return []
    return pages

# OCR
def extract_with_ocr(pdf_path):
    pages = []
    print("  Starting OCR fallback...")
    try:
        document = pymupdf.open(pdf_path)
        for page_number, page in enumerate(document, start=1,):
            try:
                pix = page.get_pixmap( matrix=pymupdf.Matrix(2.0,2.0,),alpha=False,)
                image = Image.frombytes(
                    "RGB",
                    ( pix.width, pix.height,),
                    pix.samples,
                )
                text = pytesseract.image_to_string(
                    image,
                    lang="eng",
                ).strip()
            except Exception as error:
                print(
                    f" OCR page "
                    f"{page_number} error: {error}"
                )
                text = ""
            pages.append(
                {"page": page_number,"text": text,}
            )
            if page_number % 10 == 0:
                print(
                    f" OCR processed "
                    f"{page_number} pages..."
                )
        document.close()

    except Exception as error:
        print(f" OCR failed: {error}")
        return []
    return pages

def extract_pdf_text(pdf_path):
    print(f" Extracting: {pdf_path.name}")
    pages = extract_with_pymupdf(
        pdf_path
    )
    pymupdf_non_empty = sum(
        bool(page["text"].strip())
        for page in pages
    )
    if pymupdf_non_empty > 0:
        return pages
    print("  No usable PyMuPDF text.")
    print("  Trying pdfplumber...")

    pages = extract_with_pdfplumber(
        pdf_path
    )
    pdfplumber_non_empty = sum(
        bool(page["text"].strip())
        for page in pages
    )
    if pdfplumber_non_empty > 0:
        return pages
    print(" No usable pdfplumber text.")
    pages = extract_with_ocr(
        pdf_path
    )
    return pages

def load_device_inventory():
    inventory_path = (
        MANUALS_DIR.parent
        / "device_inventory.json"
    )
    if not inventory_path.exists():
        raise FileNotFoundError(
            f"Device inventory not found: "
            f"{inventory_path}"
        )
    with open(inventory_path, "r", encoding="utf-8",) as file:
        return json.load(file)

def get_stable_device_info(pdf_path, device_info):
    updated_info = dict(device_info)
    for stable_id in ("servo_ventilator", "sc6002xl", "philips_g40",):
        manual = MANUALS.get(stable_id)
        if not manual:
            continue

        matching_names = {Path(manual["file"]).name}
        matching_names.update(
            Path(alias_file).name
            for alias_file in manual.get("aliases", [])
        )

        if pdf_path.name in matching_names:
            updated_info["device_id"] = stable_id
            updated_info["device"] = manual.get(
                "device",
                updated_info.get("device", pdf_path.stem),
            )
            updated_info["manufacturer"] = manual.get(
                "manufacturer",
                updated_info.get("manufacturer", "Unknown"),
            )
            break
    return updated_info

def make_generic_chunk(
    pdf_path,
    device_info,
    page_number,
    text,
    chunk_number,
):
    text = fix_medical_terms(text)
    return {
        "chunk_id":
            f"{device_info['device_id']}"
            f"_p{page_number}"
            f"_c{chunk_number}",
        "device_id":
            device_info["device_id"],
        "device":
            device_info["device"],
        "manufacturer":
            device_info["manufacturer"],
        "page":
            page_number,
        "section":
            "General",
        "chunk_type":
            detect_chunk_type(
                text
            ),
        "error_code":
            detect_error_code(
                text
            ),
        "manual":
            pdf_path.name,
        "text":
            text,
    }
DEPENDENT_START = re.compile(
    r"^(if (it|they|not|so|off|on)\b|it\b|they\b|this\b|these\b|then\b|otherwise\b|else\b)",
    re.IGNORECASE,
)
def extract_generic_chunks(
    pdf_path,
    device_info,
    pages=None,
    exclude_pages=None,
):
    if pages is None:
        pages = extract_pdf_text(
            pdf_path
        )

    if exclude_pages:
        pages = [
            page_data
            for page_data in pages
            if page_data["page"] not in exclude_pages
        ]
    chunks = []
    max_chars = 1200
    for page_data in pages:
        page_number = page_data["page"]
        raw_text = page_data["text"]
        if not raw_text.strip():
            continue

        text = clean_text(
            raw_text
        )
        if not text:
            continue
        parts = re.split(r"(?<=[.!?])\s+", text)
        current_chunk = ""
        last_part = ""
        chunk_number = 0
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if len(current_chunk) + len(part) + 1 <= max_chars:
                current_chunk = f"{current_chunk} {part}".strip()
                last_part = part
            else:
                if current_chunk:
                    chunk_number += 1
                    chunks.append(
                        make_generic_chunk(
                            pdf_path=pdf_path,
                            device_info=device_info,
                            page_number=page_number,
                            text=current_chunk,
                            chunk_number=chunk_number,
                        )
                    )
                carry = (
                    last_part
                    if DEPENDENT_START.match(part) and 0 < len(last_part) <= 300
                    else ""
                )
                current_chunk = f"{carry} {part}".strip()
                last_part = part

        if current_chunk:
            chunk_number += 1
            chunks.append(
                make_generic_chunk(
                    pdf_path=pdf_path,
                    device_info=device_info,
                    page_number=page_number,
                    text=current_chunk,
                    chunk_number=chunk_number,
                )
            )
    return chunks

def extract_servo_error_chunks(pages, manual,):
    chunks = []
    inside_error_table = False
    for page in pages:
        page_number = page["page"]
        raw_text = page["text"]
        if not raw_text:
            continue
        text = clean_text(raw_text)
        if (
            "Technical error codes" in text
            and "Error code" in text
            and "Error message / Possible cause" in text
            and "Recommended action" in text
        ):
            inside_error_table = True
        if not inside_error_table:
            continue

        if (
            "Preventive maintenance" in text
            and "Technical error codes" not in text
        ):
            break
        pattern = r"""
        (?<!\d)
        (\d{1,5})
        \s+
        ([A-Z][A-Z0-9_ ]+?)
        (?=
            \s+\d+\.
            |
            \s+N/A
            |
            \s+\d{1,5}\s+[A-Z]
            |
            $
        )
        """
        matches = list(re.finditer(pattern, text, re.VERBOSE,))
        for index, match in enumerate(
            matches
        ):
            error_code = (
                match.group(1)
                .strip()
            )
            if error_code == "382":
                continue
            error_message = (match.group(2).strip())
            error_message = re.sub(r"_\s+", "_", error_message,)
            start = match.end()

            if index + 1 < len(matches):
                end = (matches[index + 1].start())
            else:
                end = len(text)
            action_text = (text[start:end].strip())
            action_text = re.sub(r"^Recommended action\s*", "", action_text, flags=re.IGNORECASE,)
            chunk_text = (
                f"Error code: {error_code}. "
                f"Error message / possible cause: "
                f"{error_message}. "
                f"Recommended action: "
                f"{action_text if action_text else 'Not specified.'}"
            )
            chunks.append(
                {
                    "device_id":
                        "servo_ventilator",
                    "device":
                        manual["device"],
                    "manufacturer":
                        manual["manufacturer"],
                    "page":
                        page_number,
                    "section":
                        "Technical error codes",
                    "chunk_type":
                        "error_code",
                    "error_code":
                        error_code,
                    "error_message":
                        error_message,
                    "recommended_action":
                        action_text,
                    "manual":
                        Path(manual["file"]).name,
                    "text":
                        chunk_text,
                }
            )
    return chunks

def extract_servo_malfunction_action_chunks(pages, manual):
    chunks = []
    pages_used = set()
    for page in pages:
        page_number = page["page"]
        raw_text = page["text"]
        if not raw_text:
            continue

        lines = raw_text.splitlines()
        malfunction_idx = None
        action_idx = None
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == "Malfunction" and malfunction_idx is None:
                malfunction_idx = i
            elif (
                stripped == "Action"
                and malfunction_idx is not None
                and action_idx is None
            ):
                action_idx = i
        if malfunction_idx is None or action_idx is None:
            continue
        malfunction_block = "\n".join(
            lines[malfunction_idx + 1: action_idx]
        ).strip()
        action_block = "\n".join(lines[action_idx + 1:]).strip()
        if not malfunction_block or not action_block:
            continue

        def split_entries(block):
            raw_entries = re.split(r"\n\s*\n", block)
            entries = []
            for entry in raw_entries:
                entry = " ".join(
                    line.strip()
                    for line in entry.splitlines()
                    if line.strip()
                )
                if not entry or re.fullmatch(r"\d{1,3}", entry):
                    continue
                entries.append(entry)
            return entries

        malfunctions = split_entries(malfunction_block)
        actions = split_entries(action_block)
        if not malfunctions or not actions:
            continue

        pages_used.add(page_number)
        if len(malfunctions) == len(actions):
            for malfunction, action in zip(malfunctions, actions):
                malfunction_clean = fix_medical_terms(malfunction)
                action_clean = fix_medical_terms(action)
                chunks.append(
                    {
                        "device_id": "servo_ventilator",
                        "device": manual["device"],
                        "manufacturer": manual["manufacturer"],
                        "page": page_number,
                        "section": "Troubleshooting",
                        "chunk_type": "troubleshooting",
                        "error_code": None,
                        "symptom": malfunction_clean,
                        "action": action_clean,
                        "manual": Path(manual["file"]).name,
                        "text": (
                            f"Malfunction: {malfunction_clean} "
                            f"Action: {action_clean}"
                        ),
                    }
                )
        else:
            print(
                f"  NOTE: page {page_number} malfunction/action counts "
                f"don't match ({len(malfunctions)} vs {len(actions)}); "
                f"keeping one combined chunk for this page instead of "
                f"risking an incorrect malfunction-to-action link."
            )
            combined_malfunctions = " ".join(
                fix_medical_terms(m) for m in malfunctions
            )
            combined_actions = " ".join(
                fix_medical_terms(a) for a in actions
            )
            chunks.append(
                {
                    "device_id": "servo_ventilator",
                    "device": manual["device"],
                    "manufacturer": manual["manufacturer"],
                    "page": page_number,
                    "section": "Troubleshooting",
                    "chunk_type": "troubleshooting",
                    "error_code": None,
                    "symptom": combined_malfunctions,
                    "action": combined_actions,
                    "manual": Path(manual["file"]).name,
                    "text": (
                        f"Malfunction: {combined_malfunctions} "
                        f"Action: {combined_actions}"
                    ),
                }
            )
    return chunks, pages_used

def extract_philips_troubleshooting_chunks(
    manual,
):
    chunks = []
    current_symptom = None
    try:
        with pdfplumber.open(
            manual["file"]
        ) as pdf:
            for page_number in range(
                45,
                51,
            ):
                if page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                tables = page.extract_tables()
                for table_number, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table:
                        continue

                    current_symptom = None
                    section = "Troubleshooting"
                    if page_number in [45, 46]:
                        section = "Power Problems"

                    elif page_number == 47:
                        section = "Display Problems"

                    elif page_number == 48:
                        if table_number == 1:
                            section = "Alarm Problems"
                        elif table_number == 2:
                            section = "NIBP Problems"

                    elif page_number == 49:
                        if table_number == 1:
                            section = "NIBP Problems"
                        elif table_number == 2:
                            section = "Temperature Problems"

                    elif page_number == 50:
                        if table_number == 1:
                            section = "SpO2 Problems"
                        elif table_number == 2:
                            section = "etCO2 Problems"
                        elif table_number == 3:
                            section = "C.O. Problems"

                    for row in table:
                        if not row:
                            continue
                        if len(row) < 3:
                            continue
                        symptom = clean_cell(row[0])
                        cause = clean_cell(row[1])
                        action = clean_cell(row[2])

                        if (
                            symptom.lower()
                            == "symptom"
                            and
                            "possible cause"
                            in cause.lower()
                        ):
                            continue
                        if symptom:
                            current_symptom = (symptom)
                        if not current_symptom:
                            continue
                        if not cause or not action:
                            continue

                        symptom = fix_medical_terms(current_symptom)
                        cause = fix_medical_terms(cause)
                        action = fix_medical_terms(action)
                        chunk_text = (
                            f"Symptom: {symptom}. "
                            f"Possible cause: {cause}. "
                            f"Action: {action}"
                        )
                        chunks.append(
                            {
                                "device_id":
                                    "philips_g40",
                                "device":
                                    manual["device"],
                                "manufacturer":
                                    manual["manufacturer"],
                                "page":
                                    page_number,
                                "section":
                                    section,
                                "chunk_type":
                                    "troubleshooting",
                                "error_code":
                                    None,
                                "symptom":
                                    symptom,
                                "possible_cause":
                                    cause,
                                "action":
                                    action,
                                "manual":
                                    Path(manual["file"]).name,
                                "text":
                                    chunk_text,
                            }
                        )
    except Exception as error:
        print(
            f"    Philips specialized parser "
            f"failed: {error}"
        )
    return chunks

_PLAIN_LANGUAGE_ALIASES = [
    (
        re.compile(r"areas of display missing|color contaminated", re.IGNORECASE),
        "Also described as: part of the screen isn't showing anything, "
        "or the screen colors look wrong. ",
    ),
    (
        re.compile(r"backlight fails to provide", re.IGNORECASE),
        "Also described as: the screen looks dim or dark. ",
    ),
    (
        re.compile(r"inoperative pixels", re.IGNORECASE),
        "Also described as: the screen has dead or stuck pixels. ",
    ),
    (
        re.compile(r"NO power", re.IGNORECASE),
        "Also described as: the monitor won't turn on. ",
    ),
]

def _build_plain_language_alias(symptom_text):
    for pattern, alias in _PLAIN_LANGUAGE_ALIASES:
        if pattern.search(symptom_text):
            return alias
    return ""

def extract_sc6002xl_troubleshooting_chunks( manual,):
    chunks = []
    try:
        with pdfplumber.open(
            manual["file"]
        ) as pdf:
            for page_number in range(
                73,
                79,
            ):
                if page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                tables = page.extract_tables()

                for table_number, table in enumerate(
                    tables,
                    start=1,
                ):
                    if not table:
                        continue
                    section = "Troubleshooting"
                    if page_number in [73, 74]:
                        section = "Power Problems"
                    elif page_number == 75:
                        section_map = {
                            1: "Power-off Alarm Malfunction",
                            2: "Power-up Process Malfunction",
                            3: "Rotary Knob Malfunction",
                            4: "LCD Display Malfunction",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 76:
                        section_map = {
                            1: "LCD Display Malfunction",
                            2: "Fixed Key Malfunction",
                            3: "Alarm Malfunctions",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 77:
                        section_map = {
                            1: "NBP Malfunctions",
                            2: "etCO2 Malfunctions",
                        }
                        section = section_map.get(
                            table_number,
                            "Troubleshooting",
                        )
                    elif page_number == 78:
                        section = "Recorder Malfunctions"
                    for row in table:
                        if not row:
                            continue
                        if len(row) < 3:
                            continue
                        symptom = clean_cell(row[0])
                        cause = clean_cell(row[1])
                        action = clean_cell(row[2])
                        if not symptom:
                            continue
                        if not cause or not action:
                            continue
                        symptom_lower = (symptom.lower())
                        if (
                            symptom_lower
                            in ["conditions", "symptom(s)", "symptoms", ] and
                            "possible cause"
                            in cause.lower()
                        ):
                            continue
                        symptom = fix_medical_terms(symptom)
                        cause = fix_medical_terms(cause)
                        action = fix_medical_terms(action)
                        plain_language_alias = _build_plain_language_alias(symptom)
                        chunk_text = (
                            f"Symptom or condition: "
                            f"{symptom}. "
                            f"{plain_language_alias}"
                            f"Possible cause: "
                            f"{cause}. "
                            f"Troubleshooting and remedial action: "
                            f"{action}"
                        )
                        chunks.append(
                            {
                                "device_id":
                                    "sc6002xl",
                                "device":
                                    manual["device"],
                                "manufacturer":
                                    manual["manufacturer"],
                                "page":
                                    page_number,
                                "section":
                                    section,
                                "chunk_type":
                                    "troubleshooting",
                                "error_code":
                                    None,
                                "symptom":
                                    symptom,
                                "possible_cause":
                                    cause,
                                "action":
                                    action,
                                "manual":
                                    Path(manual["file"]).name,
                                "text":
                                    chunk_text,
                            }
                        )

    except Exception as error:
        print(
            f"    SC6002XL specialized parser "
            f"failed: {error}"
        )
    return chunks

def apply_manual_overrides(chunks):
    path = MANUALS_DIR.parent / "manual_overrides.json"
    if not path.exists():
        return chunks
    with open(path, "r", encoding="utf-8") as file:
        overrides = json.load(file)
    replaced = {(o["device_id"], o["page"]) for o in overrides}
    kept = [
        c for c in chunks
        if not (
            (c["device_id"], c["page"]) in replaced
            and c.get("chunk_type") == "troubleshooting"
        )
    ]
    applied = 0
    for o in overrides:
        meta = next(
            (c for c in chunks if c["device_id"] == o["device_id"]),
            None,
        )
        if meta is None:
            print(
                f"  WARNING: override for unknown device "
                f"{o['device_id']} skipped."
            )
            continue
        kept.append({
            "device_id": o["device_id"],
            "device": meta["device"],
            "manufacturer": meta["manufacturer"],
            "page": o["page"],
            "section": o.get("section", "Troubleshooting"),
            "chunk_type": "troubleshooting",
            "error_code": None,
            "manual": meta.get("manual", ""),
            "text": (
                f"Malfunction: {fix_medical_terms(o['malfunction'])} "
                f"Action: {fix_medical_terms(o['action'])}"
            ),
        })
        applied += 1
    print(f"Applied {applied} manual override(s).")
    return kept

def build_all_chunks():
    all_chunks = []
    inventory = load_device_inventory()
    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files.")
    for index, pdf_path in enumerate(
        pdf_files,
        start=1,
    ):
        print()
        print(
            f"[{index}/{len(pdf_files)}] "
            f"{pdf_path.name}"
        )
        device_info = inventory.get(pdf_path.name)
        if device_info is None:
            print(
                "  WARNING: No inventory entry. "
                "Skipping."
            )
            continue

        device_info = get_stable_device_info(pdf_path, device_info,)
        try:
            chunks = []
            if ("servo_ventilator" in MANUALS and pdf_path.name == Path(
                    MANUALS[
                        "servo_ventilator"
                    ]["file"]
                ).name
            ):
                print("  Using Servo specialized parser...")
                pages = extract_pdf_text(
                    pdf_path
                )
                specialized_chunks = (
                    extract_servo_error_chunks(
                        pages,
                        MANUALS["servo_ventilator"],
                    )
                )
                print(
                    f"  Specialized parser found "
                    f"{len(specialized_chunks)} error-code chunks."
                )
                malfunction_action_chunks, troubleshooting_pages = (
                    extract_servo_malfunction_action_chunks(
                        pages,
                        MANUALS["servo_ventilator"],
                    )
                )
                print(
                    f"  Malfunction/Action parser found "
                    f"{len(malfunction_action_chunks)} troubleshooting chunks "
                    f"across {len(troubleshooting_pages)} page(s)."
                )
                print(
                    "  Also running generic parser for full document "
                    "coverage (skipping pages already covered by the "
                    "Malfunction/Action parser to avoid duplicate, "
                    "diluted chunks)..."
                )
                generic_chunks = extract_generic_chunks(
                    pdf_path,
                    device_info,
                    pages=pages,
                    exclude_pages=troubleshooting_pages,
                )
                chunks = (
                    specialized_chunks
                    + malfunction_action_chunks
                    + generic_chunks
                )
            elif ("philips_g40" in MANUALS and pdf_path.name ==
                Path(
                    MANUALS[
                        "philips_g40"
                    ]["file"]
                ).name
            ):
                print("Using Philips G40 specialized parser...")
                specialized_chunks = (
                    extract_philips_troubleshooting_chunks(
                        MANUALS[
                            "philips_g40"
                        ]
                    )
                )
                print(
                    f"  Specialized parser found "
                    f"{len(specialized_chunks)} troubleshooting chunks."
                )
                print(
                    "  Also running generic parser "
                    "for full document coverage..."
                )
                troubleshooting_pages = {
                    chunk["page"] for chunk in specialized_chunks
                }
                generic_chunks = extract_generic_chunks(
                    pdf_path,
                    device_info,
                    exclude_pages=troubleshooting_pages,
                )
                chunks = specialized_chunks + generic_chunks

            elif ("sc6002xl" in MANUALS and pdf_path.name ==
                Path(
                    MANUALS[
                        "sc6002xl"
                    ]["file"]
                ).name
            ):
                print("Using SC6002XL specialized parser...")
                specialized_chunks = (
                    extract_sc6002xl_troubleshooting_chunks(
                        MANUALS[
                            "sc6002xl"
                        ]
                    )
                )
                print(
                    f"  Specialized parser found "
                    f"{len(specialized_chunks)} troubleshooting chunks."
                )
                print(
                    "  Also running generic parser "
                    "for full document coverage..."
                )
                troubleshooting_pages = {
                    chunk["page"] for chunk in specialized_chunks
                }
                generic_chunks = extract_generic_chunks(
                    pdf_path,
                    device_info,
                    exclude_pages=troubleshooting_pages,
                )
                chunks = specialized_chunks + generic_chunks
            else:
                print("  Using generic parser...")
                chunks = extract_generic_chunks(pdf_path, device_info,)
            all_chunks.extend(chunks)
            print(
                f"  Chunks created: "
                f"{len(chunks)}"
            )

        except Exception as error:
            print(
                f"  ERROR while processing "
                f"{pdf_path.name}: {error}"
            )
    all_chunks = apply_manual_overrides(all_chunks)

    for index, chunk in enumerate(all_chunks, start=1,):
        chunk["chunk_id"] = (
            f"chunk_{index:05d}"
        )
    return all_chunks

def validate_chunks(chunks):

    print()
    print("=" * 70)
    print("CHUNK VALIDATION")
    print("=" * 70)

    required_fields = [
        "chunk_id",
        "device_id",
        "device",
        "manufacturer",
        "page",
        "section",
        "chunk_type",
        "text",
    ]
    missing_field_count = 0
    empty_text_count = 0
    duplicate_ids = 0
    seen_ids = set()
    for chunk in chunks:
        for field in required_fields:
            if field not in chunk:
                print(
                    f"Missing field '{field}' "
                    f"in chunk "
                    f"{chunk.get('chunk_id')}"
                )
                missing_field_count += 1

        if not chunk.get(
            "text",
            "",
        ).strip():
            empty_text_count += 1
        chunk_id = chunk.get(
            "chunk_id"
        )

        if chunk_id in seen_ids:
            duplicate_ids += 1
        seen_ids.add(chunk_id)
    print()
    print(
        f"Missing fields: "
        f"{missing_field_count}"
    )
    print(
        f"Empty texts: "
        f"{empty_text_count}"
    )
    print(
        f"Duplicate IDs: "
        f"{duplicate_ids}"
    )
    print()

    if (
        missing_field_count == 0
        and empty_text_count == 0
        and duplicate_ids == 0
    ):
        print("Validation passed.")
    else:
        print("Validation found problems.")
    print("=" * 70)

def save_chunks_to_json(chunks):
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_file = (
        PROCESSED_DIR
        / "maintai_chunks.json"
    )

    with open(output_file, "w", encoding="utf-8",) as file:
        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False,
        )
    print()
    print(f"Saved {len(chunks)} chunks")
    print(f"Output file: {output_file}")

def run_preprocessing():
    print("=" * 70)
    print("MAINTAI PREPROCESSING")
    print("=" * 70)
    chunks = build_all_chunks()
    print()
    print(
        f"Total chunks created: "
        f"{len(chunks)}"
    )
    validate_chunks(chunks)
    save_chunks_to_json(chunks)

if __name__ == "__main__":
    run_preprocessing()