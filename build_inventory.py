import json
import re
from config import MANUALS_DIR

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")

def infer_manufacturer(filename: str) -> str:
    name = filename.lower()
    if any(
        x in name
        for x in ["siemens", "acuson", "magnetom", "somatom"]
    ):
        return "Siemens"

    if any(
        x in name
        for x in ["philips", "agilent", "heartstart"]
    ):
        return "Philips"

    if any(
        x in name
        for x in ["ge healthcare", "ge_"]
    ):
        return "GE"
    return "Unknown"

def build_inventory():
    pdf_files = sorted(MANUALS_DIR.glob("*.pdf"))
    inventory = {}

    for pdf_path in pdf_files:
        stem = pdf_path.stem
        stem = re.sub(
            r"^(copy of\s*)+",
            "",
            stem,
            flags=re.IGNORECASE,
        )
        inventory[pdf_path.name] = {
            "device_id": slugify(stem),
            "device": stem,
            "manufacturer": infer_manufacturer(
                pdf_path.name
            ),
        }
    output_path = (MANUALS_DIR.parent / "device_inventory.json")

    with open(output_path,"w",encoding="utf-8",) as f:
        json.dump( inventory,f,indent=4,ensure_ascii=False)
    print(f"PDFs found: {len(pdf_files)}")
    print(f"Inventory saved to: {output_path}")

if __name__ == "__main__":
    build_inventory()