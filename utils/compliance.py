from collections import Counter
from typing import List, Dict


REQUIRED_PPE = ["Hard Hat", "Safety Vest", "Safety Glasses", "Gloves", "Boots"]


def build_summary(detected_labels: List[str]) -> Dict[str, object]:
    counts = Counter(detected_labels)
    missing = [item for item in REQUIRED_PPE if counts.get(item, 0) == 0]

    if missing:
        status = "Non-Compliant"
    else:
        status = "Compliant"

    return {
        "status": status,
        "counts": dict(counts),
        "missing": missing,
    }
