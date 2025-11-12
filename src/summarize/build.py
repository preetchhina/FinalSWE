from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Dict, List, Iterable
import sys
from pathlib import Path

# Try to import normally first
try:
    from src.weak_ner.extract import extract_entities
except ModuleNotFoundError:
    # 1) Put the REPO ROOT on sys.path (…/Transcription-Enhancement)
    THIS = Path(__file__).resolve()
    REPO_ROOT = THIS.parents[2]  # repo/
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    try:
        from src.weak_ner.extract import extract_entities  # retry with repo root
    except ModuleNotFoundError:
        # 2) Fall back to adding the src/ dir and importing without the src. prefix
        SRC_DIR = THIS.parents[1]  # repo/src
        if str(SRC_DIR) not in sys.path:
            sys.path.append(str(SRC_DIR))
        from weak_ner.extract import extract_entities

SECTION_HDR_RE = re.compile(
    r'^\s*(ASSESSMENT\s*&\s*PLAN|ASSESSMENT/PLAN|ASSESSMENT|PLAN|A/P|P:)\s*[:\-]?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

# things that often look like section headers we should stop at
STOP_HDR_RE = re.compile(
    r'^\s*(HPI|HISTORY|MEDS|MEDICATIONS|ALLERGIES|EXAM|PHYSICAL\s+EXAM|ROS|LABS?|STUDIES|DIAGNOS(TIC|IS)|IMPRESSION|DISPOSITION|FOLLOW[- ]?UP|SIGNATURE)\s*[:\-]?\s*$',
    re.IGNORECASE | re.MULTILINE,
)

ACTION_VERBS = [
    # high-signal care-plan verbs
    "start", "continue", "restart", "initiate", "begin",
    "stop", "hold", "discontinue",
    "increase", "decrease", "titrate", "escalate", "de-escalate", "adjust",
    "switch", "change", "convert", "transition",
    "check", "obtain", "order", "repeat", "monitor",
    "refer", "consult", "arrange", "schedule",
    "counsel", "educate", "recommend",
    "prescribe", "refill",
    "follow", "follow-up", "return",
]

SENTENCE_SPLIT_RE = re.compile(r'(?<=[\.\?!])\s+(?=[A-Z0-9])')

def unique_preserve_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        key = x.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(x.strip())
    return out

def find_section_block(text: str, wanted: List[str]) -> str | None:
    """
    Return the text for the wanted section (e.g., ["PLAN","A/P"]),
    stopping at the next section header.
    """
    # Build a simple multi-header search
    # First, find all headers with positions
    headers = []
    for m in re.finditer(SECTION_HDR_RE, text):
        title = m.group(1).upper().replace(" ", "")
        headers.append((m.start(), m.end(), title))
    if not headers:
        return None

    # pick the first header whose normalized title contains any wanted token
    wanted_norm = [w.upper().replace(" ", "").replace("/", "") for w in wanted]
    target = None
    for st, en, title in headers:
        for w in wanted_norm:
            if w in title.replace("&", "").replace(":", ""):
                target = (st, en)
                break
        if target:
            break
    if not target:
        return None

    # slice from the end of the header to either next stop header or end
    start_idx = target[1]
    tail = text[start_idx:]

    # stop at next obvious header (STOP_HDR_RE) or the next known section header
    stops = []
    m1 = STOP_HDR_RE.search(tail)
    if m1:
        stops.append(m1.start())
    m2 = SECTION_HDR_RE.search(tail)
    if m2:
        stops.append(m2.start())
    stop_idx = min(stops) if stops else len(tail)

    block = tail[:stop_idx].strip()
    return block if block else None

def extract_plan_lines(text: str) -> List[str]:
    """
    Prefer explicit PLAN section; otherwise, harvest action-verb sentences.
    """
    # 1) Try to grab a real PLAN/A&P section
    plan_block = find_section_block(text, ["PLAN", "ASSESSMENT & PLAN", "A/P", "ASSESSMENT/PLAN"])
    if plan_block:
        # Split by lines, keep non-empty
        lines = [ln.strip(" -*\t") for ln in plan_block.splitlines()]
        # Filter obviously non-plan cruft (e.g., lone hyphens)
        lines = [ln for ln in lines if len(ln) > 1]
        # If it looks like one big paragraph, split into sentences
        if len(lines) <= 2:
            lines = [s.strip(" -*\t") for s in SENTENCE_SPLIT_RE.split(plan_block) if len(s.strip()) > 1]
        # Keep lines that look action-oriented or informative
        def looks_action(s: str) -> bool:
            s_low = s.lower()
            return any(v in s_low for v in ACTION_VERBS) or re.search(r'\b(q\d+w?|mg|mcg|units?|labs?|level|antibod|echo|pcr|follow)', s_low) is not None
        lines = [ln for ln in lines if looks_action(ln)]
        return unique_preserve_order(lines)

    # 2) Fallback: collect sentences with action verbs from the whole note
    sents = [s.strip() for s in SENTENCE_SPLIT_RE.split(text) if s.strip()]
    planish = []
    for s in sents:
        sl = s.lower()
        if any(v in sl for v in ACTION_VERBS):
            planish.append(s)
    return unique_preserve_order(planish)

def build_summary(text: str) -> str:
    ents = extract_entities(text)  # expects dicts with keys: text, label in {"DIAG","MED","DOSE",...}

    dx    = unique_preserve_order([e["text"] for e in ents if e.get("label") == "DIAG"])
    meds  = unique_preserve_order([e["text"] for e in ents if e.get("label") == "MED"])
    doses = unique_preserve_order([e["text"] for e in ents if e.get("label") == "DOSE"])

    plan_lines = extract_plan_lines(text)

    lines: List[str] = []
    lines.append("=== SOAP Summary ===")
    lines.append(f"Assessment (Dx): {', '.join(dx) if dx else '—'}")
    if meds:
        med_str = ", ".join(meds)
        if doses:
            med_str += f" | doses: {', '.join(doses)}"
        lines.append(f"Medications: {med_str}")
    else:
        lines.append("Medications: —")

    if plan_lines:
        lines.append("Plan:")
        for ln in plan_lines:
            # prefix bullets for readability
            lines.append(f"  • {ln}")
    else:
        lines.append("Plan: —")

    return "\n".join(lines)

def _demo() -> None:
    s = ("CONSULT: HISTORY & PHYSICAL — REASON: abdominal pain. "
         "ASSESSMENT & PLAN:\n"
         "- Check adalimumab trough and anti-drug antibodies.\n"
         "- Continue azathioprine 100 mg daily; reinforce adherence.\n"
         "- Consider switch to ustekinumab if high ADA/low trough.\n"
         "MEDS: adalimumab 40 mg q2w; azathioprine 100 mg daily.")
    print(build_summary(s))

if __name__ == "__main__":
    # Small CLI: `--text "..."` or pipe from stdin
    import argparse, sys
    ap = argparse.ArgumentParser(description="Build a simple SOAP-style summary from clinical text.")
    ap.add_argument("--text", type=str, default=None, help="Input clinical note text. If omitted, read from stdin.")
    args = ap.parse_args()
    text_in = args.text if args.text is not None else sys.stdin.read()
    print(build_summary(text_in))
