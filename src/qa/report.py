"""
CLI for QA checks.
Usage examples:
  python src/qa/report.py --text "BP 300/150. Assessment: ... Plan: ..."
  python src/qa/report.py --infile data/examples/note1.txt --format md --out reports/examples/note1.qa.md
"""
# makes script runnable via `python src/qa/report.py ...` AND `python -m src.qa.report ...`
if __package__ is None or __package__ == "":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from qa.rules import run_qa
else:
    from .rules import run_qa

import argparse, json, sys
from pathlib import Path
from typing import List, Dict

def to_markdown(issues: List[Dict]) -> str:
    if not issues:
        return "### QA Report\n\n No issues found.\n"
    lines = ["### QA Report", ""]
    for i, it in enumerate(issues, 1):
        lines.append(f"- **{it.get('type','issue')}** — {it.get('msg','')}")
    lines.append("")
    return "\n".join(lines)

def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text", help="Raw note text")
    src.add_argument("--infile", type=Path, help="Path to a text file")

    ap.add_argument("--format", choices=["md","json"], default="md", help="Output format")
    ap.add_argument("--out", type=Path, help="Optional output path")

    args = ap.parse_args()

    if args.infile:
        try:
            text = args.infile.read_text(encoding="utf-8")
        except Exception as e:
            print(f"ERROR: cannot read {args.infile}: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        text = args.text or ""

    issues = run_qa(text)

    if args.format == "json":
        payload = json.dumps({"issues": issues}, indent=2)
    else:
        payload = to_markdown(issues)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    else:
        print(payload)

if __name__ == "__main__":
    main()
