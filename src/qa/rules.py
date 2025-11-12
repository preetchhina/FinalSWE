"""
Simple, explainable QA rules for clinical notes/transcriptions.

- Extracts common vitals with tolerant regex (Temp, HR/Pulse, RR, BP, SpO2)
- Flags out-of-range values (plausibility checks)
- Flags missing key sections (Assessment, Plan, Medications) with alias support
- Designed to be lightweight & easily extended
"""

from __future__ import annotations
import re
from typing import List, Dict, Optional, Tuple

# -------------------------
# Config: plausible ranges
# -------------------------
PLAUSIBILITY = {
    "temp_f": (93.0, 107.0),     # Fahrenheit
    "temp_c": (33.9, 41.7),      # Celsius
    "hr":     (30, 220),         # bpm
    "rr":     (6, 60),           # breaths/min
    "spo2":   (70, 100),         # %
    "sbp":    (70, 250),         # systolic mmHg
    "dbp":    (30, 150),         # diastolic mmHg
}

# -------------------------
# Regex (compiled)
# -------------------------
# Temperature: "Temp 98.6 F", "Temperature: 37 C", "T 99°F"
TEMP_RE = re.compile(
    r"\b(?:T(?:emp(?:erature)?)?)[:\s]*"
    r"(?P<val>\d{2,3}(?:\.\d+)?)\s*°?\s*(?P<Unit>[FCfc])\b"
)

# Heart rate: "HR 84", "Heart rate: 102 bpm", "Pulse 60"
HR_RE = re.compile(
    r"\b(?:HR|Heart\s*rate|Pulse)\b[:\s]*"
    r"(?P<val>\d{1,3})\b"
)

# Respiratory rate: "RR 18", "Resp 22"
RR_RE = re.compile(
    r"\b(?:RR|Resp(?:irations)?)\b[:\s]*"
    r"(?P<val>\d{1,2})\b"
)

# SpO2: "SpO2 98%", "O2 sat 93 %", "O2 saturation: 96%"
SPO2_RE = re.compile(
    r"\b(?:SpO2|O2\s*sat(?:uration)?)\b[:\s]*"
    r"(?P<val>\d{2,3})\s*%?\b"
)

# Blood pressure: "BP 118/72", "Blood pressure: 130 / 80"
BP_RE = re.compile(
    r"\b(?:BP|Blood\s*pressure)\b[:\s]*"
    r"(?P<sbp>\d{2,3})\s*/\s*(?P<dbp>\d{2,3})\b"
)

# Section aliases (case-insensitive contains-check)
SECTION_ALIASES = {
    "Assessment": [r"assessment", r"a/p", r"assessment and plan", r"impression"],
    "Plan":       [r"\bplan\b", r"a/p", r"assessment and plan"],
    "Medications":[r"medications?", r"meds", r"current meds?", r"home meds?"],
}

def _contains_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t, flags=re.IGNORECASE) for p in patterns)

def flag_missing_sections(text: str) -> List[Dict]:
    flags: List[Dict] = []
    for canonical, pats in SECTION_ALIASES.items():
        if not _contains_any(text, pats):
            flags.append({"type": "section_missing", "msg": f"Missing section: {canonical}"})
    return flags

def _float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None

def _int(s: str) -> Optional[int]:
    try:
        return int(s)
    except Exception:
        return None

def _bound_check(name: str, val: float, low: float, high: float) -> Optional[Dict]:
    if val < low or val > high:
        return {"type": "vital_range", "msg": f"{name} out of plausible range ({val})"}
    return None

def _extract_temps(text: str) -> List[Tuple[float, str]]:
    vals: List[Tuple[float, str]] = []
    for m in TEMP_RE.finditer(text):
        v = _float(m.group("val"))
        u = m.group("Unit").upper()
        if v is not None:
            vals.append((v, u))
    return vals

def _extract_ints(re_pat: re.Pattern, text: str) -> List[int]:
    vals: List[int] = []
    for m in re_pat.finditer(text):
        v = _int(m.group("val"))
        if v is not None:
            vals.append(v)
    return vals

def _extract_bp(text: str) -> List[Tuple[int, int]]:
    vals: List[Tuple[int, int]] = []
    for m in BP_RE.finditer(text):
        sbp = _int(m.group("sbp"))
        dbp = _int(m.group("dbp"))
        if sbp is not None and dbp is not None:
            vals.append((sbp, dbp))
    return vals

def flag_impossible_vitals(text: str) -> List[Dict]:
    flags: List[Dict] = []

    # Temps
    for v, unit in _extract_temps(text):
        if unit == "F":
            low, high = PLAUSIBILITY["temp_f"]
            maybe = _bound_check("Temperature (F)", v, low, high)
            if maybe: flags.append(maybe)
        else:
            low, high = PLAUSIBILITY["temp_c"]
            maybe = _bound_check("Temperature (C)", v, low, high)
            if maybe: flags.append(maybe)

    # HR
    for hr in _extract_ints(HR_RE, text):
        low, high = PLAUSIBILITY["hr"]
        maybe = _bound_check("Heart rate", hr, low, high)
        if maybe: flags.append(maybe)

    # RR
    for rr in _extract_ints(RR_RE, text):
        low, high = PLAUSIBILITY["rr"]
        maybe = _bound_check("Respiratory rate", rr, low, high)
        if maybe: flags.append(maybe)

    # SpO2
    for s in _extract_ints(SPO2_RE, text):
        low, high = PLAUSIBILITY["spo2"]
        maybe = _bound_check("SpO2", s, low, high)
        if maybe: flags.append(maybe)

    # BP
    for sbp, dbp in _extract_bp(text):
        sbp_flag = _bound_check("Systolic BP", sbp, *PLAUSIBILITY["sbp"])
        dbp_flag = _bound_check("Diastolic BP", dbp, *PLAUSIBILITY["dbp"])
        if sbp_flag: flags.append(sbp_flag)
        if dbp_flag: flags.append(dbp_flag)

    return flags

def run_qa(text: str) -> List[Dict]:
    """Run all QA checks and return a flat list of issues."""
    return flag_impossible_vitals(text) + flag_missing_sections(text)
