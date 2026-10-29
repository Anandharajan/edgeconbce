"""Claim-to-evidence tracer for LaTeX manuscripts: lists sentences that make quantitative or strong
claims without a traceable source, plus dangling PENDING markers and strong words.

A sentence passes if its numbers come from macros (\\nXxx), or it cites (\\cite) / references a
figure, table, section or equation (\\ref). Hand-typed numbers, unreferenced superlatives and
claim words ("significant", "optimal", "state-of-the-art", "deployable", "guarantee") are flagged
for the reviewer agents to verify with tools.

usage: python claim_trace.py paper.tex [--out claims.json]
"""
import argparse, json, re
from pathlib import Path

STRONG = ["significant", "significantly", "optimal", "optimally", "state-of-the-art", "state of the art",
          "deployable", "guarantee", "guarantees", "outperforms", "always", "never", "best", "first to", "novel"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("tex"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t = Path(a.tex).read_text(encoding="utf-8", errors="ignore")
    body = re.split(r"\\begin\{thebibliography\}", t)[0]
    body = re.sub(r"(?m)%.*$", "", body)
    body = re.sub(r"(?s)\\begin\{(table\*?|tabular|algorithmic|equation|align)\}.*?\\end\{\1\}", " ", body)
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", re.sub(r"\s+", " ", body))
    out = {"hand_typed_numbers": [], "strong_claims_unreferenced": [], "pending": []}
    for s in sents:
        has_src = bool(re.search(r"\\(cite|ref|eqref)\{|\\n[A-Z][A-Za-z]+", s))
        clean = re.sub(r"\\[a-zA-Z]+(\[[^]]*\])?(\{[^}]*\})?", " ", s)
        clean = re.sub(r"\$[^$]*\$", " ", clean)
        nums = re.findall(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:%|\\%|ms|s\b|x\b|×|USD|GB)", s.replace("\\,", ""))
        if nums and not has_src:
            out["hand_typed_numbers"].append({"numbers": nums[:5], "sentence": s[:260]})
        strong = [w for w in STRONG if re.search(r"\b" + re.escape(w) + r"\b", clean.lower())]
        if strong and not has_src:
            out["strong_claims_unreferenced"].append({"words": strong, "sentence": s[:260]})
        if "\\pending" in s or "PENDING" in s:
            out["pending"].append(s[:200])
    out["summary"] = {k: len(v) for k, v in out.items() if isinstance(v, list)}
    js = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(js)
    print(json.dumps(out["summary"]), "\n", js[:3000])


if __name__ == "__main__":
    main()
