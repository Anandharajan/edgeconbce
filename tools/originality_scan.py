"""Originality and voice scan for a manuscript (LaTeX, Markdown or plain text).

Approximates what similarity checkers (Turnitin/iThenticate, Crossref Similarity Check) and
editors look for, so that problems are fixed before submission:
  1. Overlap: word n-gram shingles (default n=8) shared with each source text in --sources
     (cited papers, the author's earlier papers/preprints, earlier manuscript versions).
  2. Style markers editors associate with templated or machine-generated prose: stock phrases,
     uniform sentence length (low "burstiness"), low lexical variety, transition-word openers.
It does NOT reproduce any commercial checker's score and is not a way to evade one; it is a
pre-submission self-check. Fix findings by rewriting from understanding and citing/quoting.

usage: python originality_scan.py paper.tex --sources src_dir/ [--n 8] [--out report.json]
"""
import argparse, json, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

STOCK = ["delve", "in the realm of", "plays a crucial role", "plays a pivotal role", "it is worth noting",
         "it is important to note", "a testament to", "paves the way", "in today's", "ever-evolving",
         "rapidly evolving landscape", "cutting-edge", "game-changer", "seamlessly", "leverage the power",
         "unlock the potential", "a myriad of", "holistic", "robust and scalable", "in conclusion",
         "furthermore", "moreover", "additionally", "notably", "harness", "underscore", "intricate",
         "tapestry", "navigate the complexities", "shed light on", "groundbreaking", "revolutionize",
         "not only", "comprehensive overview", "plethora"]
OPENERS = {"furthermore", "moreover", "additionally", "however", "notably", "consequently", "overall", "thus", "hence"}


def strip_tex(t: str) -> str:
    if "\\begin{document}" in t:
        t = t.split("\\begin{document}", 1)[1]
    t = re.sub(r"(?s)\\begin\{(thebibliography|equation\*?|align\*?|algorithmic|tabular\*?|table\*?|figure\*?)\}.*?\\end\{\1\}", " ", t)
    t = re.sub(r"%.*", " ", t)
    t = re.sub(r"\$[^$]*\$", " MATH ", t)
    t = re.sub(r"\\(cite|ref|label|eqref|includegraphics|input|gentable)\*?(\[[^]]*\])?\{[^}]*\}", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", t)
    t = re.sub(r"[{}~\\]", " ", t)
    return re.sub(r"\s+", " ", t)


def words(t): return re.findall(r"[a-z0-9']+", t.lower())


def shingles(ws, n): return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def sentences(t): return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", t) if len(s.split()) >= 4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paper"); ap.add_argument("--sources", default=None)
    ap.add_argument("--n", type=int, default=8); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    raw = Path(a.paper).read_text(encoding="utf-8", errors="ignore")
    text = strip_tex(raw) if a.paper.endswith(".tex") else raw
    ws = words(text)
    report = {"words": len(ws), "overlap": [], "style": {}, "flagged_sentences": []}

    if a.sources:
        mine = shingles(ws, a.n)
        for src in sorted(Path(a.sources).glob("*")):
            if not src.is_file():
                continue
            st = src.read_text(encoding="utf-8", errors="ignore")
            st = strip_tex(st) if src.suffix == ".tex" else st
            common = mine & shingles(words(st), a.n)
            if common:
                report["overlap"].append({"source": src.name, "shared_ngrams": len(common),
                                          "share_of_paper": round(len(common) / max(1, len(mine)), 4),
                                          "examples": sorted(common)[:5]})
        report["overlap"].sort(key=lambda r: -r["shared_ngrams"])

    sents = sentences(text)
    lens = [len(s.split()) for s in sents]
    low = text.lower()
    stock = {p: low.count(p) for p in STOCK if low.count(p)}
    openers = Counter(s.split()[0].strip(",").lower() for s in sents)
    per_1k = lambda c: round(1000 * c / max(1, len(ws)), 2)
    report["style"] = {
        "sentences": len(sents),
        "mean_sentence_words": round(statistics.mean(lens), 1) if lens else 0,
        "sentence_length_cv": round(statistics.pstdev(lens) / statistics.mean(lens), 2) if len(lens) > 1 else 0,
        "type_token_ratio_first_5k": round(len(set(ws[:5000])) / max(1, len(ws[:5000])), 3),
        "stock_phrases_per_1k_words": per_1k(sum(stock.values())),
        "stock_phrases": dict(sorted(stock.items(), key=lambda kv: -kv[1])),
        "transition_openers_per_1k_words": per_1k(sum(openers[o] for o in OPENERS)),
        "em_dashes_per_1k_words": per_1k(raw.count("---") + raw.count("\u2014")),
    }
    for s in sents:
        hits = [p for p in STOCK if p in s.lower()]
        if len(hits) >= 2 or len(s.split()) > 45:
            report["flagged_sentences"].append({"sentence": s[:300], "stock": hits, "words": len(s.split())})
    # guidance thresholds (heuristics, not any checker's rule)
    st_ = report["style"]
    report["warnings"] = [w for w, cond in [
        ("sentence lengths very uniform (CV < 0.35): vary rhythm", st_["sentence_length_cv"] < 0.35),
        ("stock phrases > 1.5 per 1k words", st_["stock_phrases_per_1k_words"] > 1.5),
        ("transition-word openers > 6 per 1k words", st_["transition_openers_per_1k_words"] > 6),
        ("a single source shares > 3% of the paper's n-grams", any(o["share_of_paper"] > 0.03 for o in report["overlap"])),
    ] if cond]
    out = json.dumps(report, indent=2)
    if a.out:
        Path(a.out).write_text(out)
    print(out[:4000])


if __name__ == "__main__":
    main()
