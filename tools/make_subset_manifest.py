"""Emit the identity of the IEDB-derived evaluation subset.

The rule by which this subset was drawn from the source collection was not
recorded at the time. Releasing the exact membership does not recover that
rule, but it does make the evaluation set itself reproducible: anyone can
reconstruct the same 47 antigens from the source collection by identifier.

The 205 training antigens are not identified in the released artifacts, only
counted, and this script does not invent them.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--iedb-real-results", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    d = json.loads(Path(a.iedb_real_results).read_text(encoding="utf-8"))
    rows = d["external_test"]["local_only"]["per_sequence"]
    ds = d["dataset"]

    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sequence_id", "length", "positive_residue_count", "split"])
        for r in sorted(rows, key=lambda r: str(r["sequence_id"])):
            w.writerow([r["sequence_id"], r["length"],
                        r.get("positive_residue_count", ""), "test"])
    assert len(rows) == ds["test_sequence_count"], "manifest/count mismatch"
    print(f"wrote {len(rows)} test antigen identifiers to {out}")
    print(f"note: the {ds['train_sequence_count']} training antigens are counted "
          f"but not identified in the released artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
