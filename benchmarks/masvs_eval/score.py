"""Summarises a MasvsBenchmarkTest run: did the right pages reach the model, how long it took, what it answered.

    python benchmarks/masvs_eval/score.py run.json [other_run.json ...]

With several runs it also prints a per-question comparison of topic coverage and time.
"""
import json
import sys
from pathlib import Path

GOLD = json.loads((Path(__file__).parent / "gold.json").read_text(encoding="utf-8"))["questions"]


def pages(source):
    span = str(source.get("pageSpan") or source.get("page"))
    parts = [int(p) for p in span.split("-") if p.strip().isdigit()]
    return set(range(parts[0], parts[-1] + 1)) if parts else set()


def summarise(path):
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for r in run["results"]:
        gold = GOLD[str(r["n"])]
        shown = set().union(*[pages(s) for s in r["sources"]]) if r["sources"] else set()
        covered = sum(1 for g in gold["groups"] if shown & set(g))
        m = r.get("metrics") or {}
        rows.append({
            "n": r["n"],
            "covered": covered,
            "topics": len(gold["groups"]),
            "notInDoc": gold.get("notInDoc", False),
            "shown": sorted(shown),
            "tokens": m.get("assembledTokens"),
            "secs": round(r["elapsedMs"] / 1000, 1),
            "answer": r["answer"],
            "question": r["question"],
        })
    return run, rows


def main(paths):
    runs = [summarise(p) for p in paths]
    for run, rows in runs:
        full = sum(1 for r in rows if r["covered"] == r["topics"])
        print(f"\n=== {run['label']} | {run.get('model')} | {run.get('device')} | index {run.get('indexMs', 0) / 1000:.0f}s, {run.get('chunks')} chunks")
        print(f"all topics retrieved: {full}/{len(rows)} | mean time {sum(r['secs'] for r in rows) / len(rows):.1f}s")
        for r in rows:
            flag = "NOT-IN-DOC" if r["notInDoc"] else f"{r['covered']}/{r['topics']}"
            print(f"\n[Q{r['n']}] {flag} pages={r['shown']} tok={r['tokens']} {r['secs']}s :: {r['question']}")
            print("   " + r["answer"].replace("\n", "\n   "))
    if len(runs) > 1:
        print("\n=== comparison (topics retrieved / seconds)")
        print("Q   " + "  ".join(f"{run['label']:>18}" for run, _ in runs))
        for i, r in enumerate(runs[0][1]):
            cells = []
            for _, rows in runs:
                o = next((x for x in rows if x["n"] == r["n"]), None)
                cells.append(f"{o['covered']}/{o['topics']} {o['secs']:>5}s" if o else "-")
            print(f"{r['n']:<4}" + "  ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main(sys.argv[1:])
