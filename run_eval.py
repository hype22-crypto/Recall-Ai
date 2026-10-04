"""Runs the 25 test questions against your running RecallBuddy and prints a results table.
Usage (server running):   python run_eval.py          (instant mode: quotes the best note)
                          python run_eval.py --llm    (Gemma also checks the answer is really in the notes; slower)
                          ->  writes eval_results.csv
Rules: 'answer' questions must come back GROUNDED and contain the expected keyword.
       'no_claim' questions (the notes do not contain the answer) must NOT come back grounded."""
import csv
import json
import sys
import time
import urllib.request


def judge(row, out):
    state = out["state"]
    text = ((out.get("answer") or "") + " " + " ".join(c["text"] for c in out["citations"])).lower()
    if row["expect"] == "answer":
        if state == "grounded" and row["keyword"].lower() in text:
            return "CORRECT"
        return "PARTIAL" if state == "weak_evidence" else "MISSED"
    return "CORRECT REFUSAL" if state != "grounded" else "FALSE CLAIM"


def summarize(results):
    ans = [r for r in results if r["expect"] == "answer"]
    non = [r for r in results if r["expect"] != "answer"]
    c = lambda rs, v: sum(r["verdict"] == v for r in rs)
    return (f"Answerable ({len(ans)}): correct {c(ans,'CORRECT')}, partial {c(ans,'PARTIAL')}, missed {c(ans,'MISSED')}\n"
            f"Not in notes ({len(non)}): correctly refused {c(non,'CORRECT REFUSAL')}, FALSE CLAIMS {c(non,'FALSE CLAIM')}  (goal: 0)")


def main(base="http://127.0.0.1:8000", use_llm=False):
    rows = list(csv.DictReader(open("eval_questions.csv", encoding="utf-8")))
    results = []
    for i, row in enumerate(rows, 1):
        req = urllib.request.Request(base + "/recall", data=json.dumps({"question": row["question"], "use_llm": use_llm}).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        results.append({**row, "n": i, "state": out["state"], "verdict": judge(row, out), "secs": round(time.time() - t0, 2)})
        print(f"{i:>2}. {results[-1]['verdict']:<16} {out['state']:<14} {row['question']}")
    print("\n" + summarize(results))
    with open("eval_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n", "question", "expect", "keyword", "state", "verdict", "secs"])
        w.writeheader()
        w.writerows(results)
    print("Mode:", "with Gemma" if use_llm else "instant (no Gemma)")
    print("Saved eval_results.csv")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--llm"]
    main(*(args[:1]), use_llm="--llm" in sys.argv)
