"""Run with the server running:  python diagnose.py   then paste the output."""
import json
import time
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"


def get(path):
    return json.load(urllib.request.urlopen(BASE + path, timeout=120))


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))


notes = get("/timeline")
print(f"Notes loaded: {len(notes)}")
for n in notes:
    print("  -", n["id"], n["title"][:60])

for q in ["What is byte code?", "What are renewable resources?", "What is the capital of Australia?"]:
    print("\nQ:", q)
    for h in get("/search?k=3&q=" + urllib.parse.quote(q)):
        print(f'   score {h["score"]}  lex {h["lex"]}  vec {h["vec"]}  | {h["title"][:40]}  page {h["page"]}')
    t0 = time.time()
    r = post("/recall", {"question": q, "use_llm": False})
    print(f'   /recall -> {r["state"]}  verified={r.get("verified")}  in {time.time() - t0:.1f}s')
