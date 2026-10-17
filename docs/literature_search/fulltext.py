"""Download arXiv full text (HTML, PDF fallback) and print sentences with key terms.

Usage: python3 -I fulltext.py OUT_DIR RID:ARXIVID [RID:ARXIVID ...]
"""
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

TERMS = re.compile(r"covarian|uncertaint|partial(ly)? (observ|view|visib)|occlu|truncat|centroid|"
                   r"\bsize\b|extent|threshold|gate|gating|calibrat|conformal|\bbias|noise bound|bound on", re.I)


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "lit-search-script/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def text_for(aid, out):
    txt_path = out / f"{aid}.txt"
    if txt_path.exists():
        return txt_path.read_text()
    try:
        html = fetch(f"https://arxiv.org/html/{aid}").decode("utf-8", "replace")
        text = re.sub(r"<(script|style|math)[^>]*>.*?</\1>", " ", html, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        if len(text) < 20000:
            raise ValueError("short html")
    except Exception:
        pdf = fetch(f"https://arxiv.org/pdf/{aid}")
        text = subprocess.run(["pdftotext", "-", "-"], input=pdf, capture_output=True).stdout.decode("utf-8", "replace")
    text = re.sub(r"\s+", " ", text)
    txt_path.write_text(text)
    time.sleep(3)
    return text


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[2:]:
        rid, aid = arg.split(":")
        try:
            text = text_for(aid, out)
        except Exception as e:
            print(f"##### {rid} {aid}: FAILED {e!r}\n")
            continue
        sents = re.split(r"(?<=[.!?])\s+", text)
        hits = [s.strip() for s in sents if TERMS.search(s) and 40 < len(s) < 600]
        print(f"##### {rid} arXiv:{aid} chars={len(text)} matched={len(hits)}")
        for s in hits[:45]:
            print("-", s)
        print()


if __name__ == "__main__":
    main()
