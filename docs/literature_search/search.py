"""Systematic literature search: OpenAlex + arXiv queries and forward citations of seed papers.

Usage: python3 -I search.py OUT_DIR
Writes raw API responses to OUT_DIR/raw/ and merged records to OUT_DIR/records.jsonl.
Window: 2025-01-01 .. 2026-10-15.
"""
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

DATE_FROM, DATE_TO = "2025-01-01", "2026-10-15"
UA = {"User-Agent": "lit-search-script/1.0 (research use)"}

# Concept blocks; every query is an AND of blocks, each block an OR of terms.
A_OBJECT = ['"object map"', '"object maps"', '"object-level"', '"object-based"', '"object landmarks"',
            '"semantic landmarks"', '"scene graph"', '"scene graphs"', '"open-set"', '"open-vocabulary"',
            '"segment map"', '"object centroids"']
B_TASK = ['"map alignment"', '"map merging"', '"map matching"', '"data association"', 'registration',
          'relocalization', '"global localization"', '"loop closure"', '"frame alignment"',
          '"place recognition"', '"multi-robot"']
C_UNC = ['uncertainty', 'covariance', '"partial observation"', '"partial observations"', '"partial view"',
         '"partial views"', 'occlusion', 'truncation', '"noise bound"', 'conformal', 'calibrated', 'viewpoint']

QUERIES = {
    "Q1_object_alignment_uncertainty": [A_OBJECT, B_TASK, C_UNC],
    "Q2_centroid_size_partial_view": [
        ['centroid', '"object center"', '"object size"', '"bounding box"'],
        ['"partial observation"', '"partial view"', 'occlusion', 'truncation', '"partially observed"',
         '"partially visible"'],
        ['uncertainty', 'covariance', 'bias', 'calibration', 'conformal'],
        ['SLAM', 'mapping', 'localization', 'robot', '"data association"']],
    "Q3_registration_noise_bounds": [
        ['registration', '"data association"', 'correspondence', 'correspondences'],
        ['"noise bound"', '"noise bounds"', '"bounded noise"', '"unknown-but-bounded"', 'heteroscedastic',
         '"per-correspondence"', 'conformal', '"uncertainty-aware"'],
        ['robust', 'outlier', 'outliers', 'certifiable', '"maximum clique"', '"graph-theoretic"',
         '"consistency graph"']],
    "Q4_multirobot_object_alignment": [
        ['"multi-robot"', '"multi-agent"', 'collaborative', '"inter-robot"', '"cross-view"'],
        A_OBJECT + ['"semantic map"', 'segment'],
        ['"map alignment"', '"map merging"', 'registration', '"frame alignment"', '"relative pose"',
         '"global localization"', '"loop closure"']],
    # Domain scan without uncertainty terms: methods may handle partial views without saying so in the abstract.
    "Q5_object_alignment_domain": [
        A_OBJECT,
        ['"map alignment"', '"map merging"', 'relocalization', '"global localization"', '"frame alignment"',
         'submap', 'submaps', '"loop closure"', '"place recognition"']],
}

# Seed papers for forward citation search (arXiv id, published DOI or None).
SEEDS = {
    "TCAFF": ("2405.05210", None),
    "ROMAN": ("2410.08262", "10.15607/RSS.2025.XXI.029"),
    "MOTLEE": ("2304.12175", None),
    "SOS-Match": ("2401.04791", None),
    "CLIPPER_ICRA21": ("2011.10202", "10.1109/ICRA48506.2021.9561069"),
    "CLIPPER_2024": ("2402.07284", None),
    "CLIPPER+": ("2402.15464", None),
    "TEASER": ("2001.07715", "10.1109/TRO.2020.3033695"),
    "GAPSLAM": ("2303.14283", None),
    "PUMA": ("2311.03655", None),
    "DistributionEstimation": ("2509.15565", None),
    "Hickory": ("2606.01545", None),
    "Hydra-Multi": ("2304.13487", None),
    "QuadricSLAM": ("1804.04011", "10.1109/LRA.2018.2866205"),
    "CubeSLAM": ("1806.00557", "10.1109/TRO.2019.2909168"),
}


def get(url, tries=6, delay=3.0):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code in (401, 429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep(delay * (2 ** i))
                continue
            if e.code == 404:
                return None
            raise
        except (urllib.error.URLError, TimeoutError):
            if i < tries - 1:
                time.sleep(delay * (2 ** i))
                continue
            raise


def oa_abstract(inv):
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def norm_title(t):
    return re.sub(r"[^a-z0-9]", "", (t or "").lower())


def arxiv_id_from(s):
    m = re.search(r"(\d{4}\.\d{4,5})", s or "")
    return m.group(1) if m else None


def oa_record(w, source):
    loc = w.get("primary_location") or {}
    src = (loc.get("source") or {}).get("display_name")
    ids = w.get("ids") or {}
    doi = (w.get("doi") or "").replace("https://doi.org/", "") or None
    arx = arxiv_id_from(doi) if doi and "arxiv" in doi.lower() else None
    for l in w.get("locations") or []:
        arx = arx or arxiv_id_from((l.get("landing_page_url") or "") if "arxiv" in (l.get("landing_page_url") or "") else "")
    return {"title": w.get("title"), "date": w.get("publication_date"), "venue": src, "doi": doi,
            "arxiv": arx, "openalex": ids.get("openalex"), "abstract": oa_abstract(w.get("abstract_inverted_index")),
            "cited_by": w.get("cited_by_count"), "sources": [source]}


def oa_query(block_list):
    return " AND ".join("(" + " OR ".join(b) + ")" for b in block_list)


def openalex_works(filter_str, raw_path):
    if raw_path.exists():
        return [w for page in json.loads(raw_path.read_text()) for w in page["results"]]
    sel = "id,ids,doi,title,publication_date,primary_location,locations,abstract_inverted_index,cited_by_count"
    cursor, out, pages = "*", [], []
    while cursor:
        url = ("https://api.openalex.org/works?" + urllib.parse.urlencode(
            {"filter": filter_str, "per-page": 200, "cursor": cursor, "select": sel}))
        d = json.loads(get(url))
        pages.append(d)
        out.extend(d["results"])
        cursor = d["meta"].get("next_cursor") if d["results"] else None
        time.sleep(0.3)
    raw_path.write_text(json.dumps(pages))
    return out


def arxiv_query(block_list):
    def term(t):
        return f"abs:{t}" if t.startswith('"') else f"abs:{t}"
    q = " AND ".join("(" + " OR ".join(term(t) for t in b) + ")" for b in block_list)
    d0, d1 = DATE_FROM.replace("-", ""), DATE_TO.replace("-", "")
    return f"({q}) AND (cat:cs.RO OR cat:cs.CV) AND submittedDate:[{d0}0000 TO {d1}2359]"


def arxiv_search(q, raw_path):
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out, start, raws = [], 0, []
    cached = json.loads(raw_path.read_text()) if raw_path.exists() else None
    while True:
        url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            {"search_query": q, "start": start, "max_results": 500})
        txt = cached[len(raws)] if cached and len(raws) < len(cached) else get(url)
        raws.append(txt)
        root = ET.fromstring(txt)
        entries = root.findall("a:entry", ns)
        for e in entries:
            aid = arxiv_id_from(e.findtext("a:id", "", ns))
            out.append({"title": " ".join(e.findtext("a:title", "", ns).split()),
                        "date": e.findtext("a:published", "", ns)[:10], "venue": "arXiv",
                        "doi": None, "arxiv": aid, "openalex": None,
                        "abstract": " ".join(e.findtext("a:summary", "", ns).split()),
                        "cited_by": None, "sources": []})
        total = int(root.findtext("{http://a9.com/-/spec/opensearch/1.1/}totalResults") or 0)
        start += 500
        if not cached:
            time.sleep(3.5)
        if start >= total or not entries:
            break
    raw_path.write_text(json.dumps(raws))
    return out


def s2_citations(arxiv_id, raw_path):
    fields = "title,abstract,publicationDate,year,externalIds,venue,citationCount"
    out, offset, raws = [], 0, []
    cached = json.loads(raw_path.read_text()) if raw_path.exists() else None
    while True:
        if cached is not None and len(raws) >= len(cached):
            break
        url = (f"https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv_id}/citations?"
               + urllib.parse.urlencode({"fields": fields, "limit": 1000, "offset": offset}))
        if cached is not None:
            d = cached[len(raws)]
        else:
            txt = get(url, tries=8, delay=5.0)
            if txt is None:
                break
            d = json.loads(txt)
        raws.append(d)
        for c in d.get("data", []):
            p = c.get("citingPaper") or {}
            ext = p.get("externalIds") or {}
            out.append({"title": p.get("title"), "date": p.get("publicationDate") or (str(p.get("year")) if p.get("year") else None),
                        "venue": p.get("venue"), "doi": ext.get("DOI"), "arxiv": ext.get("ArXiv"), "openalex": None,
                        "abstract": p.get("abstract") or "", "cited_by": p.get("citationCount"),
                        "year": p.get("year"), "sources": []})
        if "next" not in d:
            break
        offset = d["next"]
        if cached is None:
            time.sleep(4)
    raw_path.write_text(json.dumps(raws))
    return out


def in_window(r):
    d = r.get("date") or ""
    y = r.get("year")
    if len(d) >= 10:
        return DATE_FROM <= d[:10] <= DATE_TO
    return (y or (int(d[:4]) if d[:4].isdigit() else 0)) in (2025, 2026)


def main(out_dir):
    out = Path(out_dir)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    records, log = [], {"queries": {}, "seeds": {}}
    date_f = f"from_publication_date:{DATE_FROM},to_publication_date:{DATE_TO}"

    for name, blocks in QUERIES.items():
        oa = openalex_works(f"title_and_abstract.search:{oa_query(blocks)},{date_f}", raw / f"openalex_{name}.json")
        for w in oa:
            records.append(oa_record(w, f"openalex:{name}"))
        ax = arxiv_search(arxiv_query(blocks), raw / f"arxiv_{name}.json")
        for r in ax:
            r["sources"] = [f"arxiv:{name}"]
        records.extend(ax)
        log["queries"][name] = {"openalex": len(oa), "arxiv": len(ax),
                                "openalex_query": oa_query(blocks), "arxiv_query": arxiv_query(blocks)}
        print(name, "openalex", len(oa), "arxiv", len(ax), flush=True)

    for name, (aid, doi) in SEEDS.items():
        entry = {"arxiv": aid, "doi": doi, "openalex_ids": [], "openalex_citing": 0, "s2_citing": None}
        for key in [f"doi:10.48550/arXiv.{aid}"] + ([f"doi:{doi}"] if doi else []):
            txt = get(f"https://api.openalex.org/works/{key}?select=id,title,cited_by_count")
            if not txt:
                continue
            wid = json.loads(txt)["id"].rsplit("/", 1)[-1]
            entry["openalex_ids"].append(wid)
            if wid in entry["openalex_ids"][:-1]:
                continue
            try:
                cit = openalex_works(f"cites:{wid},{date_f}", raw / f"openalex_cites_{name}_{wid}.json")
            except Exception as e:  # logged in search_log.json
                entry.setdefault("openalex_errors", []).append(f"{wid}: {e!r}")
                continue
            entry["openalex_citing"] += len(cit)
            for w in cit:
                records.append(oa_record(w, f"cites:{name}"))
        try:
            s2 = [r for r in s2_citations(aid, raw / f"s2_cites_{name}.json") if in_window(r)]
            for r in s2:
                r["sources"] = [f"cites:{name}"]
            records.extend(s2)
            entry["s2_citing"] = len(s2)
        except Exception as e:  # rate limit or API failure is logged, not hidden
            entry["s2_error"] = repr(e)
        log["seeds"][name] = entry
        print(name, entry, flush=True)
        time.sleep(2)

    # Merge duplicates by arXiv id, DOI or normalized title.
    merged, index = [], {}
    for r in records:
        if not r.get("title") or not in_window(r):
            continue
        keys = [k for k in (("ax", r.get("arxiv")), ("doi", (r.get("doi") or "").lower() or None),
                            ("t", norm_title(r["title"]))) if k[1]]
        hit = next((index[k] for k in keys if k in index), None)
        if hit is None:
            hit = len(merged)
            merged.append(dict(r))
        else:
            m = merged[hit]
            m["sources"] = sorted(set(m["sources"]) | set(r["sources"]))
            for f in ("abstract", "doi", "arxiv", "venue", "openalex"):
                if not m.get(f) and r.get(f):
                    m[f] = r[f]
            if (r.get("venue") or "arXiv") != "arXiv" and (m.get("venue") or "arXiv") == "arXiv":
                m["venue"] = r["venue"]
        for k in keys:
            index.setdefault(k, hit)
    log["records_before_dedup"] = len(records)
    log["records_after_dedup"] = len(merged)
    with open(out / "records.jsonl", "w") as f:
        for i, r in enumerate(merged):
            r["rid"] = f"R{i:04d}"
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "search_log.json").write_text(json.dumps(log, indent=1, ensure_ascii=False))
    print("records", len(records), "merged", len(merged))


if __name__ == "__main__":
    main(sys.argv[1])
