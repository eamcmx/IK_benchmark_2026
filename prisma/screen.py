#!/usr/bin/env python3
"""
PRISMA screening pipeline for the IK-Benchmark literature scan.

Reads:
  - IKS_Scopus_RAW.csv      (293 records)
  - IKS_IEEE_RAW_1.csv      (70 records, broad query)
  - IKS_IEEE_RAW_2.csv      (9 records, narrow query)

Produces:
  - merged_dedup.csv        (unique records with source attribution)
  - screening_decisions.csv (per-record decision, trigger, rationale)
  - flow_counts.txt         (PRISMA flow numbers)

The screening rules implement the IC/EC agreed in PRISMA_search_log.txt.
Each rule is keyword-/regex-based and deterministic, so a re-run produces
identical output. Records flagged "UNCERTAIN" need human review.
"""

import csv
import re
import sys
from pathlib import Path
from collections import defaultdict

csv.field_size_limit(sys.maxsize)
HERE = Path(__file__).parent

SCOPUS = HERE / "IKS_Scopus_RAW.csv"
IEEE1  = HERE / "IKS_IEEE_RAW_1.csv"
IEEE2  = HERE / "IKS_IEEE_RAW_2.csv"


# ---------------------------------------------------------------------------
# Normalisation: load each source into a common schema.
# ---------------------------------------------------------------------------
def norm_scopus(row):
    return {
        "source"   : "Scopus-S2",
        "title"    : (row.get("Title") or "").strip(),
        "authors"  : (row.get("Authors") or "").strip(),
        "year"     : (row.get("Year") or "").strip(),
        "venue"    : (row.get("Source title") or "").strip(),
        "doc_type" : (row.get("Document Type") or "").strip(),
        "language" : (row.get("Language of Original Document") or "").strip(),
        "doi"      : (row.get("DOI") or "").strip().lower(),
        "abstract" : (row.get("Abstract") or "").strip(),
        "keywords" : ((row.get("Author Keywords") or "") + "; " +
                      (row.get("Index Keywords") or "")).strip("; "),
        "cited_by" : (row.get("Cited by") or "").strip(),
        "link"     : (row.get("Link") or "").strip(),
    }

def norm_ieee(row, src_label):
    return {
        "source"   : src_label,
        "title"    : (row.get("Document Title") or "").strip(),
        "authors"  : (row.get("Authors") or "").strip(),
        "year"     : (row.get("Publication Year") or "").strip(),
        "venue"    : (row.get("Publication Title") or "").strip(),
        "doc_type" : "Article" if row.get("Volume") else "Conference paper",
        "language" : "English",   # IEEE Xplore default; not in export
        "doi"      : (row.get("DOI") or "").strip().lower(),
        "abstract" : (row.get("Abstract") or "").strip(),
        "keywords" : ((row.get("Author Keywords") or "") + "; " +
                      (row.get("IEEE Terms") or "")).strip("; "),
        "cited_by" : (row.get("Article Citation Count") or "").strip(),
        "link"     : (row.get("PDF Link") or "").strip(),
    }


def load(path, normaliser, *args):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return [normaliser(r, *args) for r in reader]


def title_key(t):
    """Normalised key for fuzzy duplicate detection."""
    s = re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()
    return s


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
def deduplicate(records):
    by_doi = {}
    by_title = {}
    out = []
    n_dup_doi = 0
    n_dup_title = 0
    for r in records:
        d = r["doi"]
        tk = title_key(r["title"])
        merged = None
        if d and d in by_doi:
            merged = by_doi[d]
            n_dup_doi += 1
        elif tk and tk in by_title:
            merged = by_title[tk]
            n_dup_title += 1
        if merged is not None:
            # merge source labels and prefer Scopus metadata if present
            sources = set(merged["source"].split("|") + [r["source"]])
            merged["source"] = "|".join(sorted(sources))
            # prefer the longer abstract
            if len(r["abstract"]) > len(merged["abstract"]):
                merged["abstract"] = r["abstract"]
            continue
        if d:
            by_doi[d] = r
        if tk:
            by_title[tk] = r
        out.append(r)
    return out, n_dup_doi, n_dup_title


# ---------------------------------------------------------------------------
# Heuristic IC/EC screening
# ---------------------------------------------------------------------------
# Token / phrase patterns
_RX_IK          = re.compile(r"inverse\s+kinemat", re.I)
_RX_DIFF_ONLY   = re.compile(r"\b(differential\s+kinematics?|inverse\s+statics?)\b", re.I)
_RX_HUMAN_POSE  = re.compile(r"\b(human\s+pose|human\s+body|mesh\s+(estimation|recovery)|"
                             r"motion\s+capture|character\s+animation|animation\s+rigging)\b", re.I)
_RX_PARALLEL    = re.compile(r"\b(parallel\s+(robot|manipulator|mechanism)|stewart|delta\s+(robot|parallel)|"
                             r"hexapod|gough[-\s]stewart|tricept)\b", re.I)
_RX_CONTINUUM   = re.compile(r"\b(continuum\s+robot|soft\s+robot|cable[-\s]driven|"
                             r"concentric\s+tube|tendon[-\s]driven|hyper[-\s]redundant\s+continuum)\b", re.I)
_RX_SERIAL_HINT = re.compile(r"\b(serial\s+(manipulator|robot|chain|arm)|"
                             r"\d+[-\s]?dof\s+(serial|industrial|robot|arm|manipulator)|"
                             r"6r\s+robot|7r\s+robot|6\s*-?dof|7\s*-?dof|8\s*-?dof|"
                             r"ur5|ur10|ur3|panda|franka|lbr\s*iiwa|kuka|baxter|"
                             r"puma\s*560|motoman|kinova|jaco|abb\s+irb)\b", re.I)
_RX_ANALYTICAL_NOCMP = re.compile(r"\b(closed[-\s]form|analytical(\s+only)?\s+(solution|method|approach))\b", re.I)
_RX_COMPARE     = re.compile(r"\b(compar(ed|ison|ative|ing)|benchmark|versus|vs\.?\s+|"
                             r"outperform|baseline|better\s+than|propose[ds]?\s+\w+\s+(against|vs|compared))\b", re.I)
_RX_METRIC      = re.compile(r"\b(success\s+rate|accuracy|error|RMSE|iteration[s]?|"
                             r"runtime|computation\s+time|execution\s+time|convergence|"
                             r"manipulability|smoothness|jerk|trajectory\s+tracking)\b", re.I)
_RX_DOF_GE5     = re.compile(r"\b([5-9]|1[0-9])\s*[-\s]?(dof|degrees?\s+of\s+freedom|"
                             r"r\s+(robot|manipulator|arm))\b", re.I)
_RX_DOF_LT5     = re.compile(r"\b([1-4])\s*[-\s]?(dof|degrees?\s+of\s+freedom|"
                             r"r\s+(robot|manipulator|arm))\b", re.I)
_RX_REVIEW      = re.compile(r"\b(systematic\s+review|literature\s+review|survey|PRISMA)\b", re.I)
_RX_EXCLUDE_DOC = re.compile(r"\b(editorial|erratum|withdrawal|retracted)\b", re.I)


def screen(rec):
    """
    Apply IC/EC. Returns (decision, trigger, rationale).
    decision in {"INCLUDE", "EXCLUDE", "UNCERTAIN"}
    """
    text = (rec["title"] + " " + rec["abstract"] + " " + rec["keywords"]).strip()
    title = rec["title"]

    # --- IC7: language ---
    if rec["language"] and rec["language"].lower() not in ("english", ""):
        return "EXCLUDE", "IC7", f"language={rec['language']}"

    # --- IC5: doc type ---
    if _RX_EXCLUDE_DOC.search(title) or rec["doc_type"].lower() in (
        "editorial", "erratum", "retracted", "note", "letter", "short survey"):
        return "EXCLUDE", "EC5", f"doc_type={rec['doc_type']}"

    # --- IC6: date ---
    try:
        y = int(rec["year"])
        if y < 2005:
            return "EXCLUDE", "IC6", f"year={y} < 2005"
    except (ValueError, TypeError):
        return "UNCERTAIN", "IC6", "year not parseable"

    # --- EC4: human pose ---
    if _RX_HUMAN_POSE.search(text):
        return "EXCLUDE", "EC4", "human pose / mesh / animation"

    # If there is no abstract, fall back to title-only judgement and lean
    # toward UNCERTAIN unless title is unambiguous.
    if not rec["abstract"]:
        if not _RX_IK.search(title):
            return "EXCLUDE", "EC7", "no abstract; title not IK-related"
        return "UNCERTAIN", "EC7", "no abstract; manual review needed"

    # --- IC1: must be IK ---
    if not _RX_IK.search(text):
        return "EXCLUDE", "IC1", "no inverse-kinematics mention in title/abstract"

    # --- EC2: differential kinematics / inverse statics only ---
    if _RX_DIFF_ONLY.search(text) and not _RX_IK.search(text + " " + rec["keywords"]):
        return "EXCLUDE", "EC2", "differential kinematics / inverse statics only"

    # --- EC3: continuum / parallel / cable / soft only (no serial hint) ---
    is_parallel  = bool(_RX_PARALLEL.search(text))
    is_continuum = bool(_RX_CONTINUUM.search(text))
    has_serial   = bool(_RX_SERIAL_HINT.search(text))
    if (is_parallel or is_continuum) and not has_serial:
        cause = "parallel" if is_parallel else "continuum/cable/soft"
        return "EXCLUDE", "EC3", f"{cause}-only, no serial-arm benchmark"

    # --- IC4: DOF >= 5 ---
    has_ge5 = bool(_RX_DOF_GE5.search(text))
    has_lt5 = bool(_RX_DOF_LT5.search(text))
    if has_lt5 and not has_ge5:
        return "EXCLUDE", "IC4", "manipulator < 5 DOF"

    # --- EC1: pure analytical with no comparison ---
    has_compare = bool(_RX_COMPARE.search(text))
    has_metric  = bool(_RX_METRIC.search(text))
    if _RX_ANALYTICAL_NOCMP.search(text) and not has_compare:
        return "EXCLUDE", "EC1", "closed-form / analytical with no comparison"

    # --- IC2: comparison or benchmarking required ---
    if not has_compare and not _RX_REVIEW.search(text):
        return "EXCLUDE", "IC2", "no comparison/benchmark wording in abstract"

    # --- IC3: at least one quantitative metric mentioned ---
    if not has_metric and not _RX_REVIEW.search(text):
        return "UNCERTAIN", "IC3", "comparison present but no metric keyword"

    # --- INCLUDE ---
    if _RX_REVIEW.search(text):
        return "INCLUDE", "OK", "systematic review / survey"
    if has_ge5:
        return "INCLUDE", "OK", "IK + comparison + metric + serial >=5DOF"
    return "UNCERTAIN", "IC4", "DOF count not stated explicitly in abstract"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def main():
    scopus = load(SCOPUS, norm_scopus)
    ieee1  = load(IEEE1,  norm_ieee, "IEEE-S3")
    ieee2  = load(IEEE2,  norm_ieee, "IEEE-S4")
    all_recs = scopus + ieee1 + ieee2
    n_raw = len(all_recs)

    deduped, n_dup_doi, n_dup_title = deduplicate(all_recs)
    n_dedup = len(deduped)

    # Apply screening
    for r in deduped:
        r["decision"], r["trigger"], r["rationale"] = screen(r)

    # Write merged dedup
    merged_path = HERE / "merged_dedup.csv"
    fields = ["source","decision","trigger","rationale","year","title",
              "authors","venue","doc_type","language","doi","cited_by",
              "keywords","link","abstract"]
    with open(merged_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        for r in deduped:
            w.writerow({k: r.get(k,"") for k in fields})

    # Write screening decisions (compact)
    decisions_path = HERE / "screening_decisions.csv"
    with open(decisions_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["row","source","year","decision","trigger","rationale","title","doi"])
        for i, r in enumerate(deduped, 1):
            w.writerow([i, r["source"], r["year"], r["decision"],
                        r["trigger"], r["rationale"], r["title"], r["doi"]])

    # Counts
    counts = defaultdict(int)
    by_trigger = defaultdict(int)
    for r in deduped:
        counts[r["decision"]] += 1
        if r["decision"] == "EXCLUDE":
            by_trigger[r["trigger"]] += 1

    # PRISMA flow text
    flow = []
    flow.append("PRISMA FLOW (auto-generated by screen.py)")
    flow.append("=" * 60)
    flow.append(f"Identification:")
    flow.append(f"  Scopus  Search 2 records:               {len(scopus):4d}")
    flow.append(f"  IEEE Xplore Search 3 records:           {len(ieee1):4d}")
    flow.append(f"  IEEE Xplore Search 4 records:           {len(ieee2):4d}")
    flow.append(f"  Total identified:                       {n_raw:4d}")
    flow.append("")
    flow.append(f"Duplicates removed:")
    flow.append(f"  By DOI match:                           {n_dup_doi:4d}")
    flow.append(f"  By normalised title match:              {n_dup_title:4d}")
    flow.append(f"  Total duplicates:                       {n_dup_doi + n_dup_title:4d}")
    flow.append(f"  Records after deduplication:            {n_dedup:4d}")
    flow.append("")
    flow.append(f"Screening (title+abstract; automated heuristic IC/EC):")
    flow.append(f"  INCLUDE:                                {counts['INCLUDE']:4d}")
    flow.append(f"  UNCERTAIN (manual review needed):       {counts['UNCERTAIN']:4d}")
    flow.append(f"  EXCLUDE:                                {counts['EXCLUDE']:4d}")
    flow.append("")
    flow.append("Exclusion reasons (by trigger):")
    for k in sorted(by_trigger):
        flow.append(f"  {k:8s} {by_trigger[k]:4d}")
    flow.append("")
    flow.append("Files written:")
    flow.append(f"  {merged_path.name}")
    flow.append(f"  {decisions_path.name}")
    flow_path = HERE / "flow_counts.txt"
    flow_path.write_text("\n".join(flow), encoding="utf-8")
    print("\n".join(flow))


if __name__ == "__main__":
    main()
