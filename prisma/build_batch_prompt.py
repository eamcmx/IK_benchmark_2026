#!/usr/bin/env python3
# Generate the canonical CLAUDE.md NotebookLM prompt for a batch of rows.
# Usage: python3 build_batch_prompt.py <start_row> <end_row>

import csv, sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
HERE = Path(__file__).parent
CSV_PATH = HERE / "merged_dedup.csv"


def main():
    if len(sys.argv) != 3:
        print("Usage: build_batch_prompt.py <start_row> <end_row>", file=sys.stderr)
        sys.exit(1)
    start = int(sys.argv[1])
    end = int(sys.argv[2])

    with open(CSV_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    paper_lines = []
    for i in range(start - 1, end):
        if i >= len(rows):
            break
        r = rows[i]
        doi = (r.get("doi") or "").strip() or "NONE"
        paper_lines.append("Row " + str(i + 1) + ": DOI " + doi)

    body = []
    body.append("I need you to help screen papers for a PRISMA systematic review against fixed eligibility criteria. The single source is merged_dedup.csv, which has 343 records. Each row has columns: source, decision, trigger, rationale, year, title, authors, venue, doc_type, language, doi, cited_by, keywords, link, abstract. Treat the existing decision/trigger columns as a previous heuristic that you can override.")
    body.append("")
    body.append("")
    body.append("INCLUSION CRITERIA (a paper must satisfy ALL):")
    body.append("IC1. Addresses the inverse kinematics problem for a serial revolute robotic manipulator.")
    body.append("IC2. Either (a) compares >=2 IK methods on a common testbed, or (b) proposes a new method and benchmarks it against >=1 prior method. Cross-family comparison NOT required (PSO vs GA vs ABC qualifies).")
    body.append("IC3. Reports >=1 quantitative comparison metric (success rate, accuracy, error, RMSE, iterations, time, smoothness, manipulability, etc).")
    body.append("IC4. Manipulator has >=5 DOF.")
    body.append("IC5. Peer-reviewed journal article OR conference paper.")
    body.append("IC6. Published 2005-01-01 or later.")
    body.append("IC7. English language.")
    body.append("")
    body.append("")
    body.append("EXCLUSION CRITERIA (any one disqualifies):")
    body.append("EC1. Pure analytical / closed-form IK with no comparison.")
    body.append("EC2. Inverse statics, differential kinematics, or trajectory optimisation only - no IK position/orientation problem solved.")
    body.append("EC3. Cable-driven, continuum, soft, or parallel-only robots (kept ONLY if also benchmarks serial-arm IK methods).")
    body.append("EC4. Human pose estimation, animation rigging, or 3D mesh reconstruction.")
    body.append("EC5. Editorial, erratum, retracted, withdrawn, book, thesis.")
    body.append("EC7. No abstract AND title insufficient to assess relevance.")
    body.append("")
    body.append("")
    body.append("For papers in ROWS " + str(start) + " THROUGH " + str(end) + " of merged_dedup.csv (skip the header; first data row = 1), evaluate each paper against the IC/EC.")
    body.append("")
    body.append("I am providing ROW + DOI so you can locate each paper unambiguously in merged_dedup.csv:")
    body.extend(paper_lines)
    body.append("")
    body.append("")
    body.append("Return ONE LINE PER PAPER, pipe-separated, in this format:")
    body.append("ROW|YEAR|DECISION|TRIGGER|RATIONALE|DOI|TITLE_SHORT")
    body.append("")
    body.append("")
    body.append("DECISION = INCLUDE / EXCLUDE / UNCERTAIN")
    body.append("TRIGGER = IC/EC code (or OK)")
    body.append("RATIONALE = one short sentence, no pipes")
    body.append("DOI = the DOI you used to identify the paper (echo back, or NONE if no DOI)")
    body.append("TITLE_SHORT = title trimmed to 60 chars")
    body.append("")
    body.append("if there is no explicit mention in the abstract about the IC or coincides with the EC the result must be exclude")
    body.append("")
    body.append("No prose. Just 10 lines.")

    print("\n".join(body))


if __name__ == "__main__":
    main()
