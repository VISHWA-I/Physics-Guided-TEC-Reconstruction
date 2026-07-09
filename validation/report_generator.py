"""
report_generator.py
===================
Generates Phase 1 Validation Reports in:
  - TXT  (plain text summary table)
  - HTML (styled, colour-coded report)
  - JSON (machine-readable results)

Author  : Senior Python QA Engineer
Project : Topside Ionosphere-Plasmasphere TEC Reconstruction
Phase   : 1 Validation
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List

from validation.validation_utils import ResultTracker, STATUS_PASS, STATUS_WARN, STATUS_FAIL


# ---------------------------------------------------------------------------
# TXT Report
# ---------------------------------------------------------------------------

def generate_txt_report(tracker: ResultTracker, output_path: Path) -> None:
    """Write a plain-text summary report."""
    lines: List[str] = []
    sep = "=" * 80

    lines.append(sep)
    lines.append("  PHASE 1 VALIDATION REPORT")
    lines.append("  Physics-Guided Hybrid Mamba-TKAN Framework")
    lines.append("  Global Topside Ionosphere-Plasmasphere TEC Reconstruction")
    lines.append(f"  Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    lines.append(f"  {'Test':<55}  {'Status':^9}")
    lines.append(f"  {'─'*55}  {'─'*9}")

    for r in tracker.results:
        badge = f"[{r.status}]"
        lines.append(f"  {r.name:<55}  {badge:^9}")

    lines.append("")
    lines.append(f"  {'─'*68}")
    lines.append(
        f"  TOTAL: {tracker.total}   "
        f"PASS: {tracker.n_pass}   "
        f"WARNING: {tracker.n_warn}   "
        f"FAIL: {tracker.n_fail}   "
        f"SKIP: {tracker.n_skip}"
    )
    lines.append("")
    if tracker.overall_pass and tracker.n_warn == 0:
        lines.append("  OVERALL STATUS:  *** READY FOR PHASE 2 ***")
    elif tracker.overall_pass:
        lines.append("  OVERALL STATUS:  *** READY FOR PHASE 2 (with warnings) ***")
    else:
        lines.append(
            f"  OVERALL STATUS:  *** NOT READY — "
            f"Fix {tracker.n_fail} failure(s) before Phase 2 ***"
        )
    lines.append("")
    lines.append(sep)

    # Detailed failures
    failures = [r for r in tracker.results if r.status == STATUS_FAIL]
    if failures:
        lines.append("")
        lines.append("  FAILURES REQUIRING ATTENTION:")
        lines.append(f"  {'─'*68}")
        for i, r in enumerate(failures, 1):
            lines.append(f"  {i:>2}. [{r.status}] {r.name}")
            if r.detail:
                lines.append(f"       Detail: {r.detail}")
        lines.append("")

    # Detailed warnings
    warnings = [r for r in tracker.results if r.status == STATUS_WARN]
    if warnings:
        lines.append("")
        lines.append("  WARNINGS (non-critical):")
        lines.append(f"  {'─'*68}")
        for i, r in enumerate(warnings, 1):
            lines.append(f"  {i:>2}. [{r.status}] {r.name}")
            if r.detail:
                lines.append(f"       Detail: {r.detail}")
        lines.append("")

    lines.append(sep)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  [TXT]  Report saved: {output_path}")


# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

_HTML_CSS = """
body {
    font-family: 'Segoe UI', Arial, sans-serif;
    background: #0f1117;
    color: #e0e0e0;
    margin: 0;
    padding: 20px;
}
.container {
    max-width: 1100px;
    margin: 0 auto;
    background: #1a1d2e;
    border-radius: 12px;
    padding: 40px;
    box-shadow: 0 4px 30px rgba(0,0,0,0.5);
}
h1 {
    color: #7eb8f7;
    font-size: 1.6em;
    border-bottom: 2px solid #2d3155;
    padding-bottom: 10px;
    margin-bottom: 6px;
}
.subtitle {
    color: #9a9dc0;
    font-size: 0.9em;
    margin-bottom: 30px;
}
.meta-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 12px;
    margin-bottom: 30px;
}
.meta-card {
    background: #252840;
    border-radius: 8px;
    padding: 14px;
    text-align: center;
}
.meta-card .count {
    font-size: 2em;
    font-weight: 700;
}
.meta-card .label {
    font-size: 0.75em;
    color: #9a9dc0;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.count.green { color: #4ade80; }
.count.yellow { color: #facc15; }
.count.red { color: #f87171; }
.count.blue { color: #7eb8f7; }
.count.grey { color: #9a9dc0; }

.overall-banner {
    padding: 18px 24px;
    border-radius: 10px;
    font-size: 1.2em;
    font-weight: 700;
    margin-bottom: 30px;
    text-align: center;
    letter-spacing: 1px;
}
.overall-pass { background: #14532d; color: #4ade80; border: 1px solid #4ade80; }
.overall-warn { background: #713f12; color: #facc15; border: 1px solid #facc15; }
.overall-fail { background: #450a0a; color: #f87171; border: 1px solid #f87171; }

table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.88em;
}
th {
    background: #252840;
    color: #9a9dc0;
    padding: 10px 14px;
    text-align: left;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-size: 0.8em;
}
td {
    padding: 8px 14px;
    border-bottom: 1px solid #252840;
    vertical-align: top;
}
tr:hover td { background: #1e2235; }
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-pass    { background: #14532d; color: #4ade80; }
.badge-warning { background: #713f12; color: #facc15; }
.badge-fail    { background: #450a0a; color: #f87171; }
.badge-skip    { background: #1e2235; color: #9a9dc0; }
.detail { color: #9a9dc0; font-size: 0.85em; }
.section-header {
    background: #252840;
    color: #7eb8f7;
    font-weight: 700;
    padding: 10px 14px;
    font-size: 0.85em;
    letter-spacing: 0.5px;
}
.footer {
    margin-top: 30px;
    text-align: center;
    color: #555875;
    font-size: 0.8em;
}
"""


def generate_html_report(tracker: ResultTracker, output_path: Path) -> None:
    """Write a styled HTML validation report."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    if tracker.overall_pass and tracker.n_warn == 0:
        banner_cls = "overall-pass"
        banner_msg = "[OK] OVERALL STATUS: READY FOR PHASE 2"
    elif tracker.overall_pass:
        banner_cls = "overall-warn"
        banner_msg = "[OK] OVERALL STATUS: READY FOR PHASE 2 (with warnings)"
    else:
        banner_cls = "overall-fail"
        banner_msg = f"[FAIL] OVERALL STATUS: NOT READY -- {tracker.n_fail} failure(s)"

    rows_html: List[str] = []
    prev_section = ""

    for r in tracker.results:
        # Detect section groups by name prefix
        current_section = r.name.split(":")[0].split("—")[0].strip()
        if current_section != prev_section and len(current_section) < 30:
            prev_section = current_section

        if r.status == STATUS_PASS:
            badge = '<span class="badge badge-pass">PASS</span>'
        elif r.status == STATUS_WARN:
            badge = '<span class="badge badge-warning">WARN</span>'
        elif r.status == STATUS_FAIL:
            badge = '<span class="badge badge-fail">FAIL</span>'
        else:
            badge = '<span class="badge badge-skip">SKIP</span>'

        detail_html = f'<span class="detail">{_esc(r.detail)}</span>' if r.detail else ""
        ts_html = f'<span class="detail">{r.timestamp}</span>'

        rows_html.append(
            f"<tr>"
            f"<td>{_esc(r.name)}</td>"
            f"<td>{badge}</td>"
            f"<td>{detail_html}</td>"
            f"<td>{ts_html}</td>"
            f"</tr>"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Phase 1 Validation Report — Mamba-TKAN TEC Reconstruction</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<div class="container">
  <h1>Phase 1 Validation Report</h1>
  <div class="subtitle">
    Physics-Guided Hybrid Mamba–TKAN Framework for Global Topside
    Ionosphere–Plasmasphere TEC Reconstruction<br>
    Generated: {ts}
  </div>

  <div class="meta-grid">
    <div class="meta-card">
      <div class="count blue">{tracker.total}</div>
      <div class="label">Total Tests</div>
    </div>
    <div class="meta-card">
      <div class="count green">{tracker.n_pass}</div>
      <div class="label">Passed</div>
    </div>
    <div class="meta-card">
      <div class="count yellow">{tracker.n_warn}</div>
      <div class="label">Warnings</div>
    </div>
    <div class="meta-card">
      <div class="count red">{tracker.n_fail}</div>
      <div class="label">Failed</div>
    </div>
    <div class="meta-card">
      <div class="count grey">{tracker.n_skip}</div>
      <div class="label">Skipped</div>
    </div>
  </div>

  <div class="overall-banner {banner_cls}">{banner_msg}</div>

  <table>
    <thead>
      <tr>
        <th style="width:45%">Test</th>
        <th style="width:10%">Status</th>
        <th style="width:30%">Detail</th>
        <th style="width:15%">Timestamp</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>

  <div class="footer">
    Phase 1 Validation Suite · Mamba-TKAN TEC Reconstruction Project ·
    {ts}
  </div>
</div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"  [HTML] Report saved: {output_path}")


# ---------------------------------------------------------------------------
# JSON Report
# ---------------------------------------------------------------------------

def generate_json_report(tracker: ResultTracker, output_path: Path) -> None:
    """Write a machine-readable JSON report."""
    data = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "project": "Physics-Guided Hybrid Mamba-TKAN TEC Reconstruction",
        "phase": 1,
        "summary": {
            "total": tracker.total,
            "pass":    tracker.n_pass,
            "warning": tracker.n_warn,
            "fail":    tracker.n_fail,
            "skip":    tracker.n_skip,
            "overall_pass": tracker.overall_pass,
            "ready_for_phase2": tracker.overall_pass,
        },
        "results": [
            {
                "name": r.name,
                "status": r.status,
                "detail": r.detail,
                "elapsed_s": round(r.elapsed_s, 4),
                "timestamp": r.timestamp,
            }
            for r in tracker.results
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"  [JSON] Report saved: {output_path}")


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _esc(s: str) -> str:
    """HTML-escape a string."""
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


def generate_all_reports(
    tracker: ResultTracker,
    output_dir: Path,
) -> None:
    """Generate TXT, HTML, and JSON reports in *output_dir*."""
    print()
    generate_txt_report(tracker, output_dir / "Phase1_Validation_Report.txt")
    generate_html_report(tracker, output_dir / "Phase1_Validation_Report.html")
    generate_json_report(tracker, output_dir / "Phase1_Validation_Report.json")
    print()
