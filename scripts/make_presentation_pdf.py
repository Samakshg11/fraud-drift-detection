"""Generate a PDF version of the presentation (opens in any PDF viewer).

    python scripts/make_presentation_pdf.py

Output: docs/project_presentation.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "docs" / "project_presentation.pdf"
PAGE = (13.333 * inch, 7.5 * inch)

NAVY = colors.HexColor("#172a45")
BLUE = colors.HexColor("#2c6cb0")
RED = colors.HexColor("#c44e52")
LIGHT = colors.HexColor("#e8eff7")

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("Title2", parent=styles["Title"], fontSize=28, textColor=NAVY,
                       spaceAfter=10, leading=34)
SUBTITLE = ParagraphStyle("Sub", parent=styles["Normal"], fontSize=13, textColor=BLUE,
                          spaceAfter=14)
BODY = ParagraphStyle("Body2", parent=styles["Normal"], fontSize=14, textColor=NAVY,
                      leading=21, spaceAfter=8)
SMALL = ParagraphStyle("Small2", parent=styles["Normal"], fontSize=11, textColor=NAVY,
                       leading=15)
CELL = ParagraphStyle("Cell", parent=styles["Normal"], fontSize=11, textColor=NAVY,
                      leading=14)
CELL_BOLD = ParagraphStyle("CellBold", parent=CELL, textColor=colors.white,
                           fontName="Helvetica-Bold")
HEADLINE = ParagraphStyle("Headline", parent=BODY, fontSize=17, textColor=RED,
                          fontName="Helvetica-Bold", alignment=TA_CENTER)


def heading(title: str, subtitle: str | None = None) -> list:
    flow = [Paragraph(title, TITLE)]
    if subtitle:
        flow.append(Paragraph(subtitle, SUBTITLE))
    flow.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=12))
    return flow


def bullets(lines: list[str], style: ParagraphStyle = BODY) -> list:
    return [Paragraph(f"- {line}", style) for line in lines]


def table(headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
    data = [[Paragraph(h, CELL_BOLD) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(v), CELL) for v in row])
    t = Table(data, colWidths=[w * inch for w in widths], hAlign="LEFT")
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b8c4d0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    for i in range(2, len(data), 2):
        style.append(("BACKGROUND", (0, i), (-1, i), LIGHT))
    t.setStyle(TableStyle(style))
    return t


def image(path: Path, max_w: float, max_h: float) -> Image:
    if not path.is_file():
        raise FileNotFoundError(path)
    iw, ih = PILImage.open(path).size
    scale = min(max_w * inch / iw, max_h * inch / ih)
    return Image(str(path), width=iw * scale, height=ih * scale)


def page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawRightString(PAGE[0] - 0.6 * inch, 0.25 * inch, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    story = []

    # 1
    story += [Spacer(1, 1.2 * inch),
              Paragraph("Leakage-Free and Drift-Aware<br/>Credit Card Fraud Detection", TITLE),
              Paragraph("Machine Learning on Long-Horizon Transaction Data", SUBTITLE),
              Spacer(1, 0.4 * inch)]
    story += bullets([
        "Group project (6 roles) — Role 2: Leakage-Free Pipeline Lead",
        "Dataset for results: Sparkov, 1.85 M transactions, 24 months",
        "November 2026",
    ])
    story.append(PageBreak())

    # 2
    story += heading("Problem & Motivation", "Why most reported fraud-detection scores cannot be trusted")
    story += bullets([
        "Random splits mix future transactions into training — a classic data leak.",
        "Imputers, scalers, encoders and SMOTE are often fitted on ALL data before splitting.",
        "Target/frequency encoders computed globally let a row's own label leak into its features.",
        "Past-only aggregates are frequently computed with future rows included.",
        "Many studies use only ~48 hours of data, so long-term drift is invisible.",
        "Fraud labels arrive late (chargebacks) — the split needs an embargo gap.",
        "Accuracy is useless at 0.5% fraud: we use PR-AUC and recall at a fixed FPR.",
    ])
    story.append(PageBreak())

    # 3
    story += heading("Three Objectives", "All three objectives are DONE - results shown on Sparkov")
    story.append(table(
        ["#", "Objective", "Status"],
        [
            ["1", "Build a leakage-free, time-aware evaluation pipeline on long-horizon data and quantify metric inflation vs a deliberately leaky baseline.", "DONE - results on Sparkov"],
            ["2", "Measure month-by-month degradation of a static model using drift diagnostics (PSI, KS, ADWIN/DDM/Page-Hinkley).", "DONE - results on Sparkov"],
            ["3", "Compare drift-aware retraining strategies (static, periodic, sliding, drift-triggered, weighted) under one walk-forward protocol.", "DONE - results on Sparkov"],
        ], [0.5, 9.6, 2.0],
    ))
    story.append(PageBreak())

    # 4
    story += heading("Project Architecture", "Six roles, one chronological evaluation backbone")
    story.append(table(
        ["Stage", "Module / Owner", "What it does"],
        [
            ["Raw → interim", "Role 1: src/data", "Load CSVs, drop index artifact, timestamp + month, downcast, parquet"],
            ["Splits", "Role 2: splits.py", "Month windows, walk-forward, embargo, ordering assertion"],
            ["Train-only state", "Role 2: encoders + preprocessing", "Past-only encoders; impute/scale/one-hot fitted per fold"],
            ["Features", "Role 3: src/features", "Rolling 24h/7D aggregates, velocity, cyclic time features"],
            ["Models + metrics", "Role 4: src/models, evaluation", "Factories; PR-AUC, recall@FPR, cost, base rate"],
            ["Drift + retraining", "Roles 5–6", "PSI/KS/detectors, degradation, five retraining strategies"],
        ], [2.0, 3.2, 6.9],
    ))
    story.append(PageBreak())

    # 5
    story += heading("My Work — Role 2", "The part I can defend line by line")
    story += bullets([
        "Month index = year×12 + month−1; December→January is correct.",
        "walk_forward_splits(): expanding or sliding, non-overlapping test blocks, 21 folds.",
        "Every split asserts max(train_time) < min(test_time); gap_days adds an embargo.",
        "Target/frequency encoders: expanding past-only values for train, frozen mapping for test.",
        "Same-timestamp rows never encode each other; unseen categories fall back to the prior.",
        "ColumnTransformer fitted on each training fold only.",
        "leaky_baseline.py is quarantined — used only to measure inflation.",
        "25 dedicated leakage tests; 55 tests pass in total.",
    ], SMALL)
    story.append(PageBreak())

    # 6
    story += heading("Objective 1 — Leakage Inflation (Real Result)",
                     "Sparkov, logistic regression, 21 walk-forward folds")
    story.append(table(
        ["Setup", "PR-AUC", "Recall @ 1% FPR", "F1", "Base rate"],
        [
            ["Leaky (random split, all-data fits)", "0.2275", "0.5611", "0.3235", "0.52%"],
            ["Leakage-free (walk-forward)", "0.0061", "0.0328", "0.0214", "0.48%"],
        ], [4.3, 1.7, 2.3, 1.4, 1.6],
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(image(FIGURES / "leakage_comparison_sparkov.png", 8.5, 3.6))
    story.append(Paragraph("≈37× inflation in PR-AUC from leakage alone", HEADLINE))
    story.append(PageBreak())

    # 7
    story += heading("Objective 2 — Model Degradation Over Months",
                     "One static model trained once, scored on every following month")
    story.append(image(FIGURES / "degradation_sparkov.png", 9.0, 4.5))
    story += bullets([
        "PR-AUC: 0.0105 → 0.0045 over 21 months (peak 0.0145).",
        "Numeric PSI max 0.00056 → features look stable.",
        "Conclusion: concept drift, not marginal feature drift.",
    ], SMALL)
    story.append(PageBreak())

    # 8
    story += heading("Objective 3 — Retraining Strategies",
                     "Same 21 walk-forward folds for every strategy")
    story.append(table(
        ["Strategy", "PR-AUC", "Recall@1%FPR", "Cost", "Train time"],
        [
            ["Sliding window", "0.0116", "0.0149", "4305", "63 s"],
            ["Static", "0.0102", "0.0103", "4361", "2 s"],
            ["Recency-weighted", "0.0083", "0.0368", "4410", "323 s"],
            ["Periodic", "0.0064", "0.0308", "4368", "351 s"],
            ["Drift-triggered", "0.0064", "0.0308", "4368", "348 s"],
        ], [3.0, 1.7, 2.3, 1.9, 2.0],
    ))
    story.append(Spacer(1, 0.15 * inch))
    story.append(image(FIGURES / "retraining_pr_auc_sparkov.png", 8.0, 3.0))
    story.append(PageBreak())

    # 9
    story += heading("Team Contributions", "Each member explains their own module")
    story.append(table(
        ["Role", "Owner", "Deliverable"],
        [
            ["1 — Data Engineer", "__________", "Loading, cleaning, parquet, EDA"],
            ["2 — Pipeline Lead (me)", "__________", "Splits, encoders, preprocessing, leaky baseline, tests"],
            ["3 — Features", "__________", "Past-only aggregates, time features, leakage checks"],
            ["4 — Modeling", "__________", "Model factories, metrics, plots"],
            ["5 — Drift", "__________", "PSI, KS, ADWIN/DDM/PH, degradation curve"],
            ["6 — Retraining + report", "__________", "Strategies, cost tracking, scripts, paper"],
        ], [3.2, 2.5, 6.4],
    ))
    story.append(PageBreak())

    # 10
    story += heading("Limitations & Future Work", "Say this before the examiner asks")
    story += bullets([
        "IEEE-CIS unavailable (Kaggle verification) — results from one dataset only.",
        "All runs use Logistic Regression; XGBoost/RF too slow on 8 GB RAM.",
        "gap_days = 0; real chargeback-lag embargo still to be set.",
        "Drift-trigger alerts are noisy at 0.5% fraud; thresholds need calibration.",
        "Literature review (2 papers per member) still pending.",
        "Next: second dataset, tree models, walk-forward tuning, final paper.",
    ], SMALL)
    story.append(PageBreak())

    # 11
    story += heading("Conclusion")
    story += bullets([
        "Built a fully leakage-free, time-aware pipeline with 55 passing tests.",
        "Quantified leakage inflation: PR-AUC 0.2275 → 0.0061 (≈37×) on 1.85 M transactions.",
        "Showed a static model degrades across 21 months (0.0105 → 0.0045 PR-AUC).",
        "Compared five retraining strategies; sliding window gives the best PR-AUC/cost trade-off.",
        "Reproducible: config-driven, seeded, scripts + results + experiment log in the repo.",
    ])
    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph("Thank you — questions?", SUBTITLE))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT), pagesize=PAGE,
        leftMargin=0.6 * inch, rightMargin=0.6 * inch,
        topMargin=0.5 * inch, bottomMargin=0.4 * inch,
        title="Leakage-Free and Drift-Aware Credit Card Fraud Detection",
    )
    doc.build(story, onFirstPage=page_number, onLaterPages=page_number)
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    build()
