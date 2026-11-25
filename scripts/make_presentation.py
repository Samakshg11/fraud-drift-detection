"""Build the group presentation from the verified results in results/.

Run from the repository root:

    python scripts/make_presentation.py

Output: docs/project_presentation.pptx (16:9, with speaker notes).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "results" / "figures"
OUTPUT = ROOT / "docs" / "project_presentation.pptx"

_ADD_NOTES = True

NAVY = RGBColor(23, 42, 69)
BLUE = RGBColor(44, 108, 176)
RED = RGBColor(196, 78, 82)
GREY = RGBColor(80, 90, 100)


def add_text(
    slide,
    text: str,
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = NAVY,
    align=None,
) -> None:
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.text = text
    paragraph.font.size = Pt(size)
    paragraph.font.bold = bold
    paragraph.font.color.rgb = color
    if align is not None:
        paragraph.alignment = align


def add_bullets(
    slide,
    lines: list[str],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    size: int = 17,
) -> None:
    box = slide.shapes.add_textbox(
        Inches(left), Inches(top), Inches(width), Inches(height)
    )
    frame = box.text_frame
    frame.word_wrap = True
    for i, line in enumerate(lines):
        paragraph = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        paragraph.text = f"• {line}"
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = NAVY
        paragraph.space_after = Pt(8)


def add_header(slide, title: str, subtitle: str | None = None) -> None:
    add_text(slide, title, 0.6, 0.35, 12.1, 0.7, size=30, bold=True, color=NAVY)
    if subtitle:
        add_text(slide, subtitle, 0.6, 1.0, 12.1, 0.45, size=16, color=BLUE)
    bar = slide.shapes.add_shape(
        1, Inches(0.6), Inches(1.42), Inches(12.1), Inches(0.04)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()


def add_table(
    slide,
    headers: list[str],
    rows: list[list[str]],
    left: float,
    top: float,
    width: float,
    height: float,
    *,
    font_size: int = 14,
) -> None:
    table = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(left), Inches(top), Inches(width), Inches(height)
    ).table
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = header
        for p in cell.text_frame.paragraphs:
            p.font.bold = True
            p.font.size = Pt(font_size)
            p.font.color.rgb = RGBColor(255, 255, 255)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
    for i, row in enumerate(rows, start=1):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = str(value)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size)
                p.font.color.rgb = NAVY
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = RGBColor(232, 239, 247)


def add_image_fit(slide, path: Path, left: float, top: float, width: float, height: float) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing result figure: {path}")
    image_width, image_height = Image.open(path).size
    box_w, box_h = Inches(width), Inches(height)
    scale = min(box_w / image_width, box_h / image_height)
    w, h = int(image_width * scale), int(image_height * scale)
    slide.shapes.add_picture(
        str(path),
        int(left * 914400 + (box_w - w) / 2),
        int(top * 914400 + (box_h - h) / 2),
        width=w,
        height=h,
    )


def notes(slide, text: str) -> None:
    if _ADD_NOTES:
        slide.notes_slide.notes_text_frame.text = text


def main() -> None:
    global _ADD_NOTES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-notes", action="store_true", help="Omit speaker notes")
    parser.add_argument("--output", type=str, default=str(OUTPUT))
    args = parser.parse_args()
    _ADD_NOTES = not args.no_notes

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    # 1 — Title
    slide = prs.slides.add_slide(blank)
    add_text(
        slide,
        "Leakage-Free and Drift-Aware\nCredit Card Fraud Detection",
        0.9, 1.6, 11.5, 2.0, size=38, bold=True,
    )
    add_text(
        slide,
        "Machine Learning on Long-Horizon Transaction Data",
        0.9, 3.7, 11.5, 0.7, size=22, color=BLUE,
    )
    add_text(
        slide,
        "Group project (6 roles)  |  Role 2: Leakage-Free Pipeline Lead\n"
        "Dataset used for results: Sparkov, 1.85 M transactions, 24 months\n"
        "November 2026",
        0.9, 4.8, 11.5, 1.4, size=17, color=GREY,
    )
    notes(slide, "Open by stating the problem: fraud papers over-report accuracy because of leakage, and short datasets cannot show drift. Our project fixes both and measures the damage.")

    # 2 — Problem
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Problem & Motivation", "Why most reported fraud-detection scores cannot be trusted")
    add_bullets(
        slide,
        [
            "Random train/test splits mix future transactions into training — a classic data leak.",
            "Imputers, scalers, encoders and SMOTE are often fitted on ALL data before splitting.",
            "Target/frequency encoders computed globally let a row's own label leak into its features.",
            "Past-only aggregates are frequently computed with future rows included.",
            "Many studies use only ~48 hours of data, so long-term drift is invisible.",
            "Fraud labels arrive late (chargebacks) — the split needs an embargo gap.",
            "Accuracy is useless at 0.5% fraud: we use PR-AUC and recall at a fixed FPR.",
        ],
        0.8, 1.8, 11.8, 5.0,
    )
    notes(slide, "Emphasise that these are not theoretical complaints: we reproduce them in a deliberately leaky baseline and measure the inflation (slide 6).")

    # 3 — Objectives (explicitly required)
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Three Objectives", "All three are implemented; results shown on Sparkov")
    add_table(
        slide,
        ["#", "Objective", "Status"],
        [
            ["1", "Build a leakage-free, time-aware evaluation pipeline on long-horizon data and quantify metric inflation versus a deliberately leaky baseline.", "Implemented + results"],
            ["2", "Measure month-by-month degradation of a static fraud model using drift diagnostics (PSI, KS, ADWIN / DDM / Page-Hinkley).", "Implemented + results"],
            ["3", "Compare drift-aware retraining strategies (static, periodic, sliding window, drift-triggered, recency-weighted) under one walk-forward protocol.", "Implemented + results"],
        ],
        0.7, 1.9, 12.0, 4.4, font_size=15,
    )
    notes(slide, "These are the exact three objectives. The invigilator asked for at least 1.5 done — we have implementation and real evidence for all three, with the caveat that only one dataset was available.")

    # 4 — Architecture
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Project Architecture", "Six roles, one chronological evaluation backbone")
    add_table(
        slide,
        ["Stage", "Module / Owner", "What it does"],
        [
            ["Raw → interim", "Role 1: src/data", "Load CSVs, drop index artifact, timestamp + month, downcast, save parquet"],
            ["Chronological splits", "Role 2: src/pipeline/splits.py", "Month windows, walk-forward, embargo, temporal ordering assertion"],
            ["Train-only state", "Role 2: encoders + preprocessing", "Past-only target/frequency encoding; impute/scale/one-hot fitted per fold"],
            ["Past-only features", "Role 3: src/features", "Rolling 24h/7D aggregates, velocity, cyclic time features"],
            ["Models + metrics", "Role 4: src/models, src/evaluation", "LogReg/RF/XGBoost factories; PR-AUC, recall@FPR, cost, base rate"],
            ["Drift + retraining", "Roles 5–6: src/drift, src/retraining", "PSI/KS/detectors, degradation curve, five retraining strategies"],
        ],
        0.6, 1.75, 12.1, 4.7, font_size=13,
    )
    notes(slide, "Only Role 2 is fully owned by me; the other modules are reference implementations that teammates may replace. The interfaces are stable so that works.")

    # 5 — Role 2 detail
    slide = prs.slides.add_slide(blank)
    add_header(slide, "My Work — Role 2: Leakage-Free Pipeline", "The part I can defend line by line")
    add_bullets(
        slide,
        [
            "Month index = year×12 + month−1; December→January is correct.",
            "walk_forward_splits(): expanding or sliding, non-overlapping test blocks, 21 folds on Sparkov.",
            "Every split asserts max(train_time) < min(test_time); gap_days adds a chargeback embargo.",
            "TimeAwareTargetEncoder: expanding, strictly-earlier values for training rows; frozen train-period mapping for test rows.",
            "Same-timestamp rows never encode each other; unseen test categories fall back to the prior.",
            "ColumnTransformer (imputer / scaler / one-hot) fitted on each training fold only.",
            "leaky_baseline.py is quarantined with warnings — used only to measure inflation.",
            "25 dedicated leakage tests; 55 tests pass in total.",
        ],
        0.8, 1.8, 11.8, 5.0, size=16,
    )
    notes(slide, "Key viva point: at scoring time we do not know earlier test labels, so updating encoders on test data would itself be leakage. That is why fit_transform and transform deliberately differ.")

    # 6 — Objective 1 results
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Objective 1 — Leakage Inflation (Real Result)", "Sparkov, logistic regression, 21 walk-forward folds")
    add_table(
        slide,
        ["Setup", "PR-AUC", "Recall @ 1% FPR", "F1", "Base rate"],
        [
            ["Leaky (random split, all-data fits)", "0.2275", "0.5611", "0.3235", "0.52%"],
            ["Leakage-free (walk-forward)", "0.0061", "0.0328", "0.0214", "0.48%"],
        ],
        0.7, 1.75, 12.0, 1.3, font_size=15,
    )
    add_image_fit(slide, FIGURES / "leakage_comparison_sparkov.png", 3.1, 3.3, 7.1, 3.6)
    add_text(slide, "≈37× inflation in PR-AUC from leakage alone", 0.9, 6.85, 11.5, 0.5, size=18, bold=True, color=RED)
    notes(slide, "This is the headline result. The leaky setup fits encoders and preprocessing on all rows and uses a random split. The leakage-free setup never sees future information. Same model, same features — only the protocol differs.")

    # 7 — Objective 2 results
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Objective 2 — Model Degradation Over Months", "One static model trained once, scored on every following month")
    add_image_fit(slide, FIGURES / "degradation_sparkov.png", 0.7, 1.75, 8.5, 4.9)
    add_bullets(
        slide,
        [
            "PR-AUC: 0.0105 → 0.0045 over 21 months (peak 0.0145).",
            "Numeric PSI max 0.00056 → features look stable.",
            "KS max statistic 0.0076 → tiny effect despite significance.",
            "Conclusion: concept drift, not marginal feature drift.",
        ],
        9.3, 2.0, 3.5, 4.5, size=14,
    )
    notes(slide, "Static training is not enough on a 2-year horizon. Note the caveat: ADWIN/DDM produced many alerts (DDM 922), which is mostly noise at a 0.5% fraud rate; trigger thresholds need tuning.")

    # 8 — Objective 3 results
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Objective 3 — Retraining Strategies", "Same 21 walk-forward folds for every strategy")
    add_table(
        slide,
        ["Strategy", "PR-AUC", "Recall@1%FPR", "Cost", "Train time"],
        [
            ["Sliding window", "0.0116", "0.0149", "4305", "63 s"],
            ["Static", "0.0102", "0.0103", "4361", "2 s"],
            ["Recency-weighted", "0.0083", "0.0368", "4410", "323 s"],
            ["Periodic (monthly)", "0.0064", "0.0308", "4368", "351 s"],
            ["Drift-triggered", "0.0064", "0.0308", "4368", "348 s"],
        ],
        0.7, 1.7, 7.2, 3.6, font_size=13,
    )
    add_image_fit(slide, FIGURES / "retraining_pr_auc_sparkov.png", 8.1, 1.9, 4.6, 3.6)
    add_bullets(
        slide,
        [
            "Sliding window = best PR-AUC at moderate cost.",
            "Static = cheapest by ~30×, slightly lower PR-AUC.",
            "Weighted = best recall at 1% FPR.",
        ],
        0.8, 5.45, 11.8, 1.4, size=15,
    )
    notes(slide, "Drift-triggered retrained every fold because the detector kept firing; that is a tuning problem, not evidence that drift triggers are useless. Be transparent about it.")

    # 9 — Role contributions
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Team Contributions", "Each member explains their own module")
    add_table(
        slide,
        ["Role", "Owner", "Deliverable"],
        [
            ["1 — Data Engineer", "__________", "Loading, cleaning, parquet, EDA; prepare_data.py"],
            ["2 — Pipeline Lead (me)", "__________", "Splits, encoders, preprocessing, leaky baseline, leakage tests"],
            ["3 — Features", "__________", "Past-only aggregates, time features, leakage checks"],
            ["4 — Modeling", "__________", "Model factories, metrics, plots, tuning"],
            ["5 — Drift", "__________", "PSI, KS, ADWIN/DDM/PH, degradation curve"],
            ["6 — Retraining + report", "__________", "Strategies, cost tracking, scripts, paper"],
        ],
        0.7, 1.85, 12.0, 4.6, font_size=15,
    )
    notes(slide, "Fill in the owner names before presenting. Roles 1, 3, 4, 5 and 6 are reference implementations in this repo and their owners may replace them.")

    # 10 — Limitations
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Limitations & Future Work", "Say this before the examiner asks")
    add_bullets(
        slide,
        [
            "IEEE-CIS dataset unavailable (Kaggle verification failed) — results are from one dataset only.",
            "All reported runs use Logistic Regression; XGBoost/RF were too slow on 8 GB RAM.",
            "gap_days = 0; real chargeback-lag embargo still to be set.",
            "Drift-trigger alerts are noisy at 0.5% fraud; thresholds need calibration.",
            "Literature review (2 papers per member) is still pending.",
            "Next: second dataset, tree models, walk-forward hyperparameter tuning, final paper.",
        ],
        0.8, 1.8, 11.8, 5.0,
    )
    notes(slide, "Being explicit about limitations builds credibility. The contribution is the methodology and the measured inflation, not a state-of-the-art score.")

    # 11 — Conclusion
    slide = prs.slides.add_slide(blank)
    add_header(slide, "Conclusion")
    add_bullets(
        slide,
        [
            "Built a fully leakage-free, time-aware evaluation pipeline (Role 2) with 55 passing tests.",
            "Quantified leakage inflation: PR-AUC 0.2275 → 0.0061 (≈37×) on 1.85 M transactions.",
            "Showed a static fraud model degrades across 21 months (0.0105 → 0.0045 PR-AUC).",
            "Compared five retraining strategies; sliding window gives the best PR-AUC/cost trade-off.",
            "Reproducible: config-driven, seeded, scripts + results + experiment log in the repo.",
        ],
        0.8, 1.8, 11.8, 4.6,
    )
    add_text(slide, "Thank you — questions?", 0.9, 6.4, 11.5, 0.7, size=24, bold=True, color=BLUE)
    notes(slide, "Close on the contribution: a defensible evaluation protocol for fraud detection on long horizons, plus a measured demonstration of how badly leakage distorts results.")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(output)
    print(f"Created {output} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
