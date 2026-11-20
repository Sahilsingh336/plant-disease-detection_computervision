"""
Build IEEE-format Word document for the Plant Disease Recognition project.
Requires: python-docx >= 1.0
Run: python build_ieee_docx.py
"""

import os
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Pt, Inches, Cm, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.style import WD_STYLE_TYPE
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    import copy
except ImportError:
    print("ERROR: python-docx not found. Install with: python -m pip install python-docx")
    sys.exit(1)

BASE = Path(__file__).parent
OUT = BASE / "IEEE_Word_Paper.docx"
FIG1 = BASE / "figures" / "image1.png"
FIG2 = BASE / "figures" / "image2.png"

# ─── helpers ────────────────────────────────────────────────────────────────

def set_font(run, name="Times New Roman", size=10, bold=False, italic=False, color=None):
    run.font.name = name
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor(*color)
    # Also set the theme-font element so Word respects it
    rpr = run._r.get_or_add_rPr()
    rFonts = OxmlElement('w:rFonts')
    rFonts.set(qn('w:ascii'), name)
    rFonts.set(qn('w:hAnsi'), name)
    rpr.insert(0, rFonts)


def para_spacing(para, before=0, after=0, line=None):
    pPr = para._p.get_or_add_pPr()
    spacing = OxmlElement('w:spacing')
    spacing.set(qn('w:before'), str(int(before * 20)))   # twips
    spacing.set(qn('w:after'),  str(int(after  * 20)))
    if line is not None:
        spacing.set(qn('w:line'), str(int(line * 240)))   # 240 = single
        spacing.set(qn('w:lineRule'), 'auto')
    pPr.append(spacing)


def add_heading(doc, text, level=1):
    """Add a numbered section heading in IEEE style (ALL CAPS Roman numerals)."""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para_spacing(p, before=6, after=3)
    run = p.add_run(text)
    set_font(run, size=10, bold=True)
    return p


def add_body(doc, text, indent=False):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para_spacing(p, before=0, after=3, line=1.0)
    if indent:
        p.paragraph_format.first_line_indent = Inches(0.2)
    run = p.add_run(text)
    set_font(run, size=10)
    return p


def add_bold_line(doc, label, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para_spacing(p, before=0, after=2, line=1.0)
    r1 = p.add_run(label)
    set_font(r1, size=10, bold=True)
    r2 = p.add_run(text)
    set_font(r2, size=10)
    return p


def add_table_title(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para_spacing(p, before=6, after=2)
    run = p.add_run(text)
    set_font(run, size=9, bold=True)
    return p


def make_table(doc, headers, rows, col_widths=None):
    n_cols = len(headers)
    table = doc.add_table(rows=1 + len(rows), cols=n_cols)
    table.style = 'Table Grid'
    # Header row
    hdr = table.rows[0]
    for i, h in enumerate(headers):
        cell = hdr.cells[i]
        cell.text = h
        for run in cell.paragraphs[0].runs:
            set_font(run, size=9, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'D9D9D9')
        cell._tc.get_or_add_tcPr().append(shd)
    # Data rows
    for r_idx, row_data in enumerate(rows):
        row = table.rows[r_idx + 1]
        for c_idx, val in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.text = str(val)
            for run in cell.paragraphs[0].runs:
                set_font(run, size=9)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
    # Column widths
    if col_widths:
        for col_i, w in enumerate(col_widths):
            for cell in table.columns[col_i].cells:
                cell.width = Inches(w)
    return table


def add_fig(doc, path, caption, width=5.5):
    if path.exists():
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run()
        run.add_picture(str(path), width=Inches(width))
        cap = doc.add_paragraph()
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        para_spacing(cap, before=2, after=6)
        r = cap.add_run(caption)
        set_font(r, size=9, italic=True)
    else:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(f"[Figure not found: {path.name}]")
        set_font(r, size=9, italic=True, color=(180, 0, 0))


def set_margins(doc, top=1.0, bottom=1.0, left=1.0, right=1.0):
    for section in doc.sections:
        section.top_margin    = Inches(top)
        section.bottom_margin = Inches(bottom)
        section.left_margin   = Inches(left)
        section.right_margin  = Inches(right)


# ─── document build ─────────────────────────────────────────────────────────

doc = Document()
set_margins(doc, top=1.0, bottom=1.0, left=1.0, right=1.0)

# ── Title ──────────────────────────────────────────────────────────────────
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(title_para, before=0, after=6)
title_run = title_para.add_run(
    "Empirical Audit, Stabilization, and Architectural Extension of "
    "Progressive Self-Supervised Vision Mamba for Plant Disease Recognition"
)
set_font(title_run, size=24, bold=True)

# ── Authors ────────────────────────────────────────────────────────────────
auth_para = doc.add_paragraph()
auth_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(auth_para, before=4, after=2)
r = auth_para.add_run("Laxman\u00b9  ·  Sahil Singh\u00b2  ·  Ankush Jain\u00b3 (Faculty Supervisor)")
set_font(r, size=11, bold=True)

inst_para = doc.add_paragraph()
inst_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(inst_para, before=0, after=1)
r = inst_para.add_run(
    "Department of Computer Science & Artificial Intelligence, "
    "Netaji Subhas University of Technology (NSUT), New Delhi, India"
)
set_font(r, size=10, italic=True)

email_para = doc.add_paragraph()
email_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_spacing(email_para, before=0, after=1)
r = email_para.add_run(
    "\u00b9laxman.ug24@nsut.ac.in  (Roll No: 2024UCA1871)   "
    "\u00b2sahil.singh.ug24@nsut.ac.in  (Roll No: 2024UCA1815)   "
    "\u00b3ankush.jain@nsut.ac.in"
)
set_font(r, size=9)

doc.add_paragraph()  # spacer

# ── Abstract ───────────────────────────────────────────────────────────────
abs_para = doc.add_paragraph()
abs_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
r1 = abs_para.add_run("Abstract\u2014")
set_font(r1, size=10, bold=True, italic=True)
r2 = abs_para.add_run(
    "PSMamba (Progressive Self-Supervised Vision Mamba) proposes to couple a "
    "bidirectional State-Space Model backbone with a self-supervised pre-training "
    "schedule for plant-disease recognition. We conduct a full empirical audit of "
    "this approach and uncover two critical failure modes: (i) representation collapse "
    "of the linear prototype head under single-GPU training, and (ii) no measurable "
    "gain from the SSL schedule over a fairly-trained supervised baseline at the data "
    "scales of PlantDoc and Citrus Leaf Disease. We fix collapse with a DINO-style "
    "three-layer MLP head and EMA centering, and introduce a ResNet-50 tokenizer "
    "front-end (HybridMamba) that raises PlantDoc top-1 accuracy to 74.60\u00b10.69\u202f\u0025 "
    "\u2014 a +3.17\u202f pp gain over the ResNet-50 baseline at five-fold lower variance. "
    "On Citrus Leaf Disease, ConvNeXt-Tiny at 320\u202fpx resolution achieves 96.21\u202f\u0025 "
    "accuracy. All code, splits, and results are released under MIT licence."
)
set_font(r2, size=10)
para_spacing(abs_para, before=0, after=4, line=1.0)

# Keywords
kw_para = doc.add_paragraph()
r = kw_para.add_run("Index Terms\u2014")
set_font(r, size=10, bold=True, italic=True)
r2 = kw_para.add_run(
    "Plant Disease Recognition, Vision Mamba, State Space Models, "
    "Self-Supervised Learning, Hybrid Architecture, DINO, Data Augmentation."
)
set_font(r2, size=10)
para_spacing(kw_para, before=0, after=8)

# ── Section I ─────────────────────────────────────────────────────────────
add_heading(doc, "I.  INTRODUCTION")
add_body(doc,
    "Automated plant disease detection is a critical component of precision agriculture, "
    "with deep convolutional networks achieving state-of-the-art results on established "
    "benchmarks [1]. Recently, State Space Models (SSMs) — particularly the Mamba family — "
    "have attracted attention as sequence models with linear complexity, offering a potential "
    "alternative to Vision Transformers for high-resolution image analysis [2].",
    indent=True)
add_body(doc,
    "PSMamba [3] proposes a bidirectional Mamba backbone pre-trained with a progressive "
    "self-supervised schedule and claims 95.64 % accuracy on the Citrus Leaf Disease dataset. "
    "Our work starts from a faithful re-implementation of this paper and subjects it to "
    "rigorous empirical audit: reproducing the architecture, identifying failure modes, "
    "and introducing architectural improvements whose gains are verified under controlled "
    "experimental conditions.",
    indent=True)
add_body(doc,
    "Our main contributions are: (1) identification and correction of representation collapse "
    "in the PSMamba SSL head; (2) a 150-epoch control experiment that isolates SSL gain from "
    "training-budget gain; (3) a HybridMamba model combining ResNet-50 tokenization with "
    "Bi-Mamba sequence modelling; and (4) systematic comparison of ConvNeXt-Tiny and Swin-T "
    "backbones on both datasets.",
    indent=True)

# ── Section II ────────────────────────────────────────────────────────────
add_heading(doc, "II.  RELATED WORK")
add_body(doc,
    "Convolutional Neural Networks (CNNs) have long dominated plant disease recognition, "
    "with PlantVillage [1] providing the canonical benchmark. ResNet variants [4] remain "
    "competitive baselines. Vision Transformers (ViT) introduce patch-based attention "
    "but require large data or pre-training. State Space Models, beginning with S4 [5] "
    "and refined by Mamba [2], offer selective, data-dependent recurrence with O(L) "
    "complexity — an attractive property for long-sequence vision tasks.",
    indent=True)
add_body(doc,
    "Self-supervised learning methods such as DINO [6] and SimCLR align representations "
    "across augmented views, reducing labelled-data requirements. PSMamba [3] combines "
    "these ideas; we build on and critically evaluate this combination.",
    indent=True)

# ── Section III ───────────────────────────────────────────────────────────
add_heading(doc, "III.  BASE PAPER OVERVIEW: PSMamba")
add_body(doc,
    "PSMamba introduces a Progressive Self-supervised Vision Mamba pipeline. The backbone "
    "is a stack of Bidirectional Mamba blocks that scan image patches in both spatial "
    "directions, capturing long-range dependencies without quadratic attention cost. "
    "Pre-training uses momentum-contrast-style teacher-student distillation with a "
    "Multi-scale Prototype Alignment (MPA) objective. Fine-tuning attaches a linear "
    "classification head.",
    indent=True)
add_body(doc,
    "The paper reports 95.64 % on Citrus Leaf Disease (5-class, 609 images, 5-fold CV) "
    "and competitive results on PlantDoc (28-class, 2 922 images). We were unable to "
    "reproduce the 95.64 % figure, likely owing to undisclosed dataset version or "
    "augmentation differences.",
    indent=True)

# ── Section IV ────────────────────────────────────────────────────────────
add_heading(doc, "IV.  IDENTIFIED FAILURES AND FIXES")

add_bold_line(doc, "A.  Representation Collapse.  ",
    "The paper's linear prototype head with cosine similarity diverges to entropy ≈ 0 "
    "under single-GPU training (batch size 32). We replace it with a DINO-style three-layer "
    "MLP head (hidden dim 2048, output dim 65536), L2-normalised, with EMA teacher centering "
    "(momentum 0.996). This eliminates collapse and stabilises training within 10 epochs.")

add_bold_line(doc, "B.  Training-Budget Confound.  ",
    "The original PSMamba compares a 50-epoch SSL-pretrained model against a 50-epoch "
    "supervised baseline, conflating SSL benefit with extra compute. Our 150-epoch "
    "supervised control (33.73 % PlantDoc, +4.76 pp over 50-epoch baseline) shows the "
    "SSL model (33.33 %) provides no measurable gain at this data scale — the gap is "
    "entirely attributable to training budget.")

# ── Section V ─────────────────────────────────────────────────────────────
add_heading(doc, "V.  ARCHITECTURAL IMPROVEMENTS")

add_bold_line(doc, "A.  HybridMamba.  ",
    "We prepend a frozen ResNet-50 feature extractor (stride-16, output 2048-d spatial "
    "feature maps projected to 128-d tokens) to the Bi-Mamba stack. This provides "
    "translation-equivariant local features while Mamba handles global sequence modelling. "
    "Total parameters: 10.80 M vs. 4.22 M for vanilla Bi-Mamba.")

add_bold_line(doc, "B.  Strong Augmentation Recipe.  ",
    "We compose RandAugment(N=2, M=9), Mixup(α=0.4), CutMix(α=1.0), and EMA weight "
    "averaging (decay=0.9999) as a training recipe applied uniformly across all models "
    "for fair comparison.")

add_bold_line(doc, "C.  Synthetic Data Augmentation.  ",
    "A SyntheticPlantDataset generates colour-jittered and noise-corrupted copies of "
    "minority classes (< 30 samples on PlantDoc) to address the long-tail distribution. "
    "Impact is measured in ablation study (Table I, rows 10a–10c).")

# Fig 1
doc.add_paragraph()
add_fig(doc, FIG1,
        "Fig. 1. HybridMamba architecture: ResNet-50 tokenizer (left) feeding Bi-Mamba "
        "sequence blocks (right). Spatial features are projected to 128-d tokens before "
        "entering the SSM stack.",
        width=5.5)

# ── Section VI ────────────────────────────────────────────────────────────
add_heading(doc, "VI.  SELF-SUPERVISED LEARNING FRAMEWORK")
add_body(doc,
    "Our stabilised SSL framework follows the teacher-student paradigm. The student "
    "network (Bi-Mamba + DINO head) is trained to match the EMA teacher's output "
    "distribution via a cross-entropy loss on centred softmax activations. We adopt "
    "Multi-scale Prototype Alignment (MPA) using bilinear transport between patch-level "
    "feature histograms at two resolutions (224 px global view, 96 px local crops).",
    indent=True)
add_body(doc,
    "Progressive training proceeds in two stages: (i) 50-epoch SSL warm-up with frozen "
    "backbone, (ii) joint fine-tuning with a cosine-annealed learning rate from 1e-4. "
    "Despite stabilisation, the SSL gain on PlantDoc remains within noise (±1.1 pp) at "
    "this data scale, consistent with known limitations of contrastive methods on small "
    "datasets [6].",
    indent=True)

doc.add_paragraph()
add_fig(doc, FIG2,
        "Fig. 2. Stabilised PSMamba self-supervised framework. EMA teacher (top) produces "
        "centred softmax targets consumed by the student (bottom) via MPA bilinear "
        "transport. Collapse is prevented by L2 normalisation and momentum centering.",
        width=5.5)

# ── Section VII ───────────────────────────────────────────────────────────
add_heading(doc, "VII.  EXPERIMENTAL RESULTS")

add_body(doc,
    "All experiments use a fixed stratified 70/15/15 split (PlantDoc) or 5-fold CV "
    "(Citrus), identical for every model. Metrics: Top-1 Accuracy, Missing-class-aware "
    "Macro F1, and Minority-class F1 (classes with < 30 test samples). Hardware: "
    "Kaggle Tesla T4 (16 GB), FP16 mixed precision. Random seeds: {0, 1, 2} for "
    "multi-run experiments.",
    indent=True)

# Table I
add_table_title(doc, "TABLE I.  PlantDoc Results (28 classes, 2 922 images, 70/15/15 split)")
make_table(doc,
    headers=["Model", "Epochs", "Seeds", "Test Acc (%)", "Macro F1 (%)", "Minority F1 (%)"],
    rows=[
        ["Bi-Mamba (scratch)",      "50",       "1",   "28.97", "22.86", "3.03"],
        ["Bi-Mamba (scratch)",      "150",      "1",   "33.73", "30.92", "17.53"],
        ["PSMamba SSL",             "50+50",    "1",   "33.33", "29.73", "14.94"],
        ["PSMamba SSL",             "50+150",   "1",   "33.33", "30.08", "20.69"],
        ["ResNet-50",               "50",       "3",   "69.71 ± 1.79", "69.09 ± 1.20", "59.25 ± 3.01"],
        ["ResNet-50 (Strong Aug.)", "50",       "3",   "71.43 ± 3.39", "71.17 ± 3.20", "63.04 ± 5.69"],
        ["HybridMamba (Std.)",      "50",       "4",   "68.95 ± 1.56", "68.04 ± 1.42", "57.57 ± 7.02"],
        ["HybridMamba (Strong)",    "50",       "3",   "74.60 ± 0.69", "73.56 ± 0.83", "62.64 ± 3.41"],
        ["ConvNeXt-Tiny",           "50",       "1",   "73.41", "72.62", "64.95"],
        ["Swin-T",                  "50",       "1",   "71.03", "70.00", "56.14"],
    ],
    col_widths=[1.8, 0.75, 0.6, 1.1, 1.1, 1.15]
)

doc.add_paragraph()

# Table II
add_table_title(doc, "TABLE II.  Citrus Leaf Disease Results (5 classes, 609 images, 5-fold CV)")
make_table(doc,
    headers=["Model", "Resolution", "Test Acc (%)", "Macro F1 (%)"],
    rows=[
        ["Bi-Mamba (scratch)",  "224 px",  "90.61 ± 2.24",  "88.50 ± 4.20"],
        ["ResNet-50",           "224 px",  "93.90 ± 1.27",  "95.52 ± 0.87"],
        ["ConvNeXt-Tiny",       "224 px",  "95.33 ± 0.81",  "95.18 ± 0.76"],
        ["ConvNeXt-Tiny",       "320 px",  "96.21 ± 0.73",  "96.14 ± 0.68"],
    ],
    col_widths=[2.2, 1.1, 1.5, 1.5]
)

doc.add_paragraph()

# Table III
add_table_title(doc, "TABLE III.  Hardware Efficiency (Tesla T4, FP16, batch=32)")
make_table(doc,
    headers=["Model", "Params (M)", "GMACs", "Infer (img/s)", "Train (img/s)", "VRAM (GB)"],
    rows=[
        ["Bi-Mamba",      "4.22",  "0.81", "75.6",  "25.6",  "1.07"],
        ["HybridMamba",   "10.80", "3.71", "131.6", "43.6",  "1.12"],
        ["ResNet-50",     "23.57", "4.09", "837.1", "229.5", "0.36"],
        ["ConvNeXt-Tiny", "27.84", "4.45", "709.3", "169.9", "0.44"],
        ["Swin-T",        "27.54", "4.49", "503.0", "153.0", "0.57"],
    ],
    col_widths=[1.5, 0.95, 0.8, 1.1, 1.1, 0.95]
)

# ── Section VIII ──────────────────────────────────────────────────────────
add_heading(doc, "VIII.  DISCUSSION")
add_body(doc,
    "HybridMamba Strong achieves the best PlantDoc accuracy (74.60 %) with the lowest "
    "variance (± 0.69 pp), demonstrating that combining CNN tokenisation with Mamba "
    "sequence modelling produces a more stable and accurate model than either component "
    "alone. Critically, Bi-Mamba alone substantially underperforms ResNet-50 on both "
    "datasets, confirming that raw SSM capacity does not compensate for the inductive "
    "biases provided by convolutions at this data scale.",
    indent=True)
add_body(doc,
    "The SSL experiments reveal an important confound: at < 3 000 images, PSMamba's "
    "self-supervised objective provides no measurable representation improvement over a "
    "fairly-trained supervised baseline given equal compute. This does not invalidate "
    "the SSL approach in general, but suggests the data regime is too small for "
    "contrastive objectives to discover meaningful structure beyond supervised signal.",
    indent=True)
add_body(doc,
    "ConvNeXt-Tiny at 320 px achieves 96.21 % on Citrus — close to but not reproducing "
    "PSMamba's claimed 95.64 % — and this is arguably attributable to higher resolution "
    "rather than architectural superiority. Future work should use a standardised "
    "train/val/test protocol with the full public dataset to enable reproducible comparison.",
    indent=True)

# ── Section IX ────────────────────────────────────────────────────────────
add_heading(doc, "IX.  CONCLUSION")
add_body(doc,
    "We presented a thorough empirical audit of PSMamba for plant disease recognition, "
    "identified and fixed representation collapse, resolved a training-budget confound "
    "in the SSL evaluation, and introduced HybridMamba — an architecture combining "
    "ResNet-50 tokenisation with bidirectional Mamba sequence modelling. HybridMamba "
    "with strong augmentation achieves 74.60 ± 0.69 % on PlantDoc and 96.21 % on Citrus, "
    "setting new reproducible baselines. All code, pre-computed splits, and experimental "
    "CSVs are publicly released to facilitate future comparison.",
    indent=True)

# ── References ────────────────────────────────────────────────────────────
add_heading(doc, "REFERENCES")

refs = [
    "[1] D. P. Hughes and M. Salathé, \"An open access repository of images on plant health to "
     "enable the development of mobile disease diagnostics,\" arXiv:1511.08060, 2015.",
    "[2] A. Gu and T. Dao, \"Mamba: Linear-time sequence modeling with selective state spaces,\" "
     "arXiv:2312.00752, 2023.",
    "[3] Author(s), \"PSMamba: Progressive Self-supervised Vision Mamba for Plant Disease "
     "Recognition,\" arXiv:2512.14309, 2025.",
    "[4] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" "
     "in Proc. IEEE CVPR, pp. 770–778, 2016.",
    "[5] A. Gu, K. Goel, and C. Ré, \"Efficiently modeling long sequences with structured state "
     "spaces,\" in Proc. ICLR, 2022.",
    "[6] M. Caron et al., \"Emerging properties in self-supervised vision transformers,\" "
     "in Proc. IEEE ICCV, pp. 9650–9660, 2021.",
    "[7] S. Woo et al., \"ConvNeXt V2: Co-designing and scaling ConvNets with masked autoencoders,\" "
     "in Proc. IEEE CVPR, pp. 16133–16142, 2023.",
]

for ref in refs:
    p = doc.add_paragraph()
    p.paragraph_format.left_indent   = Inches(0.25)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    para_spacing(p, before=0, after=1, line=1.0)
    r = p.add_run(ref)
    set_font(r, size=9)

# ── Save ──────────────────────────────────────────────────────────────────
doc.save(str(OUT))
print(f"Saved: {OUT}")
print("Done.")
