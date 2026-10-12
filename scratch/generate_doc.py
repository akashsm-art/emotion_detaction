"""
Comprehensive Word Document (.docx) Generator for Multi-Modal Emotion Recognition System (ERM).
Writes 'Multi_Modal_Emotion_Recognition_System_Documentation.docx' into the workspace root.
"""

import os
import sys
from datetime import datetime

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

# --- Color Constants ---
COLOR_PRIMARY = RGBColor(0x1E, 0x3A, 0x8A)      # Royal Navy
COLOR_SECONDARY = RGBColor(0x0D, 0x94, 0x88)    # Deep Teal
COLOR_DARK_TEXT = RGBColor(0x1E, 0x29, 0x3B)    # Slate 800
COLOR_MUTED_TEXT = RGBColor(0x64, 0x74, 0x8B)   # Slate 500
COLOR_ALERT_WARN = RGBColor(0xB4, 0x53, 0x09)   # Amber 700
COLOR_ALERT_INFO = RGBColor(0x1D, 0x4E, 0xD8)   # Blue 700
COLOR_ALERT_SUCCESS = RGBColor(0x15, 0x80, 0x3D)# Green 700

HEX_PRIMARY = "1E3A8A"
HEX_SECONDARY = "0D9488"
HEX_LIGHT_BG = "F8FAFC"
HEX_CODE_BG = "F1F5F9"
HEX_BORDER = "CBD5E1"
HEX_HEADER_BG = "1E3A8A"
HEX_ZEBRA_BG = "F8FAFC"

def set_cell_shading(cell, color_hex):
    """Set background fill color for a table cell."""
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def set_cell_margins(cell, top=120, bottom=120, left=180, right=180):
    """Set padding (in dxa) for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = parse_xml(f'''
        <w:tcMar {nsdecls("w")}>
            <w:top w:w="{top}" w:type="dxa"/>
            <w:bottom w:w="{bottom}" w:type="dxa"/>
            <w:left w:w="{left}" w:type="dxa"/>
            <w:right w:w="{right}" w:type="dxa"/>
        </w:tcMar>
    ''')
    tcPr.append(tcMar)

def set_cell_borders(cell, top="CBD5E1", bottom="CBD5E1", left="none", right="none", sz="4"):
    """Set custom borders for a table cell."""
    tcPr = cell._tc.get_or_add_tcPr()
    top_xml = f'w:val="single" w:sz="{sz}" w:space="0" w:color="{top}"' if top != "none" else 'w:val="none"'
    bottom_xml = f'w:val="single" w:sz="{sz}" w:space="0" w:color="{bottom}"' if bottom != "none" else 'w:val="none"'
    left_xml = f'w:val="single" w:sz="{sz}" w:space="0" w:color="{left}"' if left != "none" else 'w:val="none"'
    right_xml = f'w:val="single" w:sz="{sz}" w:space="0" w:color="{right}"' if right != "none" else 'w:val="none"'

    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top {top_xml}/>
            <w:left {left_xml}/>
            <w:bottom {bottom_xml}/>
            <w:right {right_xml}/>
        </w:tcBorders>
    ''')
    tcPr.append(tcBorders)

def add_header_footer(doc):
    """Configure running headers and footers with page numbering."""
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

        # Header
        header = section.header
        p_hdr = header.paragraphs[0]
        p_hdr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_hdr = p_hdr.add_run("Multi-Modal Emotion Recognition System (ERM) — Technical Manual")
        r_hdr.font.name = "Arial"
        r_hdr.font.size = Pt(8.5)
        r_hdr.font.color.rgb = COLOR_MUTED_TEXT

        # Footer
        footer = section.footer
        p_ftr = footer.paragraphs[0]
        p_ftr.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r_ftr_label = p_ftr.add_run("Page ")
        r_ftr_label.font.name = "Arial"
        r_ftr_label.font.size = Pt(9.0)
        r_ftr_label.font.color.rgb = COLOR_MUTED_TEXT

        fldChar1 = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
        instrText = parse_xml(r'<w:instrText %s xml:space="preserve"> PAGE </w:instrText>' % nsdecls('w'))
        fldChar2 = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
        fldChar3 = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
        p_ftr._p.append(fldChar1)
        p_ftr._p.append(instrText)
        p_ftr._p.append(fldChar2)
        p_ftr._p.append(fldChar3)

        r_of = p_ftr.add_run(" of ")
        r_of.font.name = "Arial"
        r_of.font.size = Pt(9.0)
        r_of.font.color.rgb = COLOR_MUTED_TEXT

        fldChar1_n = parse_xml(r'<w:fldChar %s w:fldCharType="begin"/>' % nsdecls('w'))
        instrText_n = parse_xml(r'<w:instrText %s xml:space="preserve"> NUMPAGES </w:instrText>' % nsdecls('w'))
        fldChar2_n = parse_xml(r'<w:fldChar %s w:fldCharType="separate"/>' % nsdecls('w'))
        fldChar3_n = parse_xml(r'<w:fldChar %s w:fldCharType="end"/>' % nsdecls('w'))
        p_ftr._p.append(fldChar1_n)
        p_ftr._p.append(instrText_n)
        p_ftr._p.append(fldChar2_n)
        p_ftr._p.append(fldChar3_n)

def add_heading_1(doc, text):
    """Add a styled Heading 1."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(20)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(16.5)
    r.bold = True
    r.font.color.rgb = COLOR_PRIMARY
    return p

def add_heading_2(doc, text):
    """Add a styled Heading 2."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(14)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(13.0)
    r.bold = True
    r.font.color.rgb = COLOR_SECONDARY
    return p

def add_heading_3(doc, text):
    """Add a styled Heading 3."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(11.0)
    r.bold = True
    r.font.color.rgb = COLOR_DARK_TEXT
    return p

def add_body_paragraph(doc, text, bold_prefix="", space_after=5):
    """Add a regular formatted body paragraph."""
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(10)
        r_pre.bold = True
        r_pre.font.color.rgb = COLOR_DARK_TEXT
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK_TEXT
    return p

def add_bullet_point(doc, text, bold_prefix=""):
    """Add a styled bullet point."""
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.15
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = "Arial"
        r_pre.font.size = Pt(10)
        r_pre.bold = True
        r_pre.font.color.rgb = COLOR_DARK_TEXT
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(10)
    r.font.color.rgb = COLOR_DARK_TEXT
    return p

def add_callout(doc, text, title="NOTE", alert_type="info"):
    """Add a professionally styled callout box with accent left border."""
    border_color = HEX_PRIMARY if alert_type == "info" else ("B45309" if alert_type == "warn" else "15803D")
    title_color = COLOR_ALERT_INFO if alert_type == "info" else (COLOR_ALERT_WARN if alert_type == "warn" else COLOR_ALERT_SUCCESS)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, HEX_LIGHT_BG)
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="36" w:space="0" w:color="{border_color}"/>
            <w:top w:val="none"/>
            <w:bottom w:val="none"/>
            <w:right w:val="none"/>
        </w:tcBorders>
    ''')
    cell._tc.get_or_add_tcPr().append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15

    r_title = p.add_run(f"[{title}] ")
    r_title.bold = True
    r_title.font.name = "Arial"
    r_title.font.size = Pt(9.5)
    r_title.font.color.rgb = title_color

    r_text = p.add_run(text)
    r_text.font.name = "Arial"
    r_text.font.size = Pt(9.5)
    r_text.font.color.rgb = COLOR_DARK_TEXT

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_code_block(doc, code_str, title=None):
    """Add a styled code block with Consolas font and subtle border."""
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    set_cell_shading(cell, HEX_CODE_BG)
    set_cell_margins(cell, top=100, bottom=100, left=160, right=160)

    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:left w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
            <w:top w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
            <w:bottom w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
            <w:right w:val="single" w:sz="6" w:space="0" w:color="{HEX_BORDER}"/>
        </w:tcBorders>
    ''')
    cell._tc.get_or_add_tcPr().append(tcBorders)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15

    if title:
        r_head = p.add_run(f"# {title}\n")
        r_head.font.name = "Consolas"
        r_head.font.size = Pt(8.5)
        r_head.bold = True
        r_head.font.color.rgb = COLOR_PRIMARY

    r = p.add_run(code_str)
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(4)

def add_styled_table(doc, headers, rows, col_widths=None):
    """Create a professionally styled data table with headers and zebra striping."""
    table = doc.add_table(rows=len(rows) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    hdr_cells = table.rows[0].cells
    for i, title in enumerate(headers):
        cell = hdr_cells[i]
        set_cell_shading(cell, HEX_HEADER_BG)
        set_cell_margins(cell, top=140, bottom=140, left=160, right=160)
        set_cell_borders(cell, top="1E3A8A", bottom="1E3A8A", left="none", right="none", sz="6")
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        r = p.add_run(title)
        r.bold = True
        r.font.name = "Arial"
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    trPr = table.rows[0]._tr.get_or_add_trPr()
    trPr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

    for r_idx, row_data in enumerate(rows):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = HEX_ZEBRA_BG if r_idx % 2 == 1 else "FFFFFF"
        trPr = table.rows[r_idx + 1]._tr.get_or_add_trPr()
        trPr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))

        for c_idx, val in enumerate(row_data):
            cell = row_cells[c_idx]
            set_cell_shading(cell, bg_color)
            set_cell_margins(cell, top=100, bottom=100, left=160, right=160)
            set_cell_borders(cell, top=HEX_BORDER, bottom=HEX_BORDER, left="none", right="none", sz="4")
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(1)
            r = p.add_run(str(val))
            r.font.name = "Arial"
            r.font.size = Pt(9.0)
            r.font.color.rgb = COLOR_DARK_TEXT

    if col_widths and len(col_widths) == len(headers):
        for row in table.rows:
            for idx, width in enumerate(col_widths):
                row.cells[idx].width = Inches(width)

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(0)
    p_spacer.paragraph_format.space_after = Pt(5)
    return table

def build_cover_page(doc):
    """Construct an executive cover page."""
    p_top = doc.add_paragraph()
    p_top.paragraph_format.space_before = Pt(40)
    p_top.paragraph_format.space_after = Pt(10)

    r_tag = p_top.add_run("SYSTEM ARCHITECTURE & TECHNICAL MANUAL")
    r_tag.font.name = "Arial"
    r_tag.font.size = Pt(11)
    r_tag.font.bold = True
    r_tag.font.color.rgb = COLOR_SECONDARY

    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(6)
    p_title.paragraph_format.space_after = Pt(12)
    p_title.paragraph_format.line_spacing = 1.1
    r_title = p_title.add_run("Multi-Modal Emotion\nRecognition System (ERM)")
    r_title.font.name = "Arial"
    r_title.font.size = Pt(30)
    r_title.font.bold = True
    r_title.font.color.rgb = COLOR_PRIMARY

    p_sub = doc.add_paragraph()
    p_sub.paragraph_format.space_before = Pt(0)
    p_sub.paragraph_format.space_after = Pt(25)
    r_sub = p_sub.add_run("A Local-Only, Production-Grade Affective Computing Platform Integrating RoBERTa NLP, 2D CNN Speech Acoustics, AlexNet Facial Vision, Confidence-Aware Late Decision Fusion, FastAPI, and React 19")
    r_sub.font.name = "Arial"
    r_sub.font.size = Pt(12.5)
    r_sub.font.color.rgb = COLOR_MUTED_TEXT

    # Decorative divider
    table_div = doc.add_table(rows=1, cols=1)
    table_div.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell_div = table_div.cell(0, 0)
    cell_div.width = Inches(6.5)
    set_cell_shading(cell_div, HEX_SECONDARY)
    tcBorders = parse_xml(f'''
        <w:tcBorders {nsdecls("w")}>
            <w:top w:val="none"/><w:left w:val="none"/><w:bottom w:val="none"/><w:right w:val="none"/>
        </w:tcBorders>
    ''')
    cell_div._tc.get_or_add_tcPr().append(tcBorders)
    p_div = cell_div.paragraphs[0]
    p_div.paragraph_format.space_before = Pt(1)
    p_div.paragraph_format.space_after = Pt(1)

    p_mid = doc.add_paragraph()
    p_mid.paragraph_format.space_before = Pt(30)

    meta_headers = ["Attribute", "Specification / Value"]
    meta_rows = [
        ["Project Identifier", "Multi-Modal Emotion Recognition Platform (ERM)"],
        ["System Version", "1.0.0 (Production Release)"],
        ["Design Principle", "Zero Fake AI Policy — 100% Genuine Neural Forward Passes"],
        ["Inference Topology", "100% Localhost Edge Inference (Zero External Telemetry / Cloud Calls)"],
        ["Modality Models", "Text: RoBERTa | Speech: 2D CNN (Emo-DB) | Vision: AlexNet (RAVDESS)"],
        ["Decision Fusion", "Confidence-Aware Weighted Late Decision Fusion with Concordance Analysis"],
        ["Backend Stack", "Python 3.14 / 3.10+, FastAPI, Uvicorn, PyTorch 2.4, Torchvision, Transformers"],
        ["Frontend Stack", "React 19, TypeScript, Vite, Web Audio API, Vanilla CSS Glassmorphism"],
        ["Test Coverage", "21 / 21 PyTest Integration Tests Passing | 16 / 16 Vitest Tests Passing"],
        ["Publication Date", datetime.now().strftime("%B %d, %Y")],
    ]
    add_styled_table(doc, meta_headers, meta_rows, col_widths=[2.1, 4.4])
    doc.add_page_break()

def build_table_of_contents(doc):
    """Generate structured Table of Contents."""
    add_heading_1(doc, "Table of Contents")

    toc_items = [
        ("1. Executive Summary & Zero Fake AI System Philosophy", "3"),
        ("2. End-to-End System Architecture & Component Topology", "4"),
        ("3. Canonical Emotion Ontology & Circumplex Representation", "6"),
        ("4. Modality Engine 1: Natural Language Processing (Text Engine)", "8"),
        ("5. Modality Engine 2: Acoustic Speech Processing (Audio Engine)", "11"),
        ("6. Modality Engine 3: Facial Vision Emotion Recognition (Vision Engine)", "14"),
        ("7. Vision Model Training Methodology & Empirical Evaluation", "17"),
        ("8. Multi-Modal Decision Fusion Engine", "21"),
        ("9. FastAPI Backend Architecture & API Specification", "25"),
        ("10. Frontend Application Architecture (React 19 + TypeScript + Vite)", "29"),
        ("11. Standalone Real-Time Native Programs (CLI / GUI)", "33"),
        ("12. Automated Testing, Verification, & Quality Assurance Suite", "36"),
        ("13. Security, Privacy, & Local Data Sanitization Architecture", "38"),
        ("14. Installation, Deployment, & Operations Manual", "40"),
        ("15. Legacy Codebases & Evolutionary Refactoring Journey", "43"),
        ("16. System Limitations, Ethical Boundaries, & Future Roadmap", "45"),
    ]

    toc_table = doc.add_table(rows=len(toc_items), cols=2)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, (title, page) in enumerate(toc_items):
        cell_t = toc_table.cell(i, 0)
        cell_p = toc_table.cell(i, 1)
        cell_t.width = Inches(5.6)
        cell_p.width = Inches(0.9)

        set_cell_margins(cell_t, top=60, bottom=60, left=80, right=80)
        set_cell_margins(cell_p, top=60, bottom=60, left=80, right=80)
        set_cell_borders(cell_t, top="none", bottom=HEX_BORDER, left="none", right="none", sz="2")
        set_cell_borders(cell_p, top="none", bottom=HEX_BORDER, left="none", right="none", sz="2")

        p_t = cell_t.paragraphs[0]
        p_t.paragraph_format.space_before = Pt(1)
        p_t.paragraph_format.space_after = Pt(1)
        r_t = p_t.add_run(title)
        r_t.font.name = "Arial"
        r_t.font.size = Pt(9.5)
        r_t.font.color.rgb = COLOR_DARK_TEXT
        if i in (0, 1, 2, 6, 7, 8, 9):
            r_t.bold = True

        p_p = cell_p.paragraphs[0]
        p_p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_p.paragraph_format.space_before = Pt(1)
        p_p.paragraph_format.space_after = Pt(1)
        r_p = p_p.add_run(page)
        r_p.font.name = "Arial"
        r_p.font.size = Pt(9.5)
        r_p.font.color.rgb = COLOR_MUTED_TEXT

    p_spacer = doc.add_paragraph()
    p_spacer.paragraph_format.space_before = Pt(10)
    doc.add_page_break()

def build_section_1_executive_summary(doc):
    """Section 1: Executive Summary & System Philosophy."""
    add_heading_1(doc, "1. Executive Summary & Zero Fake AI System Philosophy")

    add_body_paragraph(
        doc,
        "Human emotion expression is intrinsically multi-modal, non-synchronous, and highly context-dependent. "
        "A person communicating emotional distress may articulate neutral vocabulary while displaying subtle acoustic tremors in pitch "
        "or micro-expressions of sadness across facial landmarks. Conversely, sarcasm, social politeness, and masking frequently introduce "
        "deep discordance between verbal statements and visual affect. The Multi-Modal Emotion Recognition System (ERM) is an industrial-grade, "
        "production-tested affective computing platform engineered to solve these challenges through genuine deep learning and mathematically "
        "principled decision fusion."
    )

    add_heading_2(doc, "1.1 The Core Operating Principle: Strict Zero Fake AI")
    add_body_paragraph(
        doc,
        "In modern machine learning applications, a common anti-pattern is 'demo-ware'—systems that mock unavailable models with pseudo-random "
        "number generators, hard-coded heuristics, or simulated probability vectors. The ERM platform enforces a non-negotiable Zero Fake AI Policy:"
    )

    add_bullet_point(
        doc,
        "Every prediction returned by the Text, Speech, or Vision subsystems is computed exclusively through a real forward pass "
        "of a trained deep neural network.",
        bold_prefix="100% Genuine Forward Passes: "
    )
    add_bullet_point(
        doc,
        "If a modality's trained model checkpoint is missing, corrupted, or unsupported, the platform transparently marks the modality "
        "as unavailable (available=False) with an explicit status message. It strictly refuses to fabricate substitute probabilities.",
        bold_prefix="Transparent Failure Signaling: "
    )
    add_bullet_point(
        doc,
        "In the speech model, which is trained on four emotional categories, unsupported classes (disgust, fear, surprise) are strictly "
        "allocated 0.0000 probability mass. No artificial uniform distribution or synthetic smoothing is injected.",
        bold_prefix="Zero Artificial Mass Allocation: "
    )
    add_bullet_point(
        doc,
        "Neural Softmax outputs represent model epistemic confidence across trained patterns. The system never equates confidence to "
        "infallible ground truth, psychological diagnosis, or lie detection.",
        bold_prefix="Confidence as Weighting Prior, Not Absolute Truth: "
    )

    add_callout(
        doc,
        "Zero Fake AI is not merely an architectural guideline; it is codified as programmatic assertions across the API validation suite "
        "and unit tests. Modalities lacking checkpoints are dynamically excluded from fusion calculations rather than faked.",
        title="CORE ETHICAL COMMITMENT",
        alert_type="info"
    )

    add_heading_2(doc, "1.2 Platform Summary Matrix")
    mod_headers = ["Modality", "Model Architecture", "Underlying Checkpoint", "Operational Status", "Supported Emotion Classes"]
    mod_rows = [
        ["Text (NLP)", "RoBERTa Transformer (DistilRoBERTa-base)", "HuggingFace Hub cached (~300 MB)", "Operational (100%)", "All 7 Ekman Classes (angry, disgust, fear, happy, neutral, sad, surprise)"],
        ["Speech (Audio)", "4-Layer 2D CNN on 39 MFCCs", "models/speech/best_model_CNN.h5 (296 KB)", "Operational (100%)", "4 Classes: neutral, angry, happy, sad (disgust, fear, surprise assigned 0.0)"],
        ["Vision (Facial)", "AlexNet 7-Class Deep CNN + Haar Cascades", "models/vision/model_image_new50.pth (217.57 MB)", "Operational (100%)", "All 7 Ekman Classes (reproduced on RAVDESS disjoint actor split)"],
        ["Decision Fusion", "Confidence-Aware Weighted Late Fusion Engine", "Deterministic Algorithmic Math (core/fusion/engine.py)", "Operational (100%)", "All 7 Modality Permutations with Concordance & JS-Divergence"],
    ]
    add_styled_table(doc, mod_headers, mod_rows, col_widths=[1.1, 1.6, 1.6, 1.0, 1.2])

def build_section_2_system_architecture(doc):
    """Section 2: End-to-End System Architecture & Component Topology."""
    add_heading_1(doc, "2. End-to-End System Architecture & Component Topology")

    add_body_paragraph(
        doc,
        "The ERM platform is decoupled into four modular layers: an Ingestion & Sanitization Layer, a Single-Modality Deep Neural Network Ensemble, "
        "a Centralized Emotion Ontology Normalizer, a Confidence-Aware Late Decision Fusion Engine, and an Interactive Presentation Layer. "
        "The entire platform operates completely offline on local hardware with zero external dependencies."
    )

    add_heading_2(doc, "2.1 Architectural Flow Diagram")
    arch_ascii = (
        "+---------------------------------------------------------------------------------------+\n"
        "|                                     INPUT MEDIA                                       |\n"
        "|                 Natural Language Text  |  Acoustic Audio  |  Facial Images             |\n"
        "+-------------------------------------------+-------------------------------------------+\n"
        "                                            |\n"
        "            +-------------------------------+-------------------------------+\n"
        "            |                               |                               |\n"
        "            v                               v                               v\n"
        "  +--------------------+          +--------------------+          +--------------------+\n"
        "  |    Text Engine     |          |    Audio Engine    |          |   Vision Engine    |\n"
        "  |  DistilRoBERTa     |          |  2D CNN on MFCCs   |          |  AlexNet + Haar    |\n"
        "  |  (HuggingFace)     |          |  (Emo-DB Checkpt)  |          |  (RAVDESS Checkpt) |\n"
        "  +---------+----------+          +---------+----------+          +---------+----------+\n"
        "            |                               |                               |\n"
        "            +-------------------------------+-------------------------------+\n"
        "                                            |\n"
        "                                            v\n"
        "                            +-------------------------------+\n"
        "                            | Centralized Emotion Ontology  |\n"
        "                            | Canonical Ekman 7-Class Map   |\n"
        "                            | Zero-Mass Invariant Gating    |\n"
        "                            +---------------+---------------+\n"
        "                                            |\n"
        "                                            v\n"
        "                            +-------------------------------+\n"
        "                            | Confidence-Aware Late Fusion  |\n"
        "                            | - Modality Availability Gate  |\n"
        "                            | - Dynamic Weight Adjustment   |\n"
        "                            | - Cosine Concordance / Conflict|\n"
        "                            | - Explainability & Rationale  |\n"
        "                            +---------------+---------------+\n"
        "                                            |\n"
        "                                            v\n"
        "                            +-------------------------------+\n"
        "                            |      FusedEmotionResult       |\n"
        "                            | - Dominant Fused Emotion      |\n"
        "                            | - Overall Neural Confidence   |\n"
        "                            | - 2D Circumplex (Val/Aro)     |\n"
        "                            | - Modality Attribution Vector |\n"
        "                            +-------------------------------+\n"
        "                                            |\n"
        "            +-------------------------------+-------------------------------+\n"
        "            |                                                               |\n"
        "            v                                                               v\n"
        "  +-----------------------------------+           +-----------------------------------+\n"
        "  |    FastAPI Backend REST Service   |           |    React 19 + TypeScript Web UI   |\n"
        "  |    Uvicorn ASGI @ 127.0.0.1:8000  |           |    Vite Dev Server @ :5173        |\n"
        "  +-----------------------------------+           +-----------------------------------+\n"
        "  |    Native Python Live Streams     |           |    Native Desktop Tools           |\n"
        "  |    (live_text, live_audio, etc.)  |           |    (OpenCV HUD, SoundDevice Mic)  |\n"
        "  +-----------------------------------+           +-----------------------------------+"
    )
    add_code_block(doc, arch_ascii, title="SYSTEM TOPOLOGY & DATA FLOW PIPELINE")

    add_heading_2(doc, "2.2 Directory Hierarchy and Codebase Organization")
    add_body_paragraph(
        doc,
        "The repository follows a clean, decoupled folder structure that isolates machine learning model wrappers, backend API handlers, "
        "training pipelines, frontend components, and tests:"
    )

    tree_headers = ["Directory / File", "Type", "Core Responsibility & Description"]
    tree_rows = [
        ["api/", "Package", "FastAPI application entrypoint, dependency singletons, path redaction, and route handlers."],
        ["api/routes/", "Subpackage", "Modular HTTP route controllers for health, text, audio, vision, and multimodal prediction."],
        ["core/", "Package", "Core mathematical algorithms, ontology mapping, Pydantic schemas, and model wrappers."],
        ["core/ontology.py", "Module", "Canonical Ekman 7-emotion schema, Russell's Circumplex coordinates, and mapping fidelity."],
        ["core/schemas.py", "Module", "Strict Pydantic data contracts (ModalityResult, FusedEmotionResult, ConflictAnalysis, etc.)."],
        ["core/text/", "Subpackage", "DistilRoBERTa sequence classification wrapper and text preprocessor."],
        ["core/speech/", "Subpackage", "PyTorch 2D CNN acoustic model and 39-channel MFCC feature extractor."],
        ["core/vision/", "Subpackage", "AlexNet classifier wrapper and multi-pass OpenCV Haar Cascade face detector."],
        ["core/fusion/engine.py", "Module", "Confidence-aware late decision fusion engine with concordance and conflict analysis."],
        ["frontend/", "Project", "React 19 + TypeScript + Vite web dashboard with glassmorphism and live audio visualizer."],
        ["models/speech/", "Checkpoint", "Trained Emo-DB CNN speech checkpoint (best_model_CNN.h5, 296 KB)."],
        ["models/vision/", "Checkpoint", "Trained RAVDESS AlexNet vision checkpoint (model_image_new50.pth, 217.57 MB)."],
        ["training/vision/", "Subpackage", "Isolated training pipeline reproducing AlexNet on RAVDESS disjoint actor split."],
        ["tests/", "Test Suite", "PyTest integration test suite covering API security, late fusion math, and edge cases."],
        ["live_face.py", "Script", "Standalone real-time OpenCV webcam application with decoupled 30 FPS display / 3.5 FPS inference."],
        ["live_audio.py", "Script", "Standalone real-time microphone stream classifier using SoundDevice and circular buffering."],
        ["live_text.py", "Script", "Interactive terminal REPL for real-time text emotion classification with ASCII probability meters."],
        ["requirements.txt", "Config", "Consolidated, conflict-free Python dependency manifest."],
    ]
    add_styled_table(doc, tree_headers, tree_rows, col_widths=[1.5, 0.9, 4.1])

def build_section_3_emotion_ontology(doc):
    """Section 3: Canonical Emotion Ontology & Circumplex Representation."""
    add_heading_1(doc, "3. Canonical Emotion Ontology & Circumplex Representation")

    add_body_paragraph(
        doc,
        "A central challenge in multi-modal affective computing is semantic heterogeneity: distinct datasets and modality models operate "
        "on conflicting label vocabularies. For instance, the German Emo-DB speech dataset uses German emotional terms ('Wut', 'Langeweile'), "
        "RAVDESS video clips use 2-digit numerical codes, social media NLP models use Twitter sentiment categories ('love', 'anticipation'), "
        "and facial expression models often collapse calm into neutral. The ERM system unifies all modalities onto the universal Ekman 7-emotion schema."
    )

    add_heading_2(doc, "3.1 The Canonical Ekman 7-Class Schema")
    add_body_paragraph(
        doc,
        "All individual modality probability distributions and fused results are mapped strictly to the canonical 7 core emotions:"
    )
    add_code_block(doc, 'CORE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]', title="CANONICAL 7 CORE EMOTIONS")

    add_heading_2(doc, "3.2 Russell's Circumplex Model of Affect (Valence & Arousal)")
    add_body_paragraph(
        doc,
        "Categorical emotion labels are augmented by continuous dimensional coordinates based on James Russell's Circumplex Model of Affect (1980). "
        "Each emotional state is positioned within a 2D Cartesian plane defined by:"
    )
    add_bullet_point(doc, "Valence (Horizontal Axis, -1.0 to +1.0): Represents the hedonic positivity or negativity of the emotional experience.", bold_prefix="Valence (V): ")
    add_bullet_point(doc, "Arousal (Vertical Axis, -1.0 to +1.0): Represents the physiological activation, intensity, or energy level.", bold_prefix="Arousal (A): ")

    circ_headers = ["Canonical Emotion", "Valence (V)", "Arousal (A)", "Circumplex Quadrant", "Psychological Profile"]
    circ_rows = [
        ["happy", "+0.85", "+0.60", "Top-Right (High V, High A)", "Pleasurable physiological activation, joy, excitement, contentment."],
        ["surprise", "+0.25", "+0.85", "Top-Right (Mild V, Peak A)", "Heightened alertness, sudden sensory intake, novel environmental stimuli."],
        ["neutral", " 0.00", " 0.00", "Origin (Equilibrium)", "Baseline state, balanced cognitive equilibrium, absence of acute affect."],
        ["disgust", "-0.60", "+0.35", "Top-Left (Low V, Mild A)", "Aversive rejection, sensory revulsion, moral or physical distaste."],
        ["angry", "-0.65", "+0.80", "Top-Left (Low V, Peak A)", "High-energy defensive/offensive stance, frustration, acute hostility."],
        ["fear", "-0.70", "+0.75", "Top-Left (Lowest V, High A)", "Acute threat perception, fight-or-flight activation, perceived vulnerability."],
        ["sad", "-0.80", "-0.45", "Bottom-Left (Lowest V, Low A)", "De-activation, somatic depression, perceived loss, social withdrawal."],
    ]
    add_styled_table(doc, circ_headers, circ_rows, col_widths=[1.2, 0.9, 0.9, 1.8, 1.7])

    add_body_paragraph(
        doc,
        "Aggregate circumplex coordinates for any modality or fused result are mathematically calculated as the expectation value over the "
        "entire probability distribution vector P:"
    )
    add_code_block(
        doc,
        "V_aggregate = sum(P[emotion] * EMOTION_VALENCE_AROUSAL[emotion]['valence'] for emotion in CORE_EMOTIONS)\n"
        "A_aggregate = sum(P[emotion] * EMOTION_VALENCE_AROUSAL[emotion]['arousal'] for emotion in CORE_EMOTIONS)",
        title="CIRCUMPLEX EXPECTATION EQUATIONS"
    )

    add_heading_2(doc, "3.3 Label Mapping Fidelity Tracking")
    add_body_paragraph(
        doc,
        "To preserve scientific rigor, every label transformation tracks its MappingType fidelity:"
    )
    add_bullet_point(doc, "DIRECT: 100% semantic identity (e.g., 'joy' -> 'happy', 'anger' -> 'angry', 'Wut' -> 'angry').", bold_prefix="DIRECT: ")
    add_bullet_point(doc, "APPROXIMATE: Close emotional alignment with minor nuance shift (e.g., 'calm' -> 'neutral', 'love' -> 'happy', 'optimism' -> 'happy').", bold_prefix="APPROXIMATE: ")
    add_bullet_point(doc, "LOSSY: Broader emotional generalization (e.g., 'trust' -> 'neutral', unmapped fallback).", bold_prefix="LOSSY: ")
    add_bullet_point(doc, "UNSUPPORTED: Explicitly flagged when a modality model lacks architectural support for a category.", bold_prefix="UNSUPPORTED: ")

    add_heading_2(doc, "3.4 Zero-Mass Probability Normalization Algorithm")
    add_body_paragraph(
        doc,
        "When normalizing raw probabilities to the 7-class ontology, unsupported emotions are strictly clamped to 0.0000. "
        "The remaining supported classes are normalized to guarantee an exact sum of 1.0:"
    )
    add_code_block(
        doc,
        "def normalize_to_core_ontology(raw_probs, supported_emotions=None):\n"
        "    distribution = create_empty_distribution()  # All 7 classes initialized to 0.0\n"
        "    for raw_label, prob in raw_probs.items():\n"
        "        core_emotion, _ = map_label(raw_label)\n"
        "        distribution[core_emotion] += float(max(0.0, prob))\n\n"
        "    if supported_emotions is not None:\n"
        "        supported_set = set(supported_emotions)\n"
        "        for emotion in CORE_EMOTIONS:\n"
        "            if emotion not in supported_set:\n"
        "                distribution[emotion] = 0.0  # STRICT ZERO-MASS INVARIANT\n\n"
        "    total = sum(distribution.values())\n"
        "    if total > 0.0:\n"
        "        for emotion in distribution:\n"
        "            distribution[emotion] = round(distribution[emotion] / total, 4)\n"
        "    return distribution",
        title="ZERO-MASS NORMALIZATION IN CORE/ONTOLOGY.PY"
    )

def build_section_4_text_engine(doc):
    """Section 4: Modality Engine 1: Natural Language Processing."""
    add_heading_1(doc, "4. Modality Engine 1: Natural Language Processing (Text Engine)")

    add_body_paragraph(
        doc,
        "The Text Modality Engine processes natural language sentences, dialogues, or transcripts, extracting linguistic affect via a fine-tuned "
        "transformer model. The module is encapsulated in core/text/model.py and core/text/preprocessor.py."
    )

    add_heading_2(doc, "4.1 Transformer Architecture & Model Selection")
    add_body_paragraph(
        doc,
        "The system employs j-hartmann/emotion-english-distilroberta-base, a state-of-the-art distilled RoBERTa (Robustly Optimized BERT Approach) "
        "architecture trained specifically for multi-class emotion classification across six diverse English benchmarks (including SemEval, ISEAR, and WASSA). "
        "Key architectural characteristics include:"
    )
    add_bullet_point(doc, "DistilRoBERTa (6 Transformer encoder layers, 768 hidden units, 12 self-attention heads, ~82M parameters).", bold_prefix="Backbone: ")
    add_bullet_point(doc, "Direct Softmax classification head over [CLS] token pooling into 7 emotion logits.", bold_prefix="Head: ")
    add_bullet_point(doc, "Byte-Pair Encoding (BPE) subword tokenizer with 50,265 vocabulary size.", bold_prefix="Tokenizer: ")
    add_bullet_point(doc, "CPU inference latency: ~15 to 35 ms per sentence; GPU latency: ~4 ms.", bold_prefix="Latency: ")

    add_heading_2(doc, "4.2 Preprocessing & Salient Feature Extraction")
    add_body_paragraph(
        doc,
        "The TextPreprocessor class sanitizes raw text prior to inference while extracting salient keywords that explain the classification decision:"
    )
    add_bullet_point(doc, "Stripping extraneous URLs, HTTP hyperlinks, and @mentions.", bold_prefix="Regex Cleaning: ")
    add_bullet_point(doc, "Preserving emotional emojis and punctuation (exclamation marks, ellipses) that convey sentiment intensity.", bold_prefix="Token Preservation: ")
    add_bullet_point(doc, "Enforcing a hard ceiling of 512 characters to prevent buffer exhaustion and out-of-memory errors.", bold_prefix="Length Bounds: ")
    add_bullet_point(doc, "Extracting salient emotional tokens (e.g., 'devastated', 'ecstatic', 'terrified') to populate the explainability report.", bold_prefix="Salient Keywords: ")

    add_heading_2(doc, "4.3 Text Prediction Code Implementation")
    add_code_block(
        doc,
        "class TextEmotionModel:\n"
        "    def predict(self, raw_text: str) -> ModalityResult:\n"
        "        cleaned_text = self.preprocessor.clean(raw_text)\n"
        "        salient_keywords = self.preprocessor.extract_salient_keywords(cleaned_text)\n\n"
        "        if not self._is_loaded or self.pipeline is None:\n"
        "            return ModalityResult(modality='text', available=False,\n"
        "                status=f'TEXT_INFERENCE_BLOCKED: {self.load_error}', ...)\n\n"
        "        # Genuine transformer sequence classification\n"
        "        outputs = self.pipeline(cleaned_text[:512])[0]\n"
        "        raw_probs = {item['label'].lower(): float(item['score']) for item in outputs}\n"
        "        normalized_probs = normalize_to_core_ontology(raw_probs, supported_emotions=CORE_EMOTIONS)\n\n"
        "        top_emotion = max(normalized_probs, key=normalized_probs.get)\n"
        "        confidence = float(normalized_probs[top_emotion])\n"
        "        valence = sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]['valence'] for e in CORE_EMOTIONS)\n"
        "        arousal = sum(normalized_probs[e] * EMOTION_VALENCE_AROUSAL[e]['arousal'] for e in CORE_EMOTIONS)\n\n"
        "        return ModalityResult(\n"
        "            modality='text', available=True, status='ok',\n"
        "            emotion=top_emotion, confidence=round(confidence, 4),\n"
        "            probabilities=normalized_probs, supported_emotions=CORE_EMOTIONS,\n"
        "            valence=round(valence, 4), arousal=round(arousal, 4),\n"
        "            metadata={'word_count': len(cleaned_text.split()), 'salient_keywords': salient_keywords}\n"
        "        )",
        title="TEXT PREDICTION IN CORE/TEXT/MODEL.PY"
    )

def build_section_5_audio_engine(doc):
    """Section 5: Modality Engine 2: Acoustic Speech Processing."""
    add_heading_1(doc, "5. Modality Engine 2: Acoustic Speech Processing (Audio Engine)")

    add_body_paragraph(
        doc,
        "The Speech Modality Engine classifies emotional affect from acoustic speech waveforms. The model operates independently of linguistic content, "
        "analyzing vocal tract resonance, pitch modulation, cadence, and harmonic energy. The pipeline is contained in core/speech/model.py and "
        "core/speech/feature_extractor.py."
    )

    add_heading_2(doc, "5.1 Acoustic Feature Extraction (39-Channel MFCCs)")
    add_body_paragraph(
        doc,
        "Audio processing adheres strictly to the acoustic representation expected by the trained speech network:"
    )
    add_bullet_point(doc, "Audio inputs are resampled to 16,000 Hz single-channel (mono) 16-bit PCM.", bold_prefix="Sampling Rate: ")
    add_bullet_point(doc, "Signals are standardized to exactly 2.0 seconds (32,000 samples). Audio shorter than 2.0s is zero-padded symmetrically; audio longer than 2.0s is centrally cropped.", bold_prefix="Temporal Windowing: ")
    add_bullet_point(doc, "Extracts 39 acoustic channels: 13 Mel-Frequency Cepstral Coefficients (MFCCs) + 13 First-Order Delta coefficients + 13 Second-Order Delta-Delta coefficients across 198 time frames.", bold_prefix="MFCC Spectrograms: ")
    add_bullet_point(doc, "Computes Root Mean Square (RMS) energy, Zero-Crossing Rate (ZCR), and fundamental pitch via autocorrelation to feed explainability attribution.", bold_prefix="Acoustic Descriptors: ")

    add_heading_2(doc, "5.2 PyTorch Reimplementation of Keras 2D CNN (Zero TensorFlow Dependency)")
    add_body_paragraph(
        doc,
        "The original research repository (speech-emotion-recognition-master) defined its network using TensorFlow 1.x / Keras Sequential models. "
        "To eliminate heavy, fragile TensorFlow runtime dependencies and avoid Python 3.14 C-extension incompatibilities, the ERM engineering team "
        "reconstructed the exact neural architecture as a native PyTorch nn.Module (_SpeechEmotionCNNModule) and built a direct HDF5 weight loader via h5py:"
    )

    cnn_headers = ["Layer Number", "Layer Type", "Parameters / Kernel", "Activation", "Output Tensor Shape (B, C, H, W)"]
    cnn_rows = [
        ["Layer 1", "Conv2D + BatchNorm", "8 filters, 13x13 kernel, stride (1,1)", "ReLU", "(Batch, 8, 186, 27)"],
        ["Layer 2", "Conv2D + BatchNorm + MaxPool", "8 filters, 13x13 kernel + Pool (2,1)", "ReLU", "(Batch, 8, 87, 15)"],
        ["Layer 3", "Conv2D + BatchNorm", "8 filters, 13x13 kernel, stride (1,1)", "ReLU", "(Batch, 8, 75, 3)"],
        ["Layer 4", "Conv2D + BatchNorm + MaxPool", "8 filters, 2x2 kernel + Pool (2,1)", "ReLU", "(Batch, 8, 37, 2)"],
        ["Permute & Flatten", "Permute (0, 2, 3, 1) + View", "Transposition to replicate Keras H,W,C memory layout", "None", "(Batch, 592)"],
        ["Dense 1", "Linear + BatchNorm + Dropout", "592 in -> 64 out (weight shape [64, 592])", "ReLU", "(Batch, 64)"],
        ["Dense 2 (Head)", "Linear + Softmax", "64 in -> 4 out (weight shape [4, 64])", "Softmax", "(Batch, 4)"],
    ]
    add_styled_table(doc, cnn_headers, cnn_rows, col_widths=[1.0, 1.6, 1.8, 0.9, 1.2])

    add_heading_2(doc, "5.3 HDF5 Binary Weight Transposition Implementation")
    add_body_paragraph(
        doc,
        "Convolutional weights stored in Keras HDF5 format use the layout (H, W, In_Channels, Out_Channels), whereas PyTorch requires "
        "(Out_Channels, In_Channels, H, W). Similarly, Keras Dense layers store (In_Features, Out_Features) whereas PyTorch expects "
        "(Out_Features, In_Features). The loader explicitly transposes all weight tensors during model initialization:"
    )
    add_code_block(
        doc,
        "# Convolution kernel conversion from Keras (H, W, C_in, C_out) to PyTorch (C_out, C_in, H, W)\n"
        "c1_k = np.array(h['conv2d_1']['conv2d_1']['kernel:0']).transpose(3, 2, 0, 1)\n"
        "c1_b = np.array(h['conv2d_1']['conv2d_1']['bias:0'])\n"
        "self.model.conv1.weight.data = torch.from_numpy(c1_k).float().to(self.device)\n"
        "self.model.conv1.bias.data = torch.from_numpy(c1_b).float().to(self.device)\n\n"
        "# BatchNorm gamma, beta, moving_mean, moving_variance mapping\n"
        "self.model.bn1.weight.data = torch.from_numpy(np.array(h['batch_normalization_1']['...']['gamma:0'])).float()\n"
        "self.model.bn1.bias.data = torch.from_numpy(np.array(h['batch_normalization_1']['...']['beta:0'])).float()\n"
        "self.model.bn1.running_mean.data = torch.from_numpy(np.array(h['...']['moving_mean:0'])).float()\n"
        "self.model.bn1.running_var.data = torch.from_numpy(np.array(h['...']['moving_variance:0'])).float()\n\n"
        "# Dense layer conversion from Keras (In, Out) to PyTorch (Out, In)\n"
        "d1_k = np.array(h['dense_1']['dense_1']['kernel:0']).transpose(1, 0)\n"
        "self.model.fc1.weight.data = torch.from_numpy(d1_k).float()",
        title="DIRECT HDF5 LAYER WEIGHT EXTRACTION (CORE/SPEECH/MODEL.PY)"
    )

    add_callout(
        doc,
        "The speech CNN model natively outputs predictions across 4 classes: neutral, angry, happy, and sad. "
        "In strict adherence to the Zero Fake AI Policy, disgust, fear, and surprise receive exactly 0.0000 probability mass.",
        title="SPEECH ONTOLOGY BOUNDARY",
        alert_type="warn"
    )

def build_section_6_vision_engine(doc):
    """Section 6: Modality Engine 3: Facial Vision Emotion Recognition."""
    add_heading_1(doc, "6. Modality Engine 3: Facial Vision Emotion Recognition (Vision Engine)")

    add_body_paragraph(
        doc,
        "The Vision Modality Engine detects human faces in static imagery or video streams, isolates facial regions of interest, "
        "and classifies affective state using an AlexNet deep convolutional network. The subsystem is structured in core/vision/model.py "
        "and core/vision/face_detector.py."
    )

    add_heading_2(doc, "6.1 Multi-Pass OpenCV Haar Cascade Face Detection Subsystem")
    add_body_paragraph(
        doc,
        "Static image uploads and live webcam video streams frequently suffer from uneven illumination, shadows, and off-axis head tilt. "
        "To maximize face detection recall without requiring heavyweight neural detector dependencies, FaceDetector implements a robust 3-pass strategy:"
    )
    add_bullet_point(doc, "OpenCV haarcascade_frontalface_default.xml with scaleFactor=1.1, minNeighbors=4, minSize=(40, 40). Rapidly captures high-contrast frontal faces.", bold_prefix="Pass 1 (Primary Frontal): ")
    add_bullet_point(doc, "Converts input to grayscale and applies Contrast Limited Adaptive Histogram Equalization (CLAHE) / cv2.equalizeHist before re-running frontal detection. Recovers faces obscured by harsh shadows or dim lighting.", bold_prefix="Pass 2 (Histogram Equalization): ")
    add_bullet_point(doc, "Applies haarcascade_frontalface_alt2.xml. Specifically tuned to capture slight profile angles, tilted heads, and non-rigid facial configurations.", bold_prefix="Pass 3 (Alternative Alt2): ")
    add_bullet_point(doc, "Detected bounding boxes are expanded by 10% to retain forehead and chin landmarks, cropped, and resized via bicubic interpolation to 224x224 RGB.", bold_prefix="Crop Normalization: ")

    add_heading_2(doc, "6.2 AlexNet Vision Architecture & Classifier Head")
    add_body_paragraph(
        doc,
        "The neural vision classifier utilizes an AlexNet deep convolutional network with ImageNet pre-trained feature extractors:"
    )
    add_bullet_point(doc, "5 Convolutional Layers (conv1: 11x11, conv2: 5x5, conv3-5: 3x3) with ReLU activations and MaxPool2D(3x3, stride 2).", bold_prefix="Feature Backbone: ")
    add_bullet_point(doc, "AdaptiveAvgPool2d((6, 6)) producing a flattened 9,216-dimensional spatial feature vector.", bold_prefix="Spatial Pooling: ")
    add_bullet_point(doc, "Linear(9216 -> 4096) -> ReLU -> Dropout(0.5) -> Linear(4096 -> 4096) -> ReLU -> Dropout(0.5) -> Linear(4096 -> 7) -> Softmax(dim=1).", bold_prefix="Classifier Head: ")
    add_bullet_point(doc, "ImageNet channel normalization tensors: Mean=[0.485, 0.456, 0.406], Std=[0.229, 0.224, 0.225].", bold_prefix="Tensor Normalization: ")

    add_heading_2(doc, "6.3 Vision Prediction Code Flow")
    add_code_block(
        doc,
        "class VisionEmotionModel:\n"
        "    def predict(self, image_source: Any) -> ModalityResult:\n"
        "        if not self._is_loaded or self.model is None:\n"
        "            return ModalityResult(modality='vision', available=False,\n"
        "                status='VISION INFERENCE BLOCKED — TRAINED CHECKPOINT MISSING', ...)\n\n"
        "        # Multi-pass face detection\n"
        "        rgb_image = self.detector.load_image(image_source)\n"
        "        crops, boxes = self.detector.detect_and_crop_faces(rgb_image, fallback_if_no_face=True)\n"
        "        face_box = boxes[0] if boxes else None\n\n"
        "        # Tensor transformation\n"
        "        tensor = torch.from_numpy(crops[0]).permute(2, 0, 1).float() / 255.0\n"
        "        for c in range(3):\n"
        "            tensor[c] = (tensor[c] - self.mean[c]) / self.std[c]\n"
        "        tensor = tensor.unsqueeze(0).to(self.device)\n\n"
        "        # Genuine neural forward pass\n"
        "        with torch.no_grad():\n"
        "            probs_tensor = self.model(tensor)[0]\n"
        "            raw_probs = {CORE_EMOTIONS[i]: float(probs_tensor[i]) for i in range(len(CORE_EMOTIONS))}\n\n"
        "        normalized_probs = normalize_to_core_ontology(raw_probs, supported_emotions=CORE_EMOTIONS)\n"
        "        top_emotion = max(normalized_probs, key=normalized_probs.get)\n"
        "        confidence = float(normalized_probs[top_emotion])\n"
        "        return ModalityResult(modality='vision', available=True, status='ok',\n"
        "            emotion=top_emotion, confidence=round(confidence, 4),\n"
        "            probabilities=normalized_probs, supported_emotions=CORE_EMOTIONS, ...)",
        title="VISION PREDICTION IN CORE/VISION/MODEL.PY"
    )

def build_section_7_vision_training_evaluation(doc):
    """Section 7: Vision Model Training Methodology & Empirical Evaluation."""
    add_heading_1(doc, "7. Vision Model Training Methodology & Empirical Evaluation")

    add_body_paragraph(
        doc,
        "The legacy Multimodal-Emotion-Recognition-master repository contained research notebooks (visual_network.ipynb) and raw RAVDESS video clips, "
        "but did not provide the author's trained .pth checkpoint file. In strict adherence to our Zero Fake AI Policy, the engineering team "
        "built an isolated, fully reproducible training pipeline in training/vision/ to train and validate the AlexNet classifier on held-out actors."
    )

    add_heading_2(doc, "7.1 Dataset Composition & Zero Data Leakage")
    add_body_paragraph(
        doc,
        "The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS) facial video frame dataset was processed into clean 224x224 face crops. "
        "To ensure true generalization, an actor-independent split was enforced:"
    )
    add_bullet_point(doc, "12,647 face crops strictly from Actor_01 through Actor_20 (20 actors, 1,200 clips).", bold_prefix="Training Set: ")
    add_bullet_point(doc, "2,621 face crops strictly from Actor_21 through Actor_24 (4 actors, 240 clips).", bold_prefix="Held-Out Validation Set: ")
    add_bullet_point(doc, "0.0% actor overlap. The validation actors were completely unseen during training, preventing facial identity leakage.", bold_prefix="Actor Leakage: ")

    data_headers = ["Emotion Category", "Train Images", "Train Share (%)", "Train Clips", "Validation Images", "Validation Share (%)", "Validation Clips"]
    data_rows = [
        ["angry", "1,778", "14.06%", "160", "362", "13.81%", "32"],
        ["disgust", "1,812", "14.33%", "160", "366", "13.96%", "32"],
        ["fear", "1,623", "12.83%", "160", "338", "12.90%", "32"],
        ["happy", "1,674", "13.24%", "160", "332", "12.67%", "32"],
        ["neutral", "2,510", "19.85%", "240", "533", "20.34%", "48"],
        ["sad", "1,657", "13.10%", "160", "371", "14.15%", "32"],
        ["surprise", "1,593", "12.60%", "160", "319", "12.17%", "32"],
        ["Total Clean", "12,647", "100.00%", "1,200", "2,621", "100.00%", "240"],
    ]
    add_styled_table(doc, data_headers, data_rows, col_widths=[1.1, 0.9, 0.9, 0.8, 1.0, 1.0, 0.8])

    add_callout(
        doc,
        "The neutral class combines RAVDESS code 01 (neutral) and code 02 (calm). Training has 786 code-01 frames and 1,724 code-02 frames. "
        "Validation has 168 code-01 frames and 365 code-02 frames. Consequently, calm contributes approximately 69% of neutral frames, "
        "making neutral both larger and semantically broader than other categories.",
        title="DATASET DISTRIBUTION INSIGHT",
        alert_type="info"
    )

    add_heading_2(doc, "7.2 Training Optimization & Ablation Experiments")
    add_body_paragraph(
        doc,
        "Multiple controlled training experiments were executed to optimize classification accuracy and Macro-F1 score under inverse-frequency class weighting: "
        "w_c = (max_k N_k / N_c)^alpha, normalized to mean 1.0."
    )

    exp_headers = ["Run Configuration", "Loss Weight Strength (alpha)", "Best Epoch / Max", "Val Accuracy", "Macro-F1", "Correct / 2,621"]
    exp_rows = [
        ["Baseline (SGD, StepLR, frozen conv)", "Unweighted (0.00)", "Epoch 26 / 41", "37.39%", "0.3258", "980 / 2,621"],
        ["Ablation 1 (AdamW, Cosine, unweighted)", "0.00 (All ones)", "Epoch 4 / 9 (early)", "40.29%", "0.3562", "1,056 / 2,621"],
        ["Ablation 2 (AdamW, Cosine, mild weight)", "0.25", "Epoch 19 / 20", "41.70%", "0.3912", "1,093 / 2,621"],
        ["Selected Checkpoint (AdamW, Cosine, tuned)", "0.50 (Inverse freq)", "Epoch 15 / 20", "41.59%", "0.3956", "1,090 / 2,621"],
    ]
    add_styled_table(doc, exp_headers, exp_rows, col_widths=[1.8, 1.2, 1.0, 0.9, 0.8, 0.8])

    add_body_paragraph(
        doc,
        "Decision Rationale: While weight strength 0.25 yielded 3 more correct raw frames (41.70% vs 41.59%), weight strength 0.50 achieved "
        "the superior Macro-F1 score (0.3956 vs 0.3912) and significantly higher Happy F1 (0.5781 vs 0.5441). The 0.50 checkpoint was selected "
        "for production deployment."
    )

    add_heading_2(doc, "7.3 Per-Class Performance Metrics (Baseline vs Selected Checkpoint)")
    perf_headers = ["Emotion Class", "Baseline Precision", "Selected Precision", "Baseline Recall", "Selected Recall", "Baseline F1", "Selected F1"]
    perf_rows = [
        ["angry", "0.5545", "0.6569", "0.1685", "0.1851", "0.2585", "0.2888"],
        ["disgust", "0.5146", "0.6372", "0.2896", "0.3934", "0.3706", "0.4865"],
        ["fear", "0.4711", "0.5120", "0.1686", "0.3166", "0.2484", "0.3912"],
        ["happy", "0.6150", "0.7156", "0.4187", "0.4849", "0.4982", "0.5781"],
        ["neutral", "0.3236", "0.3521", "0.9381", "0.8199", "0.4812", "0.4927"],
        ["sad", "0.2731", "0.2806", "0.1752", "0.2534", "0.2135", "0.2663"],
        ["surprise", "0.2971", "0.2827", "0.1630", "0.2508", "0.2105", "0.2658"],
        ["Macro Average", "0.4356", "0.4910", "0.3317", "0.3863", "0.3258", "0.3956"],
    ]
    add_styled_table(doc, perf_headers, perf_rows, col_widths=[1.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])

    add_heading_2(doc, "7.4 Full Confusion Matrices")
    add_body_paragraph(doc, "Rows represent True ground truth labels; Columns represent Model predictions:")

    add_heading_3(doc, "Baseline Checkpoint Confusion Matrix (Total: 980 / 2,621 correct):")
    conf_base_headers = ["True \\ Pred", "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    conf_base_rows = [
        ["angry", "61", "24", "9", "20", "175", "46", "27"],
        ["disgust", "15", "106", "15", "12", "121", "93", "4"],
        ["fear", "8", "14", "57", "4", "172", "25", "58"],
        ["happy", "0", "3", "0", "139", "178", "0", "12"],
        ["neutral", "1", "0", "0", "18", "500", "1", "13"],
        ["sad", "5", "47", "0", "18", "227", "65", "9"],
        ["surprise", "20", "12", "40", "15", "172", "8", "52"],
    ]
    add_styled_table(doc, conf_base_headers, conf_base_rows)

    add_heading_3(doc, "Selected Production Checkpoint (Strength 0.50, Total: 1,090 / 2,621 correct):")
    conf_sel_headers = ["True \\ Pred", "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
    conf_sel_rows = [
        ["angry", "67", "27", "3", "6", "153", "58", "48"],
        ["disgust", "12", "144", "6", "6", "106", "88", "4"],
        ["fear", "0", "9", "107", "2", "104", "61", "55"],
        ["happy", "0", "0", "12", "161", "128", "1", "30"],
        ["neutral", "0", "0", "15", "19", "437", "10", "52"],
        ["sad", "3", "39", "10", "22", "189", "94", "14"],
        ["surprise", "20", "7", "56", "9", "124", "23", "80"],
    ]
    add_styled_table(doc, conf_sel_headers, conf_sel_rows)

    # Embed diagnostic plot if exists
    chart_path = os.path.abspath("reports/vision_happy_diagnostics.png")
    if os.path.isfile(chart_path):
        add_heading_2(doc, "7.5 Empirical Diagnostic Analysis")
        add_body_paragraph(doc, "Diagnostic evaluation of the Happy classification performance across ablation checkpoints:")
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(6)
        p_img.paragraph_format.space_after = Pt(6)
        run_img = p_img.add_run()
        run_img.add_picture(chart_path, width=Inches(5.8))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 7.1: Comparative Diagnostic Evaluation of Happy Facial Classification across Checkpoints.")
        r_cap.font.name = "Arial"
        r_cap.font.size = Pt(8.5)
        r_cap.font.italic = True
        r_cap.font.color.rgb = COLOR_MUTED_TEXT

def build_section_8_fusion_engine(doc):
    """Section 8: Multi-Modal Decision Fusion Engine."""
    add_heading_1(doc, "8. Multi-Modal Decision Fusion Engine")

    add_body_paragraph(
        doc,
        "The core intelligence of the platform resides in the MultiModalFusionEngine (core/fusion/engine.py). Rather than relying on naive majority "
        "voting or fixed arithmetic averaging, the engine implements confidence-aware weighted late decision fusion with cross-modal concordance "
        "and conflict analysis."
    )

    add_heading_2(doc, "8.1 Mathematical Decision Fusion Formulation")
    add_body_paragraph(
        doc,
        "Let M = {Text, Speech, Vision} denote the set of modalities. The fusion engine executes the following sequential steps:"
    )

    add_heading_3(doc, "Step 1: Modality Availability Gating")
    add_body_paragraph(
        doc,
        "Each modality prediction is represented as a ModalityResult object. An active indicator A_m in {0, 1} is evaluated:\n"
        "A_m = 1 if (m.available == True and m.status == 'ok' and ValidDistribution(P_m)), else 0.\n"
        "Modalities with A_m = 0 (such as an audio file with missing speech model or corrupt upload) are cleanly excluded and receive zero weight."
    )

    add_heading_3(doc, "Step 2: Confidence-Adjusted Effective Weighting")
    add_body_paragraph(
        doc,
        "Each active modality m has a base configured prior weight W_m (default W_m = 1.0) and a neural model confidence C_m in [0.0, 1.0]. "
        "The unnormalized effective weight w_tilde_m is computed as:"
    )
    add_code_block(
        doc,
        "w_tilde_m = W_m * max(0.0001, C_m)\n\n"
        "# Normalized effective weight:\n"
        "w_m = w_tilde_m / sum(w_tilde_k for k in ActiveModalities)",
        title="EFFECTIVE NORMALIZED WEIGHT EQUATION"
    )

    add_heading_3(doc, "Step 3: Unified Probability Vector Synthesis")
    add_body_paragraph(
        doc,
        "The fused 7-class probability distribution P_fused(e) is computed as the weighted linear combination across all active modalities:"
    )
    add_code_block(
        doc,
        "P_fused(e) = sum(w_m * P_m(e) for m in ActiveModalities), for all e in CORE_EMOTIONS\n\n"
        "# Strict re-normalization to guarantee exact sum = 1.0:\n"
        "P_fused(e) = P_fused(e) / sum(P_fused(k) for k in CORE_EMOTIONS)",
        title="FUSED PROBABILITY DISTRIBUTION EQUATION"
    )

    add_heading_3(doc, "Step 4: Cross-Modal Concordance & Conflict Analysis")
    add_body_paragraph(
        doc,
        "To quantify whether active modalities are in emotional harmony or discordance, the engine computes pairwise Cosine Similarities across "
        "probability distributions:"
    )
    add_code_block(
        doc,
        "Similarity(P_1, P_2) = (P_1 . P_2) / (||P_1||_2 * ||P_2||_2)\n\n"
        "ConcordanceScore = mean(Similarity(P_i, P_j) for all pairs (i, j))\n"
        "ConflictScore = 1.0 - ConcordanceScore",
        title="PAIRWISE CONCORDANCE & CONFLICT EQUATIONS"
    )

    add_body_paragraph(
        doc,
        "Conflict Detection Threshold: If ConflictScore >= 0.40 or if any pair of active modalities predicts differing top emotions, "
        "the engine flags has_conflict = True, lists the divergent pairs, and synthesizes an objective, non-judgmental textual rationale."
    )

    add_heading_2(doc, "8.2 The 7 Modality Permutations Matrix")
    add_body_paragraph(
        doc,
        "The fusion engine deterministically handles all 2^3 - 1 = 7 active modality combinations without crashing or degraded accuracy:"
    )

    perm_headers = ["Permutation", "Active Modalities", "Fusion Mode", "Weight Behavior", "Concordance Analysis"]
    perm_rows = [
        ["1. Text Only", "Text", "single_modality", "w_text = 1.0", "Concordance = 1.0 (N/A)"],
        ["2. Audio Only", "Audio", "single_modality", "w_audio = 1.0", "Concordance = 1.0 (N/A)"],
        ["3. Vision Only", "Vision", "single_modality", "w_vision = 1.0", "Concordance = 1.0 (N/A)"],
        ["4. Text + Audio", "Text, Audio", "multimodal", "w_m = W_m * C_m / sum(W * C)", "Pairwise Cosine Sim (Text vs Audio)"],
        ["5. Text + Vision", "Text, Vision", "multimodal", "w_m = W_m * C_m / sum(W * C)", "Pairwise Cosine Sim (Text vs Vision)"],
        ["6. Audio + Vision", "Audio, Vision", "multimodal", "w_m = W_m * C_m / sum(W * C)", "Pairwise Cosine Sim (Audio vs Vision)"],
        ["7. Text + Audio + Vision", "Text, Audio, Vision", "multimodal", "w_m = W_m * C_m / sum(W * C)", "Mean across all 3 pairwise comparisons"],
    ]
    add_styled_table(doc, perm_headers, perm_rows, col_widths=[1.5, 1.4, 1.2, 1.2, 1.2])

    add_heading_2(doc, "8.3 Explainability Report Structure")
    add_body_paragraph(
        doc,
        "Every fused response includes a structured ExplainabilityReport object containing:"
    )
    add_bullet_point(doc, "primary_modality: Identifies which modality exerted the strongest mathematical pull on the final decision.", bold_prefix="Primary Driver: ")
    add_bullet_point(doc, "concordance_score: Float in [0.0, 1.0] indicating level of agreement across sensory channels.", bold_prefix="Concordance Score: ")
    add_bullet_point(doc, "modality_weights: Dictionary of exact normalized effective weights (e.g., {'text': 0.58, 'audio': 0.42}).", bold_prefix="Weight Attribution: ")
    add_bullet_point(doc, "rationale: Natural language summary describing the evidence base and driver behind the prediction.", bold_prefix="Textual Rationale: ")
    add_bullet_point(doc, "key_features: Salient feature passthrough (extracted text keywords, audio RMS/pitch, face count).", bold_prefix="Salient Features: ")

def build_section_9_backend_api(doc):
    """Section 9: FastAPI Backend Architecture & API Specification."""
    add_heading_1(doc, "9. FastAPI Backend Architecture & API Specification")

    add_body_paragraph(
        doc,
        "The backend is built with FastAPI, an asynchronous ASGI web framework running under Uvicorn. The architecture prioritizes "
        "model lifecycle caching, zero redundant memory allocations, strict input validation, and privacy-preserving error handling."
    )

    add_heading_2(doc, "9.1 Singleton Dependency Lifecycle (api/dependencies.py)")
    add_body_paragraph(
        doc,
        "Deep neural networks require significant VRAM / RAM and initial loading latency. The ERM API implements application-level singletons "
        "via FastAPI's dependency injection system. The models are loaded once on startup or initial query and retained in memory:"
    )
    add_bullet_point(doc, "get_text_model() -> TextEmotionModel (cached singleton).", bold_prefix="Text Singleton: ")
    add_bullet_point(doc, "get_speech_model() -> SpeechEmotionModel (cached singleton).", bold_prefix="Speech Singleton: ")
    add_bullet_point(doc, "get_vision_model() -> VisionEmotionModel (cached singleton).", bold_prefix="Vision Singleton: ")
    add_bullet_point(doc, "get_fusion_engine() -> MultiModalFusionEngine (cached singleton).", bold_prefix="Fusion Singleton: ")

    add_heading_2(doc, "9.2 Path Sanitization & Information Disclosure Protection")
    add_body_paragraph(
        doc,
        "To prevent internal host environment disclosure, sanitize_metadata() recursively scans all response dictionaries and replaces "
        "absolute Windows and Unix file paths with [REDACTED_PATH] or file basenames. Local system usernames and directories are never leaked to clients."
    )

    add_heading_2(doc, "9.3 REST API Endpoints Specification")

    add_heading_3(doc, "1. GET /api/health — System Readiness & Status")
    add_body_paragraph(doc, "Returns service operational health, active compute device, and individual model readiness flags.")
    add_code_block(
        doc,
        "GET /api/health HTTP/1.1\nHost: 127.0.0.1:8000\n\n"
        "HTTP/1.1 200 OK\nContent-Type: application/json\n\n"
        "{\n"
        '  "status": "ok",\n'
        '  "service": "emotion-recognition-api",\n'
        '  "version": "1.0.0",\n'
        '  "device": "cpu",\n'
        '  "models": {\n'
        '    "text": {"operational": true, "model_name": "j-hartmann/emotion-english-distilroberta-base"},\n'
        '    "audio": {"operational": true, "weights": "best_model_CNN.h5"},\n'
        '    "vision": {"operational": true, "status": "ok", "face_detection_available": true}\n'
        "  }\n"
        "}",
        title="HEALTH ENDPOINT REQUEST & RESPONSE"
    )

    add_heading_3(doc, "2. POST /api/predict/text — Text Emotion Recognition")
    add_body_paragraph(doc, "Accepts JSON raw text string and returns ModalityResult.")
    add_code_block(
        doc,
        "POST /api/predict/text HTTP/1.1\nContent-Type: application/json\n\n"
        '{"text": "I am so happy and overjoyed with our breakthrough!"}\n\n'
        "HTTP/1.1 200 OK\n"
        "{\n"
        '  "modality": "text", "available": true, "status": "ok",\n'
        '  "emotion": "happy", "confidence": 0.9421,\n'
        '  "probabilities": {"happy": 0.9421, "surprise": 0.0315, "neutral": 0.0124, ...},\n'
        '  "valence": 0.8241, "arousal": 0.5821,\n'
        '  "metadata": {"word_count": 9, "salient_keywords": ["happy", "overjoyed", "breakthrough"]}\n'
        "}",
        title="TEXT ENDPOINT SPECIFICATION"
    )

    add_heading_3(doc, "3. POST /api/predict/audio — Acoustic Speech Emotion Recognition")
    add_body_paragraph(doc, "Accepts multipart/form-data audio file (WAV, MP3, OGG, FLAC; max 25 MB). Returns ModalityResult.")

    add_heading_3(doc, "4. POST /api/predict/vision — Facial Vision Emotion Recognition")
    add_body_paragraph(doc, "Accepts multipart/form-data image file (JPG, PNG, WebP; max 10 MB). Returns ModalityResult.")

    add_heading_3(doc, "5. POST /api/predict/multimodal — Unified Late Decision Fusion")
    add_body_paragraph(
        doc,
        "Primary multi-modal endpoint. Accepts any combination of text (string), audio (file), and image (file). "
        "Executes relevant neural models and performs late decision fusion."
    )

    add_heading_3(doc, "6. POST /api/predict/fuse — Low-Latency Pre-Evaluated Modality Fusion")
    add_body_paragraph(
        doc,
        "Accepts a JSON payload containing pre-evaluated ModalityResult objects. "
        "Executes instant mathematical fusion in under 5 ms without re-evaluating deep models. Essential for real-time frontend stream coordination."
    )

def build_section_10_frontend_architecture(doc):
    """Section 10: Frontend Application Architecture."""
    add_heading_1(doc, "10. Frontend Application Architecture (React 19 + TypeScript + Vite)")

    add_body_paragraph(
        doc,
        "The frontend is an interactive, modern single-page dashboard built with React 19, TypeScript, and Vite. "
        "Adhering to strict web standards, the UI uses Vanilla CSS for maximum styling control, featuring glassmorphism, "
        "real-time SVG/Canvas audio waveform visualizers, dynamic probability meters, and responsive grid layouts."
    )

    add_heading_2(doc, "10.1 Reactive Custom Hooks Architecture")
    add_body_paragraph(
        doc,
        "Stateful sensory stream logic is abstracted into dedicated custom hooks in frontend/src/hooks/:"
    )
    add_bullet_point(doc, "useLiveText.ts: Debounced text input classifier. Provides auto-analysis, editable sentence templates, and instant cancellation via AbortController.", bold_prefix="useLiveText: ")
    add_bullet_point(doc, "useLiveAudio.ts: Integrates Web Audio API AudioContext and AnalyserNode. Captures microphone streams, computes real-time volume RMS, renders audio frequency waveforms, and periodically submits audio buffers for classification.", bold_prefix="useLiveAudio: ")
    add_bullet_point(doc, "useLiveWebcam.ts: Captures HTML5 webcam streams, renders video to a hidden offscreen canvas, checks for face presence, and sends periodic image snapshots to /api/predict/vision.", bold_prefix="useLiveWebcam: ")
    add_bullet_point(doc, "useLiveFusionCoordinator.ts: Central state orchestrator. Aggregates streaming outputs from active modalities and dispatches fast fusion requests to /api/predict/fuse, ensuring synchronized UI updates.", bold_prefix="useLiveFusionCoordinator: ")

    add_heading_2(doc, "10.2 Component Hierarchy & Responsibilities")
    comp_headers = ["Component File", "Visual Role", "Functional Responsibility"]
    comp_rows = [
        ["Header.tsx", "Top Navigation Bar", "System branding, operational status badge, and backend connectivity indicator."],
        ["ModalityStatusCard.tsx", "Modality Readiness", "Displays operational readiness, architecture name, and checkpoint status for Text, Audio, Vision."],
        ["MasterLiveControls.tsx", "Global Control Toolbar", "Single-click Master Start / Pause / Reset controls for all active media streams."],
        ["LiveTextPanel.tsx", "Interactive Text Panel", "Debounced textarea with preset emotional sentences and real-time inference trigger."],
        ["LiveAudioPanel.tsx", "Acoustic Audio Panel", "Microphone stream toggle, volume RMS meter, and real-time audio waveform canvas."],
        ["LiveWebcamPanel.tsx", "Facial Webcam Panel", "Webcam video viewport, face detection indicator, and snapshot capture intervals."],
        ["UnifiedEmotionOutput.tsx", "Hero Output Display", "Large dominant emotion badge, confidence gauge, Russell's Circumplex 2D plot, and primary driver."],
        ["ProbabilityChart.tsx", "7-Class Probability Chart", "Real-time animated horizontal bar charts with emotion-coded colors."],
        ["ModalityContributionPanel.tsx", "Attribution Breakdown", "Visual proportional contribution bars displaying effective mathematical weights (w_m)."],
        ["ConflictPanel.tsx", "Cross-Modal Divergence", "Concordance score display, conflict alerts, and pairwise disagreement breakdowns."],
        ["ManualAnalysisDrawer.tsx", "File Upload Drawer", "Expandable drawer for manual drag-and-drop file upload analysis of static audio and image files."],
    ]
    add_styled_table(doc, comp_headers, comp_rows, col_widths=[1.8, 1.6, 3.1])

def build_section_11_native_programs(doc):
    """Section 11: Standalone Real-Time Native Programs."""
    add_heading_1(doc, "11. Standalone Real-Time Native Programs (CLI / GUI)")

    add_body_paragraph(
        doc,
        "In addition to the web dashboard, the repository includes three standalone native programs for direct edge execution "
        "without a browser or web server. These tools provide ultra-low latency testing and debugging."
    )

    add_heading_2(doc, "11.1 live_face.py — Standalone OpenCV Webcam HUD")
    add_body_paragraph(
        doc,
        "live_face.py is a real-time OpenCV desktop application that captures live video frames from a connected webcam, runs multi-pass "
        "Haar face detection, evaluates the AlexNet classifier, and renders a heads-up display (HUD) directly on the native video window."
    )
    add_bullet_point(doc, "Decoupled Camera and Inference Loops: The camera capture and OpenCV rendering execute at smooth 30 FPS, while neural inference is throttled to 3.5 FPS to prevent CPU core exhaustion.", bold_prefix="Performance Decoupling: ")
    add_bullet_point(doc, "Dynamic Bounding Box: Changes color dynamically based on predicted emotion (green=happy, red=angry, purple=fear, etc.).", bold_prefix="Color-Coded Visuals: ")
    add_bullet_point(doc, "Overlay Probability Meters: Renders horizontal ASCII/graphic probability bars for all 7 Ekman classes on screen.", bold_prefix="On-Screen Telemetry: ")
    add_bullet_point(doc, "CLI Flags: --camera-id (default 0), --fps (default 3.5), --weights, --device ('cpu' or 'cuda'), --duration.", bold_prefix="Configuration: ")

    add_heading_2(doc, "11.2 live_audio.py — Standalone Native Speech Recognition")
    add_body_paragraph(
        doc,
        "live_audio.py captures audio from the default or specified microphone using sounddevice, runs continuous 2.0-second sliding window "
        "analysis, computes 39-channel MFCCs, and evaluates the 2D CNN acoustic model."
    )
    add_bullet_point(doc, "Circular Buffer Callback: SoundDevice streaming callback continuously ingests 16 kHz audio samples into a circular buffer.", bold_prefix="Buffer Architecture: ")
    add_bullet_point(doc, "Silence Gating: Energy threshold (default 0.003 RMS) ignores room ambient noise and silence.", bold_prefix="Silence Suppression: ")
    add_bullet_point(doc, "Terminal Output: Displays formatted probability distributions and emotion predictions directly in the console.", bold_prefix="Terminal Output: ")

    add_heading_2(doc, "11.3 live_text.py — Interactive Terminal REPL")
    add_body_paragraph(
        doc,
        "live_text.py provides an interactive Read-Eval-Print Loop (REPL) for classifying user-typed sentences via the DistilRoBERTa model. "
        "Features colored ASCII confidence bars, Circumplex Valence/Arousal coordinates, and instant response."
    )

def build_section_12_automated_testing(doc):
    """Section 12: Automated Testing, Verification, & Quality Assurance Suite."""
    add_heading_1(doc, "12. Automated Testing, Verification, & Quality Assurance Suite")

    add_body_paragraph(
        doc,
        "Reliability and mathematical correctness are enforced through automated test suites spanning backend integration, "
        "decision fusion unit tests, and frontend UI components."
    )

    add_heading_2(doc, "12.1 Backend PyTest Suite (21 / 21 Tests Passing)")
    add_body_paragraph(
        doc,
        "The backend test suite is executed with pytest -v. All 21 tests pass with zero warnings or errors:"
    )

    test_headers = ["Test Module", "Test Name", "Validation Scope & Invariant Verified"]
    test_rows = [
        ["tests/test_api.py", "test_health_endpoint", "Verifies 200 OK, device status, and operational flags for all modalities."],
        ["tests/test_api.py", "test_text_prediction_success", "Validates RoBERTa inference, happy classification, and probability sum."],
        ["tests/test_api.py", "test_audio_prediction_success", "Validates audio upload, 2D CNN inference on Sad sample, and zero-mass."],
        ["tests/test_api.py", "test_vision_prediction_success", "Validates image upload, AlexNet face inference, and 7-class distribution."],
        ["tests/test_api.py", "test_multimodal_prediction_success", "Validates tri-modal fusion, effective weight attribution, and schema."],
        ["tests/test_api.py", "test_file_size_limits", "Enforces 413 Payload Too Large on uploads exceeding 25MB audio / 10MB vision."],
        ["tests/test_api.py", "test_unsupported_extensions", "Enforces 415 Unsupported Media Type on invalid file formats (e.g. .exe, .txt)."],
        ["tests/test_api.py", "test_path_redaction", "Verifies that local filesystem paths never appear in response metadata."],
        ["tests/test_api.py", "test_temp_file_cleanup", "Guarantees temporary upload files are unlinked even if inference errors occur."],
        ["tests/test_fusion.py", "test_1_text_only", "Validates single-modality bypass with w_text = 1.0."],
        ["tests/test_fusion.py", "test_2_audio_only", "Validates single-modality bypass with w_audio = 1.0."],
        ["tests/test_fusion.py", "test_3_vision_only", "Validates single-modality bypass with w_vision = 1.0."],
        ["tests/test_fusion.py", "test_4_text_audio_concordant", "Validates concordant dual-modality weighting and high concordance score."],
        ["tests/test_fusion.py", "test_5_text_vision_concordant", "Validates concordant text and vision probability combination."],
        ["tests/test_fusion.py", "test_6_audio_vision_concordant", "Validates concordant speech and vision decision fusion."],
        ["tests/test_fusion.py", "test_7_tri_modal_concordant", "Validates tri-modal fusion when all three modalities agree."],
        ["tests/test_fusion.py", "test_8_conflict_detection", "Validates conflict flag, divergent pairs, and low concordance score."],
        ["tests/test_fusion.py", "test_9_zero_valid_modalities", "Verifies graceful return of unavailable result when all inputs invalid."],
        ["tests/test_fusion.py", "test_10_missing_vision_weights", "Verifies vision exclusion without request failure when weights missing."],
        ["tests/test_fusion.py", "test_11_zero_mass_preservation", "Verifies speech unsupported classes remain strictly 0.0 in fusion."],
        ["tests/test_fusion.py", "test_12_configured_weight_override", "Verifies custom modality weight priors (W_m) are respected."],
    ]
    add_styled_table(doc, test_headers, test_rows, col_widths=[1.6, 2.0, 2.9])

    add_heading_2(doc, "12.2 Frontend Vitest Suite (16 / 16 Tests Passing)")
    add_body_paragraph(
        doc,
        "The frontend test suite is executed with npm test -- --run inside frontend/. All 16 Vitest tests pass, validating UI component rendering, "
        "operational status cards, Web Audio API mock fallbacks, and API client error handling."
    )

def build_section_13_security_privacy(doc):
    """Section 13: Security, Privacy, & Local Data Sanitization Architecture."""
    add_heading_1(doc, "13. Security, Privacy, & Local Data Sanitization Architecture")

    add_body_paragraph(
        doc,
        "The ERM system was engineered from the ground up for strict local privacy and enterprise data security. "
        "Affective computing involves biometric signals (human faces, vocal recordings); thus, data containment is paramount."
    )

    add_bullet_point(
        doc,
        "No natural language text, voice audio, or facial imagery is ever transmitted to third-party cloud services or external APIs. "
        "All model checkpoints execute locally on the host CPU or local CUDA GPU.",
        bold_prefix="100% Local Inference: "
    )
    add_bullet_point(
        doc,
        "The application contains zero Google Analytics, telemetry scripts, remote crash reporters, or tracking beacons. "
        "Network activity is strictly confined to 127.0.0.1.",
        bold_prefix="Zero Remote Telemetry: "
    )
    add_bullet_point(
        doc,
        "Multipart uploads are written to isolated operating system temporary directories (tempfile.mkstemp) and are guaranteed "
        "to be deleted immediately in try...finally blocks, even when processing exceptions occur.",
        bold_prefix="Ephemeral File Lifecycle: "
    )
    add_bullet_point(
        doc,
        "The API intercepts all responses and redacts local operating system paths, preventing attacker reconnaissance of directory structures.",
        bold_prefix="Filesystem Path Redaction: "
    )
    add_bullet_point(
        doc,
        "Strict upload payload limits (25 MB audio, 10 MB image, 1000 character text) prevent memory exhaustion and denial-of-service attacks.",
        bold_prefix="Strict Input Validation: "
    )

def build_section_14_deployment_operations(doc):
    """Section 14: Installation, Deployment, & Operations Manual."""
    add_heading_1(doc, "14. Installation, Deployment, & Operations Manual")

    add_body_paragraph(
        doc,
        "This section provides step-by-step instructions for deploying and running the entire platform locally on Windows 10/11."
    )

    add_heading_2(doc, "14.1 Hardware & Software Prerequisites")
    add_bullet_point(doc, "Windows 10 or 11 (64-bit AMD64).", bold_prefix="Operating System: ")
    add_bullet_point(doc, "Python 3.10 to 3.14 (Verified on Python 3.14.0 64-bit).", bold_prefix="Python Runtime: ")
    add_bullet_point(doc, "Node.js >= 18.0.0 (npm >= 9.0.0).", bold_prefix="Node Runtime: ")
    add_bullet_point(doc, "8 GB RAM minimum (16 GB recommended); 2 GB free disk space.", bold_prefix="Hardware: ")

    add_heading_2(doc, "14.2 Environment Setup & Installation")
    add_code_block(
        doc,
        "# 1. Clone repository and navigate to root directory\n"
        "cd c:\\Users\\Likith T M\\Music\\erm\n\n"
        "# 2. Install Python backend dependencies\n"
        "pip install -r requirements.txt\n\n"
        "# 3. Install Frontend dependencies\n"
        "cd frontend\n"
        "npm install\n"
        "cd ..",
        title="ENVIRONMENT INSTALLATION COMMANDS"
    )

    add_heading_2(doc, "14.3 Launching the Services")
    add_body_paragraph(doc, "The platform requires two parallel processes (or terminal windows):")

    add_heading_3(doc, "Terminal 1: FastAPI Backend API Server")
    add_code_block(
        doc,
        "uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload\n\n"
        "# Service Base URL:      http://127.0.0.1:8000\n"
        "# Interactive Swagger:   http://127.0.0.1:8000/api/docs\n"
        "# ReDoc Documentation:   http://127.0.0.1:8000/api/redoc",
        title="LAUNCHING BACKEND SERVER"
    )

    add_heading_3(doc, "Terminal 2: React 19 Frontend Web Dashboard")
    add_code_block(
        doc,
        "cd frontend\n"
        "npm run dev\n\n"
        "# Web UI URL:            http://127.0.0.1:5173",
        title="LAUNCHING FRONTEND WEB DASHBOARD"
    )

    add_heading_3(doc, "Launching Native Desktop Scripts")
    add_code_block(
        doc,
        "# Run Native OpenCV Live Webcam HUD:\n"
        "python live_face.py --fps 3.5\n\n"
        "# Run Native Microphone Audio Stream:\n"
        "python live_audio.py --threshold 0.003\n\n"
        "# Run Interactive Terminal Text REPL:\n"
        "python live_text.py",
        title="LAUNCHING NATIVE SCRIPTS"
    )

def build_section_15_legacy_codebases(doc):
    """Section 15: Legacy Codebases & Evolutionary Refactoring Journey."""
    add_heading_1(doc, "15. Legacy Codebases & Evolutionary Refactoring Journey")

    add_body_paragraph(
        doc,
        "The current production system is the culmination of refactoring, standardizing, and unifying multiple exploratory academic codebases "
        "and legacy repositories located within the workspace:"
    )

    leg_headers = ["Legacy Directory / Module", "Original Source / Heritage", "Architectural Limitation", "Production Engineering Refactoring"]
    leg_rows = [
        [
            "Emotion-Detection Text",
            "Initial exploratory text emotion classification research notebooks.",
            "Jupyter notebook format (.ipynb), global variables, un-typed functions, hardcoded paths.",
            "Refactored into core/text/model.py and preprocessor.py with HuggingFace pipeline, Pydantic schemas, and typed API."
        ],
        [
            "Multimodal-Emotion-Recognition-Audio",
            "Keras Sequential CNN experiments on German Emo-DB dataset.",
            "Heavy TensorFlow 1.x dependency, deprecated Keras APIs, incompatible with modern Python 3.14.",
            "Re-implemented as native PyTorch nn.Module with direct h5py binary weight transposition (zero TensorFlow dependency)."
        ],
        [
            "Multimodal-Emotion-Recognition-Face",
            "AlexNet facial feature extraction notebooks and Kinect capture scripts.",
            "Missing trained .pth weights file in upstream repo; incomplete image preprocessing pipeline.",
            "Constructed dedicated training/vision/ pipeline, trained AlexNet on RAVDESS disjoint actors, and exported model_image_new50.pth."
        ],
    ]
    add_styled_table(doc, leg_headers, leg_rows, col_widths=[1.5, 1.3, 1.7, 2.0])

def build_section_16_limitations_roadmap(doc):
    """Section 16: System Limitations, Ethical Boundaries, & Future Roadmap."""
    add_heading_1(doc, "16. System Limitations, Ethical Boundaries, & Future Roadmap")

    add_heading_2(doc, "16.1 Honest Evaluation of Known Limitations")
    add_bullet_point(
        doc,
        "The trained AlexNet classifier achieves 37.39% baseline accuracy (41.59% fine-tuned) on held-out RAVDESS actors. While substantially "
        "above the 14.28% random baseline, facial expression classification on unfamiliar identities in unconstrained webcam environments "
        "is subject to lighting and pose variance.",
        bold_prefix="Facial Emotion Accuracy on Unseen Faces: "
    )
    add_bullet_point(
        doc,
        "The acoustic Emo-DB CNN model predicts 4 emotions (neutral, angry, happy, sad) and does not predict disgust, fear, or surprise. "
        "These classes are correctly assigned 0.0 mass but limit acoustic coverage.",
        bold_prefix="Speech Model Class Coverage: "
    )
    add_bullet_point(
        doc,
        "The current system performs frame-level and window-level decision fusion. It does not yet incorporate long-range temporal LSTM or "
        "Transformer recurrence across multi-minute conversations.",
        bold_prefix="Temporal Sequence Modeling: "
    )

    add_heading_2(doc, "16.2 Affective Computing Ethics & Non-Claims")
    add_callout(
        doc,
        "NON-CLINICAL DISCLAIMER: The Multi-Modal Emotion Recognition System is an academic engineering integration designed for human-computer "
        "interaction research. It measures behavioral, linguistic, and acoustic expression correlates. It CANNOT diagnose psychological conditions, "
        "assess mental health, determine legal culpability, or detect deception. Model confidence reflects neural network activation certainty, "
        "NOT objective psychological truth.",
        title="ETHICAL BOUNDARY STATEMENT",
        alert_type="warn"
    )

    add_heading_2(doc, "16.3 Future Engineering Roadmap")
    add_bullet_point(doc, "Integrate fine-tuned Wav2Vec2.0 or Whisper acoustic encoders to expand speech classification to all 7 Ekman emotions.", bold_prefix="Full-Ontology Speech Model: ")
    add_bullet_point(doc, "Implement multi-frame temporal attention / LSTM fusion over video clip segments to capture subtle micro-expressions.", bold_prefix="Temporal Video Transformer: ")
    add_bullet_point(doc, "Upgrade face detection to MediaPipe Face Mesh (468 landmarks) to compute action units (FACS AUs) for enhanced explainability.", bold_prefix="Facial Action Units (FACS): ")
    add_bullet_point(doc, "Provide Docker containers with ONNX Runtime acceleration for sub-millisecond multi-platform edge inference.", bold_prefix="ONNX Edge Containerization: ")

def main():
    print("=" * 70)
    print("  MULTI-MODAL EMOTION RECOGNITION SYSTEM — WORD DOC GENERATOR")
    print("=" * 70)

    doc = docx.Document()
    add_header_footer(doc)

    print("[1/17] Building Cover Page...")
    build_cover_page(doc)

    print("[2/17] Building Table of Contents...")
    build_table_of_contents(doc)

    print("[3/17] Building Section 1: Executive Summary & Philosophy...")
    build_section_1_executive_summary(doc)

    print("[4/17] Building Section 2: End-to-End System Architecture...")
    build_section_2_system_architecture(doc)

    print("[5/17] Building Section 3: Canonical Emotion Ontology...")
    build_section_3_emotion_ontology(doc)

    print("[6/17] Building Section 4: Text Modality Engine...")
    build_section_4_text_engine(doc)

    print("[7/17] Building Section 5: Acoustic Speech Modality Engine...")
    build_section_5_audio_engine(doc)

    print("[8/17] Building Section 6: Facial Vision Modality Engine...")
    build_section_6_vision_engine(doc)

    print("[9/17] Building Section 7: Vision Training & Evaluation...")
    build_section_7_vision_training_evaluation(doc)

    print("[10/17] Building Section 8: Multi-Modal Decision Fusion Engine...")
    build_section_8_fusion_engine(doc)

    print("[11/17] Building Section 9: FastAPI Backend Architecture...")
    build_section_9_backend_api(doc)

    print("[12/17] Building Section 10: Frontend Application Architecture...")
    build_section_10_frontend_architecture(doc)

    print("[13/17] Building Section 11: Standalone Real-Time Native Programs...")
    build_section_11_native_programs(doc)

    print("[14/17] Building Section 12: Automated Testing & Verification Suite...")
    build_section_12_automated_testing(doc)

    print("[15/17] Building Section 13: Security & Privacy Architecture...")
    build_section_13_security_privacy(doc)

    print("[16/17] Building Section 14: Installation, Deployment, & Operations...")
    build_section_14_deployment_operations(doc)

    print("[17/17] Building Sections 15 & 16: Legacy Codebases, Limitations & Roadmap...")
    build_section_15_legacy_codebases(doc)
    build_section_16_limitations_roadmap(doc)

    output_path = os.path.abspath("Multi_Modal_Emotion_Recognition_System_Documentation.docx")
    print(f"\nSaving Word Document to: {output_path}...")
    doc.save(output_path)

    file_size_bytes = os.path.getsize(output_path)
    file_size_kb = file_size_bytes / 1024.0
    file_size_mb = file_size_kb / 1024.0

    print("=" * 70)
    print(f"SUCCESS! Document generated successfully.")
    print(f"File Path: {output_path}")
    print(f"File Size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
