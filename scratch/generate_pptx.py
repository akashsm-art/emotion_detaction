"""
Comprehensive PowerPoint Presentation Generator for Multi-Modal Emotion Recognition System (ERM).
Builds 'Multi_Modal_Emotion_Recognition_System_Presentation.pptx' corresponding directly to
'Multi_Modal_Emotion_Recognition_System_Documentation.docx'.
"""

import os
import sys
from datetime import datetime

import pptx
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor

# --- Color Constants ---
C_NAVY_DARK   = RGBColor(15, 23, 42)      # #0F172A (Slate 900)
C_NAVY_ROYAL  = RGBColor(30, 58, 138)     # #1E3A8A (Royal Navy)
C_BLUE_ACCENT = RGBColor(37, 99, 235)     # #2563EB (Vibrant Cobalt)
C_TEAL_ACCENT = RGBColor(13, 148, 136)    # #0D9488 (Teal)
C_BG_LIGHT    = RGBColor(248, 250, 252)   # #F8FAFC (Slate 50)
C_CARD_BG     = RGBColor(255, 255, 255)   # #FFFFFF (White)
C_CARD_ALT    = RGBColor(241, 245, 249)   # #F1F5F9 (Slate 100)
C_BORDER      = RGBColor(203, 213, 225)   # #CBD5E1 (Slate 300)
C_TEXT_DARK   = RGBColor(15, 23, 42)      # #0F172A (Header Text)
C_TEXT_BODY   = RGBColor(51, 65, 85)      # #334155 (Body Text)
C_TEXT_MUTED  = RGBColor(100, 116, 139)   # #64748B (Muted / Subtitle)
C_WHITE       = RGBColor(255, 255, 255)
C_AMBER       = RGBColor(217, 119, 6)     # #D97706 (Amber Alert)
C_GREEN       = RGBColor(22, 163, 74)     # #16A34A (Green Success)
C_CRIMSON     = RGBColor(220, 38, 38)     # #DC2626 (Red Critical)

def set_slide_background(slide, color):
    """Fill slide background with a solid color."""
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_header(slide, tag: str, title: str, subtitle: str = ""):
    """Create a standardized top header with category pill, slide title, and subtitle."""
    tag_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.32))
    tf_tag = tag_box.text_frame
    tf_tag.word_wrap = True
    tf_tag.margin_left = tf_tag.margin_top = tf_tag.margin_right = tf_tag.margin_bottom = 0
    p_tag = tf_tag.paragraphs[0]
    p_tag.text = tag.upper()
    p_tag.font.name = "Arial"
    p_tag.font.size = Pt(10)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_BLUE_ACCENT

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.68), Inches(11.7), Inches(0.55))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_top = tf_title.margin_right = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.name = "Arial"
    p_title.font.size = Pt(22)
    p_title.font.bold = True
    p_title.font.color.rgb = C_NAVY_DARK

    if subtitle:
        sub_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.22), Inches(11.7), Inches(0.38))
        tf_sub = sub_box.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_top = tf_sub.margin_right = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = "Arial"
        p_sub.font.size = Pt(11)
        p_sub.font.color.rgb = C_TEXT_MUTED

    footer_box = slide.shapes.add_textbox(Inches(0.8), Inches(7.05), Inches(11.733), Inches(0.28))
    tf_ftr = footer_box.text_frame
    tf_ftr.margin_left = tf_ftr.margin_top = tf_ftr.margin_right = tf_ftr.margin_bottom = 0
    p_ftr = tf_ftr.paragraphs[0]
    p_ftr.text = f"Multi-Modal Emotion Recognition Platform (ERM)  |  {tag}  |  Technical Architecture Manual"
    p_ftr.font.name = "Arial"
    p_ftr.font.size = Pt(8.5)
    p_ftr.font.color.rgb = C_TEXT_MUTED

def add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER):
    """Draw a rounded rectangular card panel."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    if border_color:
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def add_stat_box(slide, left, top, width, height, stat_val, stat_title, subtitle=""):
    """Create a prominent numerical stat callout card."""
    add_card(slide, left, top, width, height, bg_color=C_CARD_BG, border_color=C_BORDER)

    tb = slide.shapes.add_textbox(left + Inches(0.15), top + Inches(0.12), width - Inches(0.3), height - Inches(0.24))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_top = tf.margin_right = tf.margin_bottom = 0

    p_val = tf.paragraphs[0]
    p_val.text = str(stat_val)
    p_val.font.name = "Arial"
    p_val.font.size = Pt(26)
    p_val.font.bold = True
    p_val.font.color.rgb = C_BLUE_ACCENT

    p_lbl = tf.add_paragraph()
    p_lbl.text = stat_title
    p_lbl.font.name = "Arial"
    p_lbl.font.size = Pt(10.5)
    p_lbl.font.bold = True
    p_lbl.font.color.rgb = C_NAVY_DARK
    p_lbl.space_before = Pt(2)

    if subtitle:
        p_sub = tf.add_paragraph()
        p_sub.text = subtitle
        p_sub.font.name = "Arial"
        p_sub.font.size = Pt(8.5)
        p_sub.font.color.rgb = C_TEXT_MUTED
        p_sub.space_before = Pt(2)

def add_table_custom(slide, left, top, width, height, headers, rows, col_widths=None):
    """Create a clean, styled table on the slide."""
    num_rows = len(rows) + 1
    num_cols = len(headers)
    table_shape = slide.shapes.add_table(num_rows, num_cols, left, top, width, height)
    table = table_shape.table

    if col_widths and len(col_widths) == num_cols:
        for idx, w in enumerate(col_widths):
            table.columns[idx].width = Inches(w)

    for idx, heading in enumerate(headers):
        cell = table.cell(0, idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = C_NAVY_ROYAL
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = cell.text_frame.paragraphs[0]
        p.text = heading
        p.font.name = "Arial"
        p.font.size = Pt(9.5)
        p.font.bold = True
        p.font.color.rgb = C_WHITE

    for r_idx, row_data in enumerate(rows):
        bg = C_CARD_ALT if r_idx % 2 == 1 else C_CARD_BG
        for c_idx, val in enumerate(row_data):
            cell = table.cell(r_idx + 1, c_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            p = cell.text_frame.paragraphs[0]
            p.text = str(val)
            p.font.name = "Arial"
            p.font.size = Pt(9.0)
            p.font.color.rgb = C_TEXT_DARK

    return table_shape

def add_bullet_list(tf, items):
    """Populate a text frame with clean bullet points."""
    for idx, item in enumerate(items):
        p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
        p.space_before = Pt(3)
        p.space_after = Pt(3)
        p.line_spacing = 1.15
        
        if isinstance(item, tuple):
            lead, text = item
            r_lead = p.add_run()
            r_lead.text = f"{lead} "
            r_lead.font.name = "Arial"
            r_lead.font.size = Pt(10)
            r_lead.font.bold = True
            r_lead.font.color.rgb = C_NAVY_DARK

            r_txt = p.add_run()
            r_txt.text = text
            r_txt.font.name = "Arial"
            r_txt.font.size = Pt(10)
            r_txt.font.color.rgb = C_TEXT_BODY
        else:
            r_txt = p.add_run()
            r_txt.text = item
            r_txt.font.name = "Arial"
            r_txt.font.size = Pt(10)
            r_txt.font.color.rgb = C_TEXT_BODY

# ==============================================================================
# SLIDE BUILDERS (24 SLIDES TOTAL)
# ==============================================================================

def build_slide_1(prs):
    """Slide 1: Dark Executive Title Slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_NAVY_DARK)

    # Accent decorative bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(1.2), Inches(0.12), Inches(4.5))
    bar.fill.solid()
    bar.fill.fore_color.rgb = C_TEAL_ACCENT
    bar.line.fill.background()

    # Title box
    tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(11.2), Inches(4.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p_tag = tf.paragraphs[0]
    p_tag.text = "SYSTEM ARCHITECTURE & TECHNICAL SPECIFICATION"
    p_tag.font.name = "Arial"
    p_tag.font.size = Pt(12)
    p_tag.font.bold = True
    p_tag.font.color.rgb = C_TEAL_ACCENT

    p_title = tf.add_paragraph()
    p_title.text = "Multi-Modal Emotion Recognition Platform"
    p_title.font.name = "Arial"
    p_title.font.size = Pt(36)
    p_title.font.bold = True
    p_title.font.color.rgb = C_WHITE
    p_title.space_before = Pt(8)

    p_sub = tf.add_paragraph()
    p_sub.text = "A Production-Grade, Local-Only Affective Computing Platform Integrating RoBERTa NLP, 2D CNN Speech Acoustics, AlexNet Vision, Confidence-Aware Late Fusion, FastAPI, and React 19"
    p_sub.font.name = "Arial"
    p_sub.font.size = Pt(14)
    p_sub.font.color.rgb = RGBColor(148, 163, 184) # Slate 400
    p_sub.space_before = Pt(12)

    # Feature badges box
    p_badges = tf.add_paragraph()
    p_badges.space_before = Pt(28)
    badges = ["[ ZERO FAKE AI POLICY ]", "[ 100% LOCALHOST PRIVACY ]", "[ CONFIDENCE-AWARE FUSION ]", "[ 21/21 TESTS PASSING ]"]
    for b in badges:
        r = p_badges.add_run()
        r.text = f"{b}   "
        r.font.name = "Consolas"
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = C_BLUE_ACCENT

    # Footer metadata
    p_meta = tf.add_paragraph()
    p_meta.text = f"Version 1.0.0 (Production)  |  FastAPI + PyTorch 2.4 + React 19  |  {datetime.now().strftime('%B %Y')}"
    p_meta.font.name = "Arial"
    p_meta.font.size = Pt(10)
    p_meta.font.color.rgb = RGBColor(100, 116, 139)
    p_meta.space_before = Pt(20)

def build_slide_2(prs):
    """Slide 2: Executive Summary & Affective Challenge."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "1. Executive Summary", "The Affective Computing Challenge", "Understanding complex human emotional states across non-synchronous sensory modalities")

    # Card 1: The Problem
    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(3.6))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(3.2))
    tf1 = tb1.text_frame
    tf1.word_wrap = True
    p1 = tf1.paragraphs[0]
    p1.text = "The Multi-Modal Emotion Problem"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    bullets1 = [
        ("Asynchronous Signals:", "Human emotions manifest across voice tremors, facial expressions, and vocabulary at differing time horizons."),
        ("Channel Discordance:", "Sarcasm, social politeness, and emotional masking frequently produce opposing cues between text and vision."),
        ("Noisy Sensor Inputs:", "Microphones capture room reverberation; webcams suffer from lighting shifts, head poses, and occlusion."),
        ("Scientific Need:", "Requires principled mathematical decision fusion rather than simplistic majority voting or heuristic guessing."),
    ]
    add_bullet_list(tf1, bullets1)

    # Card 2: The Solution
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(3.6))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(3.2))
    tf2 = tb2.text_frame
    tf2.word_wrap = True
    p2 = tf2.paragraphs[0]
    p2.text = "The ERM Architectural Solution"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_ACCENT

    bullets2 = [
        ("Independent Deep Ensembles:", "Fine-tuned RoBERTa for text, 2D CNN acoustic model on MFCCs, and AlexNet for facial landmarks."),
        ("Universal Ekman Ontology:", "Standardizes all disparate outputs onto canonical 7-class schema with 2D Circumplex coordinates."),
        ("Confidence-Aware Fusion:", "Weights each active channel dynamically by its neural softmax certainty, calculating inter-modal concordance."),
        ("Edge-First Deployment:", "100% offline inference on 127.0.0.1 with zero cloud latency and zero privacy leakage."),
    ]
    add_bullet_list(tf2, bullets2)

    # Stat boxes bottom
    add_stat_box(slide, Inches(0.8), Inches(5.6), Inches(2.7), Inches(1.2), "3 Modalities", "Text, Audio, Vision", "Native neural inference")
    add_stat_box(slide, Inches(3.8), Inches(5.6), Inches(2.7), Inches(1.2), "7 Emotions", "Canonical Ekman", "Universal ontology")
    add_stat_box(slide, Inches(6.8), Inches(5.6), Inches(2.7), Inches(1.2), "100% Local", "Zero Cloud APIs", "Strict edge privacy")
    add_stat_box(slide, Inches(9.8), Inches(5.6), Inches(2.7), Inches(1.2), "21 / 21 Tests", "PyTest Suite", "Full QA verification")

def build_slide_3(prs):
    """Slide 3: Core Operating Principle: Strict Zero Fake AI Policy."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "1. Executive Summary", "Core Operating Principle: Strict Zero Fake AI Policy", "Absolute transparency and genuine neural inference without heuristics or simulated outputs")

    cards_data = [
        ("1. 100% Genuine Forward Passes", C_BLUE_ACCENT, [
            ("No Synthetic Heuristics:", "Every prediction originates exclusively from deep neural network tensor operations (PyTorch/Transformers)."),
            ("No Mocked Distributions:", "Refuses to output pseudo-random numbers or hard-coded uniform vectors."),
            ("Real Compute:", "Executes real transformer attention matrices, 2D convolutions, and Softmax layers."),
        ]),
        ("2. Transparent Failure Signaling", C_AMBER, [
            ("Honest Error Gating:", "If model weights are missing or unreadable, the system marks the modality available=False."),
            ("Explicit Status Messages:", "Returns descriptive blocked messages (e.g. 'VISION INFERENCE BLOCKED — WEIGHTS MISSING')."),
            ("Zero Silent Fallbacks:", "Never silently substitutes mock classifications when an engine fails."),
        ]),
        ("3. Zero-Mass Invariant Gating", C_GREEN, [
            ("Speech Model Scope:", "The 2D CNN acoustic model natively predicts 4 emotions (neutral, angry, happy, sad)."),
            ("Strict 0.0000 Assignment:", "Unsupported classes (disgust, fear, surprise) receive strictly zero probability mass."),
            ("No Smoothing Artifacts:", "Never artificially injects uniform probabilities into unpredicted classes."),
        ]),
        ("4. Confidence as Evidence Prior", C_NAVY_ROYAL, [
            ("Epistemic Uncertainty:", "Softmax certainty reflects neural pattern alignment, acting as a dynamic weighting prior."),
            ("Non-Diagnostic Boundary:", "Explicitly rejected as clinical diagnosis, psychological ground truth, or lie detection."),
            ("Objective Attribution:", "Confidence modulates fusion influence without making unsupported assertions."),
        ]),
    ]

    card_w = Inches(5.7)
    card_h = Inches(2.4)
    positions = [
        (Inches(0.8), Inches(1.8)),
        (Inches(6.8), Inches(1.8)),
        (Inches(0.8), Inches(4.4)),
        (Inches(6.8), Inches(4.4)),
    ]

    for idx, (title, color, bullets) in enumerate(cards_data):
        x, y = positions[idx]
        add_card(slide, x, y, card_w, card_h)
        tb = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.15), card_w - Inches(0.4), card_h - Inches(0.3))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color

        add_bullet_list(tf, bullets)

def build_slide_4(prs):
    """Slide 4: End-to-End System Topology & Dataflow."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "2. System Architecture", "End-to-End System Topology & Layered Architecture", "Decoupled 4-tier pipeline coordinating ingestion, deep ensembles, fusion math, and presentation")

    # 4 Flow Columns
    cols = [
        ("Layer 1: Ingestion", C_NAVY_ROYAL, [
            "Multipart/Form-Data Parser",
            "MIME & File Extension Checks",
            "Payload Limits: 25MB Audio, 10MB Image, 1000 Chars",
            "Ephemeral OS Temp Files",
            "Automatic Unlinking in Finally Blocks",
        ]),
        ("Layer 2: Neural Ensembles", C_BLUE_ACCENT, [
            "Text: DistilRoBERTa-base (~82M parameters, 6 layers)",
            "Audio: 4-Layer 2D CNN (39-channel MFCC spectrograms)",
            "Vision: AlexNet 7-Class + Multi-Pass Haar Cascades",
            "Parallel Model Execution",
            "Isolated ModalityResult Output",
        ]),
        ("Layer 3: Ontology & Fusion", C_TEAL_ACCENT, [
            "Ekman 7-Class Normalization",
            "Russell 2D Circumplex Mapping",
            "Modality Availability Gating",
            "Confidence Weight: w_m = W_m*C_m",
            "Cosine Concordance & Conflict",
            "Explainability Report Generator",
        ]),
        ("Layer 4: Presentation", C_NAVY_DARK, [
            "FastAPI Async REST Service",
            "React 19 + Vite Web Dashboard",
            "Web Audio API Real-Time FFT",
            "live_face.py OpenCV Native HUD",
            "live_audio.py Mic Terminal",
            "live_text.py Interactive REPL",
        ]),
    ]

    col_w = Inches(2.75)
    gap = Inches(0.24)
    start_x = Inches(0.8)

    for i, (title, color, items) in enumerate(cols):
        x = start_x + i * (col_w + gap)
        add_card(slide, x, Inches(1.8), col_w, Inches(4.9))

        tb = slide.shapes.add_textbox(x + Inches(0.15), Inches(1.95), col_w - Inches(0.3), Inches(4.6))
        tf = tb.text_frame
        tf.word_wrap = True

        p_hdr = tf.paragraphs[0]
        p_hdr.text = title
        p_hdr.font.name = "Arial"
        p_hdr.font.size = Pt(12)
        p_hdr.font.bold = True
        p_hdr.font.color.rgb = color

        for item in items:
            p = tf.add_paragraph()
            p.text = f"• {item}"
            p.font.name = "Arial"
            p.font.size = Pt(9.5)
            p.font.color.rgb = C_TEXT_BODY
            p.space_before = Pt(6)

def build_slide_5(prs):
    """Slide 5: Canonical Emotion Ontology & 7-Class Schema."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "3. Emotion Ontology", "Canonical Emotion Ontology & 7-Class Schema", "Unifying disparate dataset vocabularies onto the standardized Ekman affective schema")

    headers = ["Canonical Emotion", "Facial Action Cues", "Acoustic / Prosodic Markers", "Linguistic / Text Patterns"]
    rows = [
        ["happy", "Lip corner pull (AU12), cheek raise (AU6), crow's feet", "Elevated pitch, pitch variance, bright timbre", "delighted, joyful, thrilled, celebration, amazing"],
        ["surprise", "Brow raise (AU1+2), upper lid raise (AU5), jaw drop", "Sudden fundamental pitch jump, sharp energy attack", "unexpected, stunned, amazed, astonished, what!"],
        ["neutral", "Relaxed facial muscles, mouth closed, unwrinkled brow", "Moderate speech rate, stable pitch, balanced cadence", "factual statements, reports, informative, balanced"],
        ["disgust", "Nose wrinkle (AU9), upper lip raise (AU10), narrowed eyes", "Guttural phonation, lower pitch, vocal fry, brief pauses", "revolting, gross, sickening, awful, distasteful"],
        ["angry", "Brow lowerer (AU4), upper lid raise, lip tightener (AU23)", "High acoustic intensity (RMS), accelerated rate, harsh timbre", "furious, outrage, hate, intolerable, frustrated"],
        ["fear", "Brow raise and knit (AU1+2+4), eye widen (AU5), tense lips", "High pitch, vocal jitter, tremor, breathiness", "terrified, panicking, dread, frightened, dangerous"],
        ["sad", "Inner brow raise (AU1), brow lower (AU4), lip corner depressor", "Muffled acoustic energy, low pitch, extended pauses", "devastated, heartbroken, lonely, hopeless, sorrow"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[1.5, 3.4, 3.4, 3.433])

def build_slide_6(prs):
    """Slide 6: Russell's 2D Circumplex Model of Affect."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "3. Emotion Ontology", "Russell's 2D Circumplex Model of Affect", "Mapping discrete emotional states into continuous Valence-Arousal dimensional space")

    # Table on Left
    headers = ["Emotion Class", "Valence (V)", "Arousal (A)", "Circumplex Quadrant"]
    rows = [
        ["happy", "+0.85", "+0.60", "Q1: High Valence, High Arousal"],
        ["surprise", "+0.25", "+0.85", "Q1: Mild Valence, Peak Arousal"],
        ["neutral", " 0.00", " 0.00", "Origin: Emotional Equilibrium"],
        ["disgust", "-0.60", "+0.35", "Q2: Low Valence, Mild Arousal"],
        ["angry", "-0.65", "+0.80", "Q2: Low Valence, Peak Arousal"],
        ["fear", "-0.70", "+0.75", "Q2: Lowest Valence, High Arousal"],
        ["sad", "-0.80", "-0.45", "Q3: Lowest Valence, Low Arousal"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(6.2), Inches(4.8), headers, rows, col_widths=[1.5, 1.1, 1.1, 2.5])

    # Mathematical Explanations on Right
    add_card(slide, Inches(7.3), Inches(1.8), Inches(5.233), Inches(4.8))
    tb = slide.shapes.add_textbox(Inches(7.5), Inches(2.0), Inches(4.833), Inches(4.4))
    tf = tb.text_frame
    tf.word_wrap = True

    p_title = tf.paragraphs[0]
    p_title.text = "Mathematical Formulation"
    p_title.font.name = "Arial"
    p_title.font.size = Pt(14)
    p_title.font.bold = True
    p_title.font.color.rgb = C_BLUE_ACCENT

    bullets = [
        ("Valence Dimension (V):", "Quantifies psychological pleasure vs displeasure on a continuous scale from -1.0 (severe aversion) to +1.0 (peak joy)."),
        ("Arousal Dimension (A):", "Quantifies neurophysiological activation from -1.0 (lethargy, sleepiness) to +1.0 (intense fight-or-flight mobilization)."),
        ("Expectation Equations:", "Aggregate coordinates are computed as the expected value over the 7-class probability vector P:"),
    ]
    add_bullet_list(tf, bullets)

    # Formula callout
    p_eq = tf.add_paragraph()
    p_eq.text = (
        "\n  V_aggregate = Σ P(e) · Valence(e)\n"
        "  A_aggregate = Σ P(e) · Arousal(e)"
    )
    p_eq.font.name = "Consolas"
    p_eq.font.size = Pt(11)
    p_eq.font.bold = True
    p_eq.font.color.rgb = C_NAVY_DARK

    p_note = tf.add_paragraph()
    p_note.text = "\nEnables direct spatial visualization on the React frontend 2D Circumplex canvas."
    p_note.font.name = "Arial"
    p_note.font.size = Pt(9.5)
    p_note.font.color.rgb = C_TEXT_MUTED

def build_slide_7(prs):
    """Slide 7: Cross-Dataset Label Mapping & Fidelity Tracking."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "3. Emotion Ontology", "Cross-Dataset Label Mapping & Fidelity Tracking", "Preserving semantic fidelity across disparate academic datasets with strict tracking")

    # Table of mappings
    headers = ["Source Dataset / Domain", "Raw Label Input", "Canonical Ekman Target", "Mapping Fidelity", "Fidelity Definition"]
    rows = [
        ["German Emo-DB (Speech)", "wut, freude, trauer, angst", "angry, happy, sad, fear", "DIRECT (100%)", "Exact 1-to-1 semantic equivalence"],
        ["German Emo-DB (Speech)", "langeweile (boredom)", "neutral", "APPROXIMATE", "Nuance shift toward low-arousal baseline"],
        ["Polish Speech Dataset", "an, di, fe, ha, ne, sa, su", "angry, disgust, fear, happy...", "DIRECT (100%)", "Standardized two-letter abbreviations"],
        ["RAVDESS Video Frames", "01 (neutral), 03 (happy), 05 (angry)...", "neutral, happy, angry...", "DIRECT (100%)", "Official RAVDESS emotional index"],
        ["RAVDESS Video Frames", "02 (calm)", "neutral", "APPROXIMATE", "Calm collapsed into neutral (original design)"],
        ["Twitter / BERT Lexicons", "love, optimism", "happy", "APPROXIMATE", "Positive high-valence affect mapping"],
        ["Twitter / BERT Lexicons", "trust", "neutral", "LOSSY", "Broad reduction of interpersonal nuance"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[2.1, 2.5, 2.1, 1.8, 3.233])

def build_slide_8(prs):
    """Slide 8: Modality Engine 1 — Text Emotion Recognition."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "4. Modality Engine: Text", "Text Modality Engine: DistilRoBERTa Sequence Classifier", "Deep linguistic emotion recognition via distilled transformer architecture")

    # 3 Feature Cards
    cards = [
        ("Transformer Architecture", C_NAVY_ROYAL, [
            ("Backbone:", "j-hartmann/emotion-english-distilroberta-base (~82M parameters)."),
            ("Design:", "6 Transformer encoder layers, 768 hidden units, 12 attention heads."),
            ("Head:", "Linear classification head over [CLS] token pooling into 7 emotion logits."),
            ("Latency:", "15–35 ms on modern x86 CPU; ~4 ms on local CUDA GPU."),
        ]),
        ("Preprocessing Pipeline", C_BLUE_ACCENT, [
            ("Cleaning:", "Regex removal of URLs, email addresses, and Twitter user mentions."),
            ("Emoji Preservation:", "Retains and processes emotional emojis expressing sentiment intensity."),
            ("Length Boundary:", "Hard cap at 512 characters to prevent buffer overflow attacks."),
            ("Salient Keywords:", "Extracts salient emotional tokens (e.g. 'thrilled', 'hopeless')."),
        ]),
        ("Output Schema & Telemetry", C_TEAL_ACCENT, [
            ("Schema:", "Returns standardized ModalityResult Pydantic contract."),
            ("Distribution:", "Softmax probabilities across all 7 Ekman core classes."),
            ("Circumplex:", "Expected Valence and Arousal coordinates."),
            ("Explainability:", "Provides word count and extracted affective keywords in metadata."),
        ]),
    ]

    card_w = Inches(3.75)
    gap = Inches(0.24)
    for i, (title, color, bullets) in enumerate(cards):
        x = Inches(0.8) + i * (card_w + gap)
        add_card(slide, x, Inches(1.8), card_w, Inches(4.9))

        tb = slide.shapes.add_textbox(x + Inches(0.2), Inches(2.0), card_w - Inches(0.4), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color

        add_bullet_list(tf, bullets)

def build_slide_9(prs):
    """Slide 9: Modality Engine 2 — Speech Acoustic Processing."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "5. Modality Engine: Speech", "Acoustic Feature Extraction: 39-Channel MFCC Spectrograms", "Linguistic-independent acoustic feature engineering capturing vocal tract and prosodic dynamics")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Signal Preprocessing & Normalization"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    b1 = [
        ("Sampling Rate:", "Standardized to 16,000 Hz single-channel (mono) 16-bit PCM."),
        ("Multi-Channel Downmixing:", "Stereo or multi-channel audio is averaged across channels: sig = np.mean(sig, axis=1)."),
        ("Fixed Temporal Window:", "Calibrated to 2.0 seconds (32,000 samples). Shorter signals are zero-padded; longer signals are centrally cropped."),
        ("Signal Bounds:", "Float audio is clipped to [-1.0, 1.0] and converted to integer int16 scaling."),
    ]
    add_bullet_list(tf1, b1)

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "39 MFCC Spectrograms & Acoustic Descriptors"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_ACCENT

    b2 = [
        ("13 Static MFCCs:", "Captures spectral envelope and vocal tract filter resonances."),
        ("13 Delta Coefficients:", "Captures first-order velocity of spectral trajectory over time."),
        ("13 Delta-Delta Coefficients:", "Captures second-order acceleration of spectral transitions."),
        ("Temporal Resolution:", "Yields an input feature tensor of shape (1, 198, 39) across 198 time frames."),
        ("Acoustic Metadata:", "Computes RMS energy, zero-crossing rate (ZCR), and pitch autocorrelation for explainability."),
    ]
    add_bullet_list(tf2, b2)

def build_slide_10(prs):
    """Slide 10: Speech 2D CNN — Pure PyTorch Reimplementation."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "5. Modality Engine: Speech", "PyTorch 2D CNN Acoustic Architecture (Zero TensorFlow)", "Eliminating TensorFlow runtime bloat through native PyTorch layer reconstruction and HDF5 weight loading")

    # Table of CNN Layers
    headers = ["Layer", "PyTorch Equivalent", "Kernel / Stride", "Activation", "Output Shape (B, C, H, W)"]
    rows = [
        ["Conv 1", "nn.Conv2d(1, 8) + BatchNorm", "13x13, stride (1,1)", "ReLU", "(B, 8, 186, 27)"],
        ["Conv 2", "nn.Conv2d(8, 8) + BatchNorm + MaxPool", "13x13 + Pool(2,1)", "ReLU", "(B, 8, 87, 15)"],
        ["Conv 3", "nn.Conv2d(8, 8) + BatchNorm", "13x13, stride (1,1)", "ReLU", "(B, 8, 75, 3)"],
        ["Conv 4", "nn.Conv2d(8, 8) + BatchNorm + MaxPool", "2x2 + Pool(2,1)", "ReLU", "(B, 8, 37, 2)"],
        ["Flatten", "x.permute(0, 2, 3, 1).contiguous().view()", "Replicates Keras memory layout", "None", "(B, 592)"],
        ["Dense 1", "nn.Linear(592, 64) + BatchNorm + Dropout", "592 in -> 64 out", "ReLU", "(B, 64)"],
        ["Dense 2", "nn.Linear(64, 4) + Softmax", "64 in -> 4 out", "Softmax", "(B, 4) [neutral, angry, happy, sad]"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[1.1, 3.2, 2.2, 1.2, 4.033])

def build_slide_11(prs):
    """Slide 11: Modality Engine 3 — Facial Vision Emotion Recognition."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "6. Modality Engine: Vision", "Multi-Pass OpenCV Haar Cascade Face Detector", "Robust facial region-of-interest extraction compensating for illumination, tilt, and shadows")

    passes = [
        ("Pass 1: Frontal Default Cascade", C_NAVY_ROYAL, [
            ("Classifier:", "haarcascade_frontalface_default.xml"),
            ("Parameters:", "scaleFactor=1.1, minNeighbors=4, minSize=(40, 40)"),
            ("Role:", "Rapidly detects clear, high-contrast frontal faces under standard studio or ambient lighting."),
            ("Latency:", "~8 ms per frame on CPU."),
        ]),
        ("Pass 2: CLAHE Contrast Recovery", C_BLUE_ACCENT, [
            ("Preprocessing:", "Converts image to grayscale + cv2.equalizeHist contrast normalization."),
            ("Role:", "Recovers facial features in dimly lit environments, heavy side-shadows, or back-lit cameras."),
            ("Robustness:", "Prevents dropped detections when subject moves away from direct light."),
        ]),
        ("Pass 3: Alt2 Frontal Cascade", C_TEAL_ACCENT, [
            ("Classifier:", "haarcascade_frontalface_alt2.xml"),
            ("Role:", "Tuned specifically for angled heads, slight profiles, tilted faces, and non-rigid expressions."),
            ("Crop Bounding:", "Expands detected face box by 10% and normalizes to 224x224 RGB tensor."),
        ]),
    ]

    card_w = Inches(3.75)
    gap = Inches(0.24)
    for i, (title, color, bullets) in enumerate(passes):
        x = Inches(0.8) + i * (card_w + gap)
        add_card(slide, x, Inches(1.8), card_w, Inches(4.9))

        tb = slide.shapes.add_textbox(x + Inches(0.2), Inches(2.0), card_w - Inches(0.4), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color

        add_bullet_list(tf, bullets)

def build_slide_12(prs):
    """Slide 12: AlexNet Deep Vision Architecture & Webcam Engine."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "6. Modality Engine: Vision", "AlexNet Deep Neural Vision Architecture & Live HUD", "7-Class convolutional neural network with decoupled real-time webcam frame processing")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "AlexNet 7-Class Deep CNN Backbone"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    b1 = [
        ("Input Tensor:", "224x224x3 RGB image normalized with ImageNet stats (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])."),
        ("Feature Extractor:", "5 Convolutional layers with ReLU activations and MaxPool2D(3x3, stride 2)."),
        ("Spatial Pooling:", "AdaptiveAvgPool2d((6, 6)) yielding a 9,216-dim spatial representation vector."),
        ("Classifier Head:", "Linear(9216->4096) -> ReLU -> Dropout(0.5) -> Linear(4096->4096) -> Linear(4096->7) -> Softmax(dim=1)."),
    ]
    add_bullet_list(tf1, b1)

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "Decoupled Live Webcam Architecture (live_face.py)"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_ACCENT

    b2 = [
        ("Frame-Rate Decoupling:", "OpenCV camera capture and rendering runs at 30 FPS; neural inference is throttled to 3.5 FPS to prevent CPU core saturation."),
        ("Dynamic Bounding Box:", "Changes border color dynamically (Green for Happy, Red for Angry, Purple for Fear, Yellow for Surprise)."),
        ("On-Screen HUD Meters:", "Renders live horizontal probability meters for all 7 Ekman classes directly on video frames."),
        ("Keyboard Telemetry:", "'q' to quit cleanly, 's' for snapshot capture, 'd' to toggle latency telemetry."),
    ]
    add_bullet_list(tf2, b2)

def build_slide_13(prs):
    """Slide 13: Vision Training Pipeline & Zero Data Leakage."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "7. Vision Training", "Vision Training Pipeline & Zero Data Leakage", "Reproducing AlexNet on RAVDESS with 100% disjoint actor partitioning")

    # Table of splits
    headers = ["Emotion Category", "Train Images (Actors 01-20)", "Train Share (%)", "Train Clips", "Validation Images (Actors 21-24)", "Val Share (%)", "Val Clips"]
    rows = [
        ["angry", "1,778", "14.06%", "160", "362", "13.81%", "32"],
        ["disgust", "1,812", "14.33%", "160", "366", "13.96%", "32"],
        ["fear", "1,623", "12.83%", "160", "338", "12.90%", "32"],
        ["happy", "1,674", "13.24%", "160", "332", "12.67%", "32"],
        ["neutral (incl. calm)", "2,510", "19.85%", "240", "533", "20.34%", "48"],
        ["sad", "1,657", "13.10%", "160", "371", "14.15%", "32"],
        ["surprise", "1,593", "12.60%", "160", "319", "12.17%", "32"],
        ["Total Clean Samples", "12,647", "100.00%", "1,200", "2,621", "100.00%", "240"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[1.8, 1.7, 1.4, 1.3, 1.8, 1.4, 2.333])

def build_slide_14(prs):
    """Slide 14: Vision Model Controlled Ablation Experiments."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "7. Vision Training", "Controlled Ablation Experiments & Selected Checkpoint", "Systematic evaluation of inverse-frequency class weighting across disjoint validation actors")

    headers = ["Experiment Run", "Loss Weight Strength (α)", "Optimizer & Schedule", "Validation Accuracy", "Macro-F1 Score", "Correct / 2,621"]
    rows = [
        ["Baseline Checkpoint", "0.00 (Unweighted CE)", "SGD (lr=0.016), StepLR, Frozen Conv", "37.39%", "0.3258", "980 / 2,621"],
        ["Ablation 1", "0.00 (Unweighted CE)", "AdamW, Cosine Annealing, conv5 unfrozen", "40.29%", "0.3562", "1,056 / 2,621"],
        ["Ablation 2", "0.25 (Mild Weighting)", "AdamW, Cosine Annealing, conv5 unfrozen", "41.70%", "0.3912", "1,093 / 2,621"],
        ["Selected Production Checkpoint", "0.50 (Inverse Frequency)", "AdamW, Cosine Annealing, conv5 unfrozen", "41.59%", "0.3956 (BEST)", "1,090 / 2,621"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(2.8), headers, rows, col_widths=[2.8, 2.0, 2.8, 1.4, 1.4, 1.333])

    # Rationale Callout Card below
    add_card(slide, Inches(0.8), Inches(4.8), Inches(11.733), Inches(1.9))
    tb = slide.shapes.add_textbox(Inches(1.0), Inches(4.95), Inches(11.333), Inches(1.6))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Model Selection Rationale: Prioritizing Balanced Generalization over Raw Accuracy"
    p.font.name = "Arial"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = C_BLUE_ACCENT

    bullets = [
        ("Macro-F1 Superiority:", "While weight strength 0.25 gained 3 additional correct frames (41.70% vs 41.59%), strength 0.50 produced the superior Macro-F1 score (0.3956 vs 0.3912)."),
        ("Happy Class Discrimination:", "Weight strength 0.50 achieved significantly higher Happy F1 (0.5781 vs 0.5441) and recall (0.4849 vs 0.4578), drastically reducing neutral confusion."),
        ("Scientific Decision:", "The 0.50 checkpoint (model_image_new50.pth) was retained as the production checkpoint for vision inference."),
    ]
    add_bullet_list(tf, bullets)

def build_slide_15(prs):
    """Slide 15: Per-Class Evaluation Metrics & Full Confusion Matrix."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "7. Vision Training", "Per-Class Evaluation Metrics & Confusion Matrix", "Honest quantitative evaluation across all 7 Ekman emotional classes on unseen actors")

    # Table 1: Metrics
    headers1 = ["Emotion Class", "Baseline Precision", "Selected Precision", "Baseline Recall", "Selected Recall", "Baseline F1", "Selected F1"]
    rows1 = [
        ["angry", "0.5545", "0.6569", "0.1685", "0.1851", "0.2585", "0.2888"],
        ["disgust", "0.5146", "0.6372", "0.2896", "0.3934", "0.3706", "0.4865"],
        ["fear", "0.4711", "0.5120", "0.1686", "0.3166", "0.2484", "0.3912"],
        ["happy", "0.6150", "0.7156", "0.4187", "0.4849", "0.4982", "0.5781"],
        ["neutral", "0.3236", "0.3521", "0.9381", "0.8199", "0.4812", "0.4927"],
        ["sad", "0.2731", "0.2806", "0.1752", "0.2534", "0.2135", "0.2663"],
        ["surprise", "0.2971", "0.2827", "0.1630", "0.2508", "0.2105", "0.2658"],
        ["Macro Average", "0.4356", "0.4910", "0.3317", "0.3863", "0.3258", "0.3956"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(6.0), Inches(4.8), headers1, rows1, col_widths=[1.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])

    # Table 2: Selected Confusion Matrix
    headers2 = ["True \\ Pred", "angr", "disg", "fear", "happ", "neut", "sad", "surp"]
    rows2 = [
        ["angry (362)", "67", "27", "3", "6", "153", "58", "48"],
        ["disgust (366)", "12", "144", "6", "6", "106", "88", "4"],
        ["fear (338)", "0", "9", "107", "2", "104", "61", "55"],
        ["happy (332)", "0", "0", "12", "161", "128", "1", "30"],
        ["neutral (533)", "0", "0", "15", "19", "437", "10", "52"],
        ["sad (371)", "3", "39", "10", "22", "189", "94", "14"],
        ["surprise (319)", "20", "7", "56", "9", "124", "23", "80"],
    ]
    add_table_custom(slide, Inches(7.1), Inches(1.8), Inches(5.433), Inches(4.8), headers2, rows2, col_widths=[1.233, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6])

def build_slide_16(prs):
    """Slide 16: Empirical Diagnostic Analysis & Visual Insights."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "7. Vision Training", "Empirical Diagnostic Analysis & Visual Findings", "Deep dive into model confusion dynamics and dataset composition effects")

    # Embed diagnostic plot if exists
    chart_path = os.path.abspath("reports/vision_happy_diagnostics.png")
    if os.path.isfile(chart_path):
        slide.shapes.add_picture(chart_path, Inches(0.8), Inches(1.8), width=Inches(6.2))

    # Diagnostic text on Right
    add_card(slide, Inches(7.3), Inches(1.8), Inches(5.233), Inches(4.9))
    tb = slide.shapes.add_textbox(Inches(7.5), Inches(2.0), Inches(4.833), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = "Key Empirical Diagnostic Findings"
    p.font.name = "Arial"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_BLUE_ACCENT

    bullets = [
        ("Neutral / Calm Over-Representation:", "Neutral combines RAVDESS code 01 (pure neutral) and code 02 (calm). Calm constitutes ~69% of neutral frames, pulling lower-intensity expressions toward neutral."),
        ("Disgust & Happy Distinctiveness:", "Happy (F1=0.5781) and Disgust (F1=0.4865) achieve the highest discrimination due to pronounced morphological cues (smiling mouth, wrinkled nose)."),
        ("Subtle Expression Confusion:", "Sadness and fear without exaggerated eye widening frequently register as low-arousal neutral across unfamiliar faces."),
        ("Controlled Domain Generalization:", "While laboratory accuracy is 41.59%, unconstrained webcam footage exhibits lighting variance, necessitating multi-pass face detection."),
    ]
    add_bullet_list(tf, bullets)

def build_slide_17(prs):
    """Slide 17: Confidence-Aware Multi-Modal Decision Fusion Engine."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "8. Decision Fusion", "Confidence-Aware Late Decision Fusion Engine", "Mathematical formulation for dynamically combining multi-modal probability vectors")

    # Left: Equations
    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Mathematical Formulation"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    b1 = [
        ("Modality Availability Gate:", "A_m in {0, 1}. Modality m is eligible if and only if available==True and status=='ok'."),
        ("Effective Weight Calculation:", "Combines configured prior weight W_m and neural model confidence C_m:"),
    ]
    add_bullet_list(tf1, b1)

    p_eq1 = tf1.add_paragraph()
    p_eq1.text = (
        "  w_tilde_m = W_m · max(0.0001, C_m)\n"
        "  w_m = w_tilde_m / Σ w_tilde_k"
    )
    p_eq1.font.name = "Consolas"
    p_eq1.font.size = Pt(11)
    p_eq1.font.bold = True
    p_eq1.font.color.rgb = C_BLUE_ACCENT
    p_eq1.space_before = Pt(4)

    b1_post = [
        ("Fused Probability Synthesis:", "Weighted linear combination across active modalities:"),
    ]
    add_bullet_list(tf1, b1_post)

    p_eq2 = tf1.add_paragraph()
    p_eq2.text = "  P_fused(e) = Σ w_m · P_m(e), ∀ e ∈ Ekman"
    p_eq2.font.name = "Consolas"
    p_eq2.font.size = Pt(11)
    p_eq2.font.bold = True
    p_eq2.font.color.rgb = C_NAVY_DARK
    p_eq2.space_before = Pt(4)

    # Right: Operational Rules
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "Core Fusion Invariants"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_ACCENT

    b2 = [
        ("Dynamic Confidence Scaling:", "A modality with high epistemic confidence (e.g. text=0.95) naturally outweighs a noisy, uncertain channel (e.g. audio=0.35)."),
        ("Prior Weight Overrides:", "Admins can configure modality priors W_text, W_audio, W_vision to reflect application context."),
        ("Strict Normalization:", "Probability distribution is re-normalized to guarantee exact sum = 1.0000 with zero floating-point drift."),
        ("Zero-Mass Preservation:", "Classes unsupported by audio (disgust, fear, surprise) inherit probabilities solely from text and vision."),
    ]
    add_bullet_list(tf2, b2)

def build_slide_18(prs):
    """Slide 18: Cross-Modal Concordance & Conflict Detection."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "8. Decision Fusion", "Cross-Modal Concordance & Conflict Detection", "Quantifying emotional discordance, affective conflict, and synthesizing explainability rationales")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Concordance & Conflict Mathematics"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    b1 = [
        ("Pairwise Cosine Similarity:", "Evaluates alignment between probability vectors P_1 and P_2 across all active pairs:"),
    ]
    add_bullet_list(tf1, b1)

    p_eq = tf1.add_paragraph()
    p_eq.text = (
        "  Sim(P_1, P_2) = (P_1 · P_2) / (||P_1||_2 · ||P_2||_2)\n\n"
        "  ConcordanceScore = mean(Sim(P_i, P_j))\n"
        "  ConflictScore = 1.0 - ConcordanceScore"
    )
    p_eq.font.name = "Consolas"
    p_eq.font.size = Pt(10.5)
    p_eq.font.bold = True
    p_eq.font.color.rgb = C_BLUE_ACCENT
    p_eq.space_before = Pt(4)

    b1_post = [
        ("Conflict Threshold (θ = 0.40):", "If ConflictScore >= 0.40 or if top emotions diverge, has_conflict is flagged True."),
        ("Divergent Pair Isolation:", "Identifies exact opposing channels (e.g. text: happy vs audio: sad)."),
    ]
    add_bullet_list(tf1, b1_post)

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "Explainability & Attribution Engine"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_TEAL_ACCENT

    b2 = [
        ("Primary Modality Identification:", "Identifies which sensory channel exerted the highest mathematical influence on the final result."),
        ("Natural Language Rationale:", "Synthesizes human-readable explanations summarizing the decision driver and confidence breakdown."),
        ("Salient Feature Attribution:", "Passes through text keywords (e.g. 'furious'), vocal pitch/energy, and facial box geometry."),
        ("Clinical Non-Judgment:", "Phrases conflicts objectively without psychological speculation or deception accusation."),
    ]
    add_bullet_list(tf2, b2)

def build_slide_19(prs):
    """Slide 19: All 7 Modality Permutations Matrix."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "8. Decision Fusion", "Comprehensive Modality Permutations Matrix", "Verified deterministic operation across all 2^3 - 1 = 7 active sensory combinations")

    headers = ["Permutation", "Active Modalities", "Fusion Mode", "Weight Attribution Dynamics", "Concordance Behavior"]
    rows = [
        ["1. Text Only", "Text", "single_modality", "w_text = 1.00 (Single channel bypass)", "Concordance = 1.0000 (N/A)"],
        ["2. Audio Only", "Audio", "single_modality", "w_audio = 1.00 (Single channel bypass)", "Concordance = 1.0000 (N/A)"],
        ["3. Vision Only", "Vision", "single_modality", "w_vision = 1.00 (Single channel bypass)", "Concordance = 1.0000 (N/A)"],
        ["4. Text + Audio", "Text, Audio", "multimodal", "w_m = W_m * C_m / (W_t*C_t + W_a*C_a)", "Pairwise cosine similarity: Text vs Audio"],
        ["5. Text + Vision", "Text, Vision", "multimodal", "w_m = W_m * C_m / (W_t*C_t + W_v*C_v)", "Pairwise cosine similarity: Text vs Vision"],
        ["6. Audio + Vision", "Audio, Vision", "multimodal", "w_m = W_m * C_m / (W_a*C_a + W_v*C_v)", "Pairwise cosine similarity: Audio vs Vision"],
        ["7. Tri-Modal", "Text, Audio, Vision", "multimodal", "Tri-channel confidence-adjusted weights", "Mean across all 3 pairwise comparisons"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[1.6, 2.0, 1.8, 3.3, 3.033])

def build_slide_20(prs):
    """Slide 20: FastAPI Backend Architecture & API Specification."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "9. Backend API", "FastAPI Service Architecture & REST Specification", "Asynchronous ASGI service with singleton lifecycle caching, input sanitization, and typed contracts")

    headers = ["HTTP Method & Route", "Input Payload / Format", "Response Model", "Latency", "Functional Scope"]
    rows = [
        ["GET /api/health", "None", "HealthStatus", "< 2 ms", "Returns service readiness, active compute device, and model states"],
        ["POST /api/predict/text", "JSON: {'text': string}", "ModalityResult", "15–35 ms", "DistilRoBERTa NLP classification, keyword extraction"],
        ["POST /api/predict/audio", "Multipart file (WAV/MP3 <= 25MB)", "ModalityResult", "40–90 ms", "Acoustic 2D CNN inference over 39-channel MFCCs"],
        ["POST /api/predict/vision", "Multipart file (JPG/PNG <= 10MB)", "ModalityResult", "30–60 ms", "Multi-pass Haar face detection and AlexNet classification"],
        ["POST /api/predict/multimodal", "Multipart: text, audio file, image file", "FusedEmotionResult", "60–120 ms", "Full multi-modal late decision fusion across uploaded media"],
        ["POST /api/predict/fuse", "JSON: {'results': {modality: ModalityResult}}", "FusedEmotionResult", "< 5 ms", "Real-time pre-evaluated fusion for live streaming UI coordinators"],
    ]
    add_table_custom(slide, Inches(0.8), Inches(1.8), Inches(11.733), Inches(4.8), headers, rows, col_widths=[2.4, 2.4, 1.8, 1.1, 4.033])

def build_slide_21(prs):
    """Slide 21: React 19 + TypeScript Frontend Dashboard."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "10. Frontend UI", "React 19 + TypeScript Web Application", "Real-time glassmorphic dashboard with Web Audio API waveform visualizer and reactive hooks")

    cards = [
        ("Reactive Stream Hooks", C_NAVY_ROYAL, [
            ("useLiveText:", "Debounced text classifier with sample presets and cancellation."),
            ("useLiveAudio:", "Web Audio API AudioContext + AnalyserNode capturing mic stream, computing volume RMS, and audio FFT waveforms."),
            ("useLiveWebcam:", "HTML5 video stream capture, hidden canvas frame extraction, face presence gating."),
            ("useLiveFusionCoordinator:", "Aggregates streaming outputs, requesting low-latency fusion via /api/predict/fuse."),
        ]),
        ("Interactive Components", C_BLUE_ACCENT, [
            ("UnifiedEmotionOutput:", "Large dominant emotion badge, confidence gauge, and Circumplex plot."),
            ("ProbabilityChart:", "Animated 7-class probability bar graphs with emotion-coded hues."),
            ("ModalityContributionPanel:", "Visual proportional weight allocation bars (w_m)."),
            ("ConflictPanel:", "Cross-modal divergence monitor with concordance metrics."),
        ]),
        ("Design & Aesthetics", C_TEAL_ACCENT, [
            ("Glassmorphism:", "Modern translucent cards with subtle borders and blurred backdrops."),
            ("Vanilla CSS System:", "Ultra-fast rendering without heavyweight UI framework overhead."),
            ("Responsive Grid:", "Adapts seamlessly from desktop monitors to mobile viewports."),
            ("Manual Drawer:", "Drag-and-drop file upload drawer for static testing."),
        ]),
    ]

    card_w = Inches(3.75)
    gap = Inches(0.24)
    for i, (title, color, bullets) in enumerate(cards):
        x = Inches(0.8) + i * (card_w + gap)
        add_card(slide, x, Inches(1.8), card_w, Inches(4.9))

        tb = slide.shapes.add_textbox(x + Inches(0.2), Inches(2.0), card_w - Inches(0.4), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color

        add_bullet_list(tf, bullets)

def build_slide_22(prs):
    """Slide 22: Standalone Native Edge Programs."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "11. Native Programs", "Standalone Native Real-Time Programs (CLI / GUI)", "Zero-dependency desktop execution tools for ultra-low latency edge monitoring and testing")

    tools = [
        ("live_face.py: OpenCV Webcam HUD", C_NAVY_ROYAL, [
            ("Native OpenCV Window:", "Direct desktop video rendering bypassing browser layers."),
            ("Decoupled Loop:", "30 FPS smooth rendering + 3.5 FPS throttled inference."),
            ("Dynamic Bounding Boxes:", "Emotion-colored visual tracking overlays."),
            ("Probability Bar HUD:", "Renders live 7-class probability meters on screen."),
            ("CLI Options:", "--camera-id, --fps, --weights, --device, --duration."),
        ]),
        ("live_audio.py: Native Speech Classifier", C_BLUE_ACCENT, [
            ("SoundDevice Streaming:", "Non-blocking microphone capture with NumPy circular buffer."),
            ("Sliding Window Analysis:", "2.0s audio buffer window evaluated at 1.0s step intervals."),
            ("Silence Thresholding:", "RMS energy gating ignores background room noise."),
            ("Terminal Output:", "Real-time ASCII probability bars and predictions in console."),
            ("CLI Options:", "--device-index, --window, --step, --threshold."),
        ]),
        ("live_text.py: Interactive Terminal REPL", C_TEAL_ACCENT, [
            ("Interactive REPL:", "Instant text emotion classification from terminal prompt."),
            ("Transformer Inference:", "Direct RoBERTa PyTorch forward pass."),
            ("Colored Probability Bars:", "Formatted terminal visualizer for all 7 classes."),
            ("Circumplex Telemetry:", "Outputs Valence and Arousal coordinates."),
            ("Commands:", "'exit', 'quit', 'clear' screen commands."),
        ]),
    ]

    card_w = Inches(3.75)
    gap = Inches(0.24)
    for i, (title, color, bullets) in enumerate(tools):
        x = Inches(0.8) + i * (card_w + gap)
        add_card(slide, x, Inches(1.8), card_w, Inches(4.9))

        tb = slide.shapes.add_textbox(x + Inches(0.2), Inches(2.0), card_w - Inches(0.4), Inches(4.5))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = title
        p.font.name = "Arial"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color

        add_bullet_list(tf, bullets)

def build_slide_23(prs):
    """Slide 23: Quality Assurance, Automated Testing, & Security."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "12. Quality Assurance", "Quality Assurance, Testing & Privacy Architecture", "Rigorous automated test coverage and enterprise data protection standards")

    # Left: Testing
    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Automated Test Matrix: 100% Passing"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_GREEN

    b1 = [
        ("PyTest Backend Suite (21 / 21 Passing):", "Comprehensive coverage in tests/test_api.py and test_fusion.py."),
        ("Endpoint Integration:", "Validates health, text, audio, vision, and multimodal HTTP responses."),
        ("Fusion Mathematical Invariants:", "Verifies probability sums = 1.0, zero-mass handling, and all 7 modality permutations."),
        ("Boundary & Security Checks:", "Tests 413 file size limits, 415 extension blocks, and path redaction."),
        ("Vitest Frontend Suite (16 / 16 Passing):", "Validates component rendering, API service error handling, and audio visualizer fallbacks."),
    ]
    add_bullet_list(tf1, b1)

    # Right: Privacy
    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "Enterprise Privacy & Data Sanitization"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_BLUE_ACCENT

    b2 = [
        ("100% Local Inference:", "Zero audio waveforms, facial images, or text queries are transmitted to external cloud APIs or third parties."),
        ("Zero Telemetry:", "Contains zero Google Analytics, tracking beacons, or remote telemetry scripts."),
        ("Ephemeral File Lifecycle:", "Multipart uploads are written to system temp folders and strictly unlinked in finally blocks."),
        ("Path Redaction:", "API responses recursively strip internal filesystem paths to prevent host environment disclosure."),
    ]
    add_bullet_list(tf2, b2)

def build_slide_24(prs):
    """Slide 24: Operational Deployment, Ethics, & Future Roadmap."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    set_slide_background(slide, C_BG_LIGHT)
    add_header(slide, "14. Operations & Roadmap", "Deployment Guide, Ethical Boundaries & Roadmap", "Operational instructions, non-clinical research boundaries, and future enhancements")

    add_card(slide, Inches(0.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb1 = slide.shapes.add_textbox(Inches(1.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p1 = tf1.paragraphs[0]
    p1.text = "Quickstart Operational Runbook"
    p1.font.name = "Arial"
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = C_NAVY_DARK

    b1 = [
        ("Prerequisites:", "Windows 10/11 64-bit, Python 3.10-3.14, Node.js >= 18."),
        ("1. Backend Launch:", "uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload"),
        ("2. Frontend Launch:", "cd frontend && npm run dev (http://127.0.0.1:5173)"),
        ("3. Interactive Docs:", "Swagger UI available at http://127.0.0.1:8000/api/docs"),
        ("4. Native Live Tools:", "python live_face.py | python live_audio.py | python live_text.py"),
    ]
    add_bullet_list(tf1, b1)

    add_card(slide, Inches(6.8), Inches(1.8), Inches(5.7), Inches(4.9))
    tb2 = slide.shapes.add_textbox(Inches(7.0), Inches(2.0), Inches(5.3), Inches(4.5))
    tf2 = tb2.text_frame
    tf2.word_wrap = True

    p2 = tf2.paragraphs[0]
    p2.text = "Ethical Boundaries & Technical Roadmap"
    p2.font.name = "Arial"
    p2.font.size = Pt(14)
    p2.font.bold = True
    p2.font.color.rgb = C_AMBER

    b2 = [
        ("Non-Clinical Disclaimer:", "Designed for HCI research. Measures behavioral correlates, NOT psychological truth, psychiatric conditions, or deception."),
        ("Wav2Vec2 Full Speech Model:", "Fine-tuning self-supervised acoustic transformers to expand audio recognition to all 7 Ekman classes."),
        ("Temporal Video Transformers:", "Incorporating multi-frame temporal recurrence to capture micro-expressions across multi-minute conversations."),
        ("MediaPipe Face Mesh (FACS):", "Extracting 468 landmarks for facial action unit quantification."),
    ]
    add_bullet_list(tf2, b2)

def main():
    print("=" * 70)
    print("  MULTI-MODAL EMOTION RECOGNITION SYSTEM — PPTX PRESENTATION GENERATOR")
    print("=" * 70)

    prs = pptx.Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    slides_builders = [
        ("Slide 1: Title Slide", build_slide_1),
        ("Slide 2: Executive Summary & Affective Challenge", build_slide_2),
        ("Slide 3: Core Operating Principle: Zero Fake AI", build_slide_3),
        ("Slide 4: End-to-End System Topology", build_slide_4),
        ("Slide 5: Canonical Emotion Ontology", build_slide_5),
        ("Slide 6: Russell's 2D Circumplex Model", build_slide_6),
        ("Slide 7: Cross-Dataset Label Mapping", build_slide_7),
        ("Slide 8: Text Modality Engine (DistilRoBERTa)", build_slide_8),
        ("Slide 9: Speech Acoustic Processing (39 MFCCs)", build_slide_9),
        ("Slide 10: Speech 2D CNN (Pure PyTorch)", build_slide_10),
        ("Slide 11: Facial Vision Multi-Pass Detection", build_slide_11),
        ("Slide 12: AlexNet Architecture & Webcam HUD", build_slide_12),
        ("Slide 13: Vision Training & Zero Data Leakage", build_slide_13),
        ("Slide 14: Vision Model Ablation Experiments", build_slide_14),
        ("Slide 15: Per-Class Metrics & Confusion Matrix", build_slide_15),
        ("Slide 16: Empirical Diagnostic Analysis", build_slide_16),
        ("Slide 17: Decision Fusion Formulation", build_slide_17),
        ("Slide 18: Cross-Modal Concordance & Conflict", build_slide_18),
        ("Slide 19: All 7 Modality Permutations Matrix", build_slide_19),
        ("Slide 20: FastAPI Service & API Reference", build_slide_20),
        ("Slide 21: React 19 Frontend Web Dashboard", build_slide_21),
        ("Slide 22: Standalone Native Edge Programs", build_slide_22),
        ("Slide 23: Quality Assurance, QA & Security", build_slide_23),
        ("Slide 24: Deployment Guide, Ethics & Roadmap", build_slide_24),
    ]

    for idx, (label, builder) in enumerate(slides_builders):
        print(f"[{idx+1}/{len(slides_builders)}] Generating {label}...")
        builder(prs)

    output_path = os.path.abspath("Multi_Modal_Emotion_Recognition_System_Presentation.pptx")
    print(f"\nSaving PowerPoint Presentation to: {output_path}...")
    prs.save(output_path)

    file_size_bytes = os.path.getsize(output_path)
    file_size_kb = file_size_bytes / 1024.0
    file_size_mb = file_size_kb / 1024.0

    print("=" * 70)
    print(f"SUCCESS! Presentation generated successfully.")
    print(f"File Path: {output_path}")
    print(f"Total Slides: {len(prs.slides)}")
    print(f"File Size: {file_size_kb:.1f} KB ({file_size_mb:.2f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    main()
