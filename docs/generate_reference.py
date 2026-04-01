"""
Talon Assistant — Architecture & Developer Reference PDF Generator
Run from the project root: python docs/generate_reference.py
"""

import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak,
    Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

PAGE_W, PAGE_H = A4
MARGIN = 2.2 * cm

# ── Colour palette ────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#1e1e2e")
C_BLUE    = colors.HexColor("#89b4fa")
C_MAUVE   = colors.HexColor("#cba6f7")
C_GREEN   = colors.HexColor("#a6e3a1")
C_YELLOW  = colors.HexColor("#f9e2af")
C_RED     = colors.HexColor("#f38ba8")
C_SUBTEXT = colors.HexColor("#6c7086")
C_SURFACE = colors.HexColor("#313244")
C_OVERLAY = colors.HexColor("#45475a")
C_TEXT    = colors.HexColor("#cdd6f4")
C_WHITE   = colors.white
C_BLACK   = colors.black

# Derived colours for table styling
C_HEADER_BG = C_BLUE
C_HEADER_FG = C_DARK
C_ROW_EVEN  = colors.HexColor("#f8f8fc")
C_ROW_ODD   = colors.HexColor("#f0f0f5")
C_TABLE_BG  = colors.HexColor("#f5f5f5")
C_TABLE_GRID = colors.HexColor("#e0e0e0")

# ── Styles ────────────────────────────────────────────────────────────────────

def build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["cover_title"] = ParagraphStyle(
        "cover_title",
        fontName="Helvetica-Bold",
        fontSize=36,
        textColor=C_BLUE,
        alignment=TA_CENTER,
        spaceAfter=8,
        leading=44,
    )
    styles["cover_subtitle"] = ParagraphStyle(
        "cover_subtitle",
        fontName="Helvetica",
        fontSize=18,
        textColor=C_TEXT,
        alignment=TA_CENTER,
        spaceAfter=6,
    )
    styles["cover_body"] = ParagraphStyle(
        "cover_body",
        fontName="Helvetica",
        fontSize=12,
        textColor=C_SUBTEXT,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    styles["h1"] = ParagraphStyle(
        "h1",
        fontName="Helvetica-Bold",
        fontSize=22,
        textColor=C_BLUE,
        spaceBefore=18,
        spaceAfter=10,
        leading=28,
        borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "h2",
        fontName="Helvetica-Bold",
        fontSize=15,
        textColor=C_MAUVE,
        spaceBefore=14,
        spaceAfter=6,
        leading=20,
    )
    styles["h3"] = ParagraphStyle(
        "h3",
        fontName="Helvetica-Bold",
        fontSize=12,
        textColor=C_YELLOW,
        spaceBefore=10,
        spaceAfter=4,
        leading=16,
    )
    styles["body"] = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10,
        textColor=C_BLACK,
        spaceAfter=5,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "bullet",
        fontName="Helvetica",
        fontSize=10,
        textColor=C_BLACK,
        leftIndent=16,
        spaceAfter=3,
        leading=13,
        bulletIndent=8,
    )
    styles["code"] = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=8,
        textColor=C_DARK,
        backColor=colors.HexColor("#f0f0f0"),
        leftIndent=12,
        rightIndent=12,
        spaceBefore=4,
        spaceAfter=4,
        leading=11,
        borderPad=6,
        borderWidth=0.5,
        borderColor=C_OVERLAY,
        borderRadius=3,
    )
    styles["note"] = ParagraphStyle(
        "note",
        fontName="Helvetica-Oblique",
        fontSize=9,
        textColor=C_SUBTEXT,
        leftIndent=12,
        spaceAfter=4,
        leading=12,
    )
    styles["toc_h1"] = ParagraphStyle(
        "toc_h1",
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=C_DARK,
        leftIndent=0,
        spaceAfter=3,
        leading=14,
    )
    styles["toc_h2"] = ParagraphStyle(
        "toc_h2",
        fontName="Helvetica",
        fontSize=10,
        textColor=C_DARK,
        leftIndent=16,
        spaceAfter=2,
        leading=13,
    )
    styles["chapter_label"] = ParagraphStyle(
        "chapter_label",
        fontName="Helvetica",
        fontSize=10,
        textColor=C_SUBTEXT,
        spaceAfter=2,
    )
    # Small body text for table cells
    styles["cell"] = ParagraphStyle(
        "cell",
        fontName="Helvetica",
        fontSize=9,
        textColor=C_DARK,
        leading=12,
    )
    styles["cell_bold"] = ParagraphStyle(
        "cell_bold",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=C_DARK,
        leading=12,
    )
    styles["cell_header"] = ParagraphStyle(
        "cell_header",
        fontName="Helvetica-Bold",
        fontSize=9,
        textColor=C_WHITE,
        leading=12,
    )
    styles["cell_code"] = ParagraphStyle(
        "cell_code",
        fontName="Courier",
        fontSize=8,
        textColor=C_DARK,
        leading=11,
    )
    return styles


# ── Page decorators ───────────────────────────────────────────────────────────

def draw_header_footer(canvas, doc):
    canvas.saveState()
    # Header bar
    canvas.setFillColor(C_DARK)
    canvas.rect(0, PAGE_H - 1.2*cm, PAGE_W, 1.2*cm, fill=1, stroke=0)
    canvas.setFillColor(C_BLUE)
    canvas.setFont("Helvetica-Bold", 9)
    canvas.drawString(MARGIN, PAGE_H - 0.75*cm, "Talon Assistant — Architecture & Developer Reference")
    canvas.setFillColor(C_SUBTEXT)
    canvas.setFont("Helvetica", 8)
    if hasattr(doc, "_current_chapter"):
        canvas.drawRightString(PAGE_W - MARGIN, PAGE_H - 0.75*cm, doc._current_chapter)
    # Footer
    canvas.setFillColor(C_SUBTEXT)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(MARGIN, 0.75*cm, "Talon Assistant Developer Reference")
    canvas.drawCentredString(PAGE_W/2, 0.75*cm, f"Page {doc.page}")
    canvas.drawRightString(PAGE_W - MARGIN, 0.75*cm, "GPL-3.0 — Open Source")
    # Footer line
    canvas.setStrokeColor(C_OVERLAY)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 1.1*cm, PAGE_W - MARGIN, 1.1*cm)
    canvas.restoreState()


def draw_cover_page(canvas, doc):
    canvas.saveState()
    # Dark background
    canvas.setFillColor(C_DARK)
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Top accent bar
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, PAGE_H - 0.6*cm, PAGE_W, 0.6*cm, fill=1, stroke=0)
    # Bottom accent bar
    canvas.setFillColor(C_MAUVE)
    canvas.rect(0, 0, PAGE_W, 0.4*cm, fill=1, stroke=0)
    # Side accent
    canvas.setFillColor(C_BLUE)
    canvas.rect(0, 0, 0.5*cm, PAGE_H, fill=1, stroke=0)
    # Decorative circles
    canvas.setFillColor(colors.HexColor("#313244"))
    canvas.circle(PAGE_W - 3*cm, PAGE_H - 6*cm, 4*cm, fill=1, stroke=0)
    canvas.circle(3*cm, 5*cm, 2.5*cm, fill=1, stroke=0)
    canvas.restoreState()


# ── Helper builders ───────────────────────────────────────────────────────────

def divider(styles):
    return [HRFlowable(width="100%", thickness=0.5, color=C_OVERLAY, spaceAfter=8, spaceBefore=4)]


def chapter_heading(num, title, styles, doc=None):
    if doc:
        doc._current_chapter = f"Chapter {num}: {title}"
    elems = []
    elems.append(Spacer(1, 0.3*cm))
    elems.append(Paragraph(f"CHAPTER {num}", styles["chapter_label"]))
    elems.append(Paragraph(title, styles["h1"]))
    elems += divider(styles)
    return elems


def section(title, styles):
    return [Paragraph(title, styles["h2"])]


def subsection(title, styles):
    return [Paragraph(title, styles["h3"])]


def body(text, styles):
    return [Paragraph(text, styles["body"])]


def bullets(items, styles):
    return [Paragraph(f"\u2022 {item}", styles["bullet"]) for item in items]


def code_block(code_text, styles):
    safe = code_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = safe.split("\n")
    paras = []
    for line in lines:
        paras.append(Paragraph(line if line.strip() else " ", styles["code"]))
    return paras


def note(text, styles):
    return [Paragraph(f"Note: {text}", styles["note"])]


def sp(n=1):
    return [Spacer(1, n * 0.3 * cm)]


# ── Table diagram builders ───────────────────────────────────────────────────

def make_info_table(rows, styles, col_widths=None):
    """Create a styled two-column info table (key-value).

    rows: list of (key, value) tuples.
    """
    data = [[Paragraph(f"<b>{k}</b>", styles["cell_bold"]),
             Paragraph(v, styles["cell"])] for k, v in rows]
    if col_widths is None:
        col_widths = [4.5*cm, PAGE_W - 2*MARGIN - 4.5*cm - 10]
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#eeeef5")),
        ('BACKGROUND', (1, 0), (1, -1), C_TABLE_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, C_OVERLAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, C_TABLE_GRID),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return [t]


def make_layer_table(headers, rows, styles, col_widths=None):
    """Create a styled table with coloured header row.

    headers: list of header strings.
    rows: list of lists of cell strings.
    """
    header_cells = [Paragraph(h, styles["cell_header"]) for h in headers]
    body_cells = []
    for row in rows:
        body_cells.append([Paragraph(cell, styles["cell"]) for cell in row])
    data = [header_cells] + body_cells
    if col_widths is None:
        w = (PAGE_W - 2*MARGIN - 10) / len(headers)
        col_widths = [w] * len(headers)
    t = Table(data, colWidths=col_widths)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), C_BLUE),
        ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
        ('BOX', (0, 0), (-1, -1), 1, C_OVERLAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_TABLE_GRID),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]
    # Alternate row colours
    for i in range(len(rows)):
        bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        style_cmds.append(('BACKGROUND', (0, i+1), (-1, i+1), bg))
    t.setStyle(TableStyle(style_cmds))
    return [t]


def make_flow_table(rows, styles):
    """Create a styled vertical flow diagram using Table.

    rows: list of strings (one per step).
    """
    data = [[Paragraph(row.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"),
                        styles["cell_code"])] for row in rows]
    t = Table(data, colWidths=[PAGE_W - 2*MARGIN - 20])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), C_TABLE_BG),
        ('BOX', (0, 0), (-1, -1), 0.5, C_OVERLAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.25, C_TABLE_GRID),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    return [t]


def make_step_table(steps, styles):
    """Create a flow diagram table with step numbers and descriptions.

    steps: list of (num, action, note) tuples.
    """
    header = [
        Paragraph("<b>#</b>", styles["cell_header"]),
        Paragraph("<b>Action</b>", styles["cell_header"]),
        Paragraph("<b>Notes</b>", styles["cell_header"]),
    ]
    body_rows = []
    for num, action, note_text in steps:
        body_rows.append([
            Paragraph(str(num), styles["cell_bold"]),
            Paragraph(action, styles["cell"]),
            Paragraph(note_text, styles["cell"]),
        ])
    data = [header] + body_rows
    col_widths = [1*cm, 7*cm, PAGE_W - 2*MARGIN - 8*cm - 10]
    t = Table(data, colWidths=col_widths)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), C_MAUVE),
        ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
        ('BOX', (0, 0), (-1, -1), 1, C_OVERLAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_TABLE_GRID),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]
    for i in range(len(steps)):
        bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        style_cmds.append(('BACKGROUND', (0, i+1), (-1, i+1), bg))
    t.setStyle(TableStyle(style_cmds))
    return [t]


def make_config_table(rows, styles):
    """Create a config reference table with key, type, description columns.

    rows: list of (key, type, description) tuples.
    """
    header = [
        Paragraph("<b>Key</b>", styles["cell_header"]),
        Paragraph("<b>Type</b>", styles["cell_header"]),
        Paragraph("<b>Description</b>", styles["cell_header"]),
    ]
    body_rows = []
    for key, typ, desc in rows:
        body_rows.append([
            Paragraph(f"<font face='Courier' size='8'>{key}</font>", styles["cell"]),
            Paragraph(typ, styles["cell"]),
            Paragraph(desc, styles["cell"]),
        ])
    data = [header] + body_rows
    col_widths = [4*cm, 1.8*cm, PAGE_W - 2*MARGIN - 5.8*cm - 10]
    t = Table(data, colWidths=col_widths)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), C_SURFACE),
        ('TEXTCOLOR', (0, 0), (-1, 0), C_WHITE),
        ('BOX', (0, 0), (-1, -1), 1, C_OVERLAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, C_TABLE_GRID),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]
    for i in range(len(rows)):
        bg = C_ROW_EVEN if i % 2 == 0 else C_ROW_ODD
        style_cmds.append(('BACKGROUND', (0, i+1), (-1, i+1), bg))
    t.setStyle(TableStyle(style_cmds))
    return [t]


# ── Build document ────────────────────────────────────────────────────────────

def build_document():
    output_path = os.path.join(os.path.dirname(__file__), "Talon_Developer_Reference.pdf")
    S = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=1.6*cm,
        bottomMargin=1.6*cm,
        title="Talon Assistant — Architecture & Developer Reference",
        author="Talon Assistant Project",
    )
    doc._current_chapter = ""

    story = []

    # =========================================================================
    # COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 6*cm))
    story.append(Paragraph("Talon Assistant", S["cover_title"]))
    story.append(Spacer(1, 0.5*cm))
    story.append(Paragraph("Architecture &amp; Developer Reference", S["cover_subtitle"]))
    story.append(Spacer(1, 0.8*cm))
    story.append(Paragraph(
        "A complete guide to the codebase for offline development and maintenance.",
        S["cover_body"]))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph("Version: Current  |  April 2026  |  GPL-3.0", S["cover_body"]))
    story.append(PageBreak())

    # =========================================================================
    # TABLE OF CONTENTS
    # =========================================================================
    story.append(Paragraph("Table of Contents", S["h1"]))
    story += divider(S)

    toc_entries = [
        ("1",  "Overview &amp; Architecture"),
        ("2",  "RAG &amp; Document System"),
        ("3",  "LoRA Training"),
        ("4",  "Behavioral Rules &amp; Suggestions"),
        ("5",  "Session Memory &amp; Reflections"),
        ("6",  "Core Module — assistant.py"),
        ("7",  "Core Module — memory.py"),
        ("8",  "Core Module — llm_client.py"),
        ("9",  "Core Module — conversation.py"),
        ("10", "Core Module — document_retriever.py"),
        ("11", "Core Module — llm_server.py"),
        ("12", "Core Module — security.py"),
        ("13", "Core Module — voice.py"),
        ("14", "Core Module — vision.py &amp; document_extractor.py"),
        ("15", "Core Module — logging_config.py &amp; config.py"),
        ("16", "Core Module — scheduler.py, credential_store.py &amp; chat_store.py"),
        ("17", "Core Module — embeddings.py, reranker.py, training_harvester.py &amp; marketplace.py"),
        ("18", "GUI — main_window.py &amp; assistant_bridge.py"),
        ("19", "GUI — workers.py, system_tray.py, theme_manager.py &amp; output_interceptor.py"),
        ("20", "Talent System — base.py &amp; planner.py"),
        ("21", "Talent — email_talent.py"),
        ("22", "Talent — reminder.py &amp; notes.py"),
        ("23", "Talent — hue_lights.py &amp; desktop_control.py"),
        ("24", "Talent — news.py &amp; news_digest.py"),
        ("25", "Talent — rules.py, history.py &amp; clipboard_transform.py"),
        ("26", "Talent — lora_train.py &amp; signal_remote.py"),
        ("27", "Testing"),
        ("A",  "Appendix A: Configuration Reference"),
        ("B",  "Appendix B: Adding a New Talent"),
        ("C",  "Appendix C: Extending the Security Filter"),
    ]

    for num, title in toc_entries:
        prefix = "Chapter" if num.isdigit() else "Appendix"
        story.append(Paragraph(f"{prefix} {num}  ........  {title}", S["toc_h1"]))

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 1: OVERVIEW & ARCHITECTURE
    # =========================================================================
    story += chapter_heading(1, "Overview &amp; Architecture", S, doc)

    story += section("What is Talon?", S)
    story += body(
        "Talon is a local-first personal AI desktop assistant built on Python and PyQt6. "
        "It runs entirely on the user's machine with no cloud dependency. "
        "The LLM backend is either KoboldCpp or llama.cpp (built-in server mode), "
        "both running open-weight models locally via CUDA acceleration. "
        "Talon combines voice interaction (Whisper STT + Edge TTS), a GUI chat interface, "
        "a plugin talent system for extensibility, and a dual-database memory layer "
        "(SQLite for structured data, ChromaDB for semantic vector search). "
        "The project is open source under GPL-3.0.", S)

    story += section("Technology Stack", S)
    story += make_layer_table(
        ["Component", "Technology"],
        [
            ["GUI", "PyQt6 — tabbed windows, system tray, hotkey (Ctrl+Shift+J)"],
            ["LLM Backend", "KoboldCpp or llama.cpp server (CUDA, Windows)"],
            ["Model", "Qwen3 8B (or compatible GGUF) with optional mmproj for vision"],
            ["STT", "faster-whisper (Whisper small, CUDA float16, CPU int8 fallback)"],
            ["TTS", "edge-tts (en-US-AriaNeural by default)"],
            ["Vector DB", "ChromaDB with BAAI/bge-base-en-v1.5 embeddings"],
            ["Relational DB", "SQLite (commands, actions, notes, rules, corrections, security_alerts)"],
            ["Reranker", "BAAI/bge-reranker-base (cross-encoder, CPU)"],
            ["Vision", "PIL + base64 encoding for LLM multimodal input"],
            ["Credentials", "OS keyring via keyring library"],
            ["Logging", "Python logging — console (INFO) + rotating file (DEBUG)"],
            ["Testing", "pytest — 97 tests across 7 files, fully mocked"],
        ],
        S, col_widths=[3.5*cm, PAGE_W - 2*MARGIN - 3.5*cm - 10],
    )

    story += section("High-Level Component Map", S)
    story += make_layer_table(
        ["Layer", "Components"],
        [
            ["User Interface", "Voice Input (Whisper STT), GUI (PyQt6 MainWindow)"],
            ["Bridge Layer", "AssistantBridge (Qt Signals), CommandWorker (QThread)"],
            ["Core Orchestrator", "TalonAssistant — process_command() gateway"],
            ["Conversation", "ConversationEngine — buffer, summary, vision, RAG injection"],
            ["LLM", "LLMClient — KoboldCpp / llama.cpp / OpenAI-compatible"],
            ["Memory", "MemorySystem — SQLite + ChromaDB, 4 collections"],
            ["RAG Pipeline", "DocumentRetriever — multi-query, RRF, reranking, multi-hop"],
            ["Security", "SecurityFilter — input/output scan, rate limit, confirmation gates"],
            ["Voice", "VoiceSystem — Whisper STT + edge-tts"],
            ["Scheduler", "Background cron-style task scheduler"],
            ["Talents", "15+ plugin talents (BaseTalent ABC), PlannerTalent at priority 85"],
        ],
        S, col_widths=[3.5*cm, PAGE_W - 2*MARGIN - 3.5*cm - 10],
    )

    story += section("The Gateway Pattern", S)
    story += body(
        "Every command — whether it arrives from voice input, the GUI chat box, "
        "the scheduler, or a planner sub-step — passes through a single method: "
        "TalonAssistant.process_command(). This gateway handles rate-limiting, "
        "input security scanning, rule matching, talent routing, LLM fallback, "
        "promise interception, output scanning, memory logging, and TTS output. "
        "Nothing bypasses this pipeline.", S)

    story += section("Command Processing Pipeline", S)
    story += make_step_table([
        ("1", "Security: rate limit check", "Blocked? Return error message"),
        ("2", "Security: input filter", "Blocked? Return error message"),
        ("3", "Session reflection fast-path", "Checks for 'reflect on today' etc."),
        ("4", "Correction detection", "'No I meant...' re-routes corrected intent"),
        ("5", "Repeat last action?", "Re-executes previous successful action"),
        ("6", "Behavioral rule match", "Rule found? Execute rule action"),
        ("7", "Build context dict", "memory, llm, vision, voice, config"),
        ("8", "Route to talent via LLM", "LLM picks best handler from roster"),
        ("9", "Talent execute or conversation fallback", "ConversationEngine.handle()"),
        ("10", "Promise interception", "Detects 'I'll search for X' and re-routes"),
        ("11", "Security: output scan", "System prompt leak, API key detection"),
        ("12", "Log to SQLite + speak/return", "Memory persistence and TTS"),
    ], S)

    story += section("Directory Structure", S)
    story += make_layer_table(
        ["Path", "Description"],
        [
            ["main.py", "Entry point: GUI, voice, or text mode"],
            ["ingest_documents.py", "CLI tool: batch RAG document ingestion"],
            ["requirements.txt", "Dependencies"],
            ["config/settings.json", "Runtime config (gitignored)"],
            ["config/settings.example.json", "Committed template"],
            ["config/talents.json", "Per-talent enable/config (gitignored)"],
            ["core/assistant.py", "TalonAssistant orchestrator (~1587 lines)"],
            ["core/conversation.py", "ConversationEngine — buffer, RAG, vision (~636 lines)"],
            ["core/document_retriever.py", "DocumentRetriever — RAG pipeline (~420 lines)"],
            ["core/memory.py", "SQLite + ChromaDB memory system (~1217 lines)"],
            ["core/llm_client.py", "Multi-backend LLM API client + LLMError (~367 lines)"],
            ["core/llm_server.py", "Built-in llama.cpp server manager"],
            ["core/security.py", "SecurityFilter + wrap_external()"],
            ["core/voice.py", "VoiceSystem (Whisper + edge-tts)"],
            ["core/vision.py", "Screenshot + image loading"],
            ["core/config.py", "deep_merge() utility"],
            ["core/logging_config.py", "setup_logging() — console + rotating file"],
            ["core/scheduler.py", "Background cron-style task scheduler"],
            ["core/credential_store.py", "OS keyring wrapper"],
            ["core/chat_store.py", "Conversation session persistence"],
            ["core/embeddings.py", "BGE embedding model wrapper"],
            ["core/reranker.py", "Cross-encoder reranker"],
            ["core/training_harvester.py", "Alpaca JSONL training pair collector"],
            ["core/document_extractor.py", "Inline document review (not RAG)"],
            ["core/marketplace.py", "Talent marketplace (discovery + install)"],
            ["gui/", "PyQt6 GUI: MainWindow, Bridge, Workers, Tray, Theme"],
            ["talents/", "Plugin talent modules (base.py, planner.py, 15+ talents)"],
            ["talents/user/", "User-created talent modules (auto-discovered)"],
            ["tests/", "pytest suite — 97 tests across 7 test files"],
            ["data/talon_memory.db", "SQLite database"],
            ["data/chroma_db/", "ChromaDB vector store"],
            ["data/logs/", "Rotating log files (talon.log, 5MB, 3 backups)"],
            ["documents/", "Drop PDFs/docs here for RAG ingestion"],
            ["bin/", "llama-server.exe + CUDA DLLs (builtin mode)"],
        ],
        S, col_widths=[4.5*cm, PAGE_W - 2*MARGIN - 4.5*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 2: RAG & DOCUMENT SYSTEM
    # =========================================================================
    story += chapter_heading(2, "RAG &amp; Document System", S, doc)

    story += body(
        "Talon uses a Retrieval-Augmented Generation (RAG) system to inject relevant "
        "knowledge into LLM prompts at query time. The system is built on two layers: "
        "ChromaDB for semantic vector search and SQLite for structured metadata. "
        "Documents are ingested via ingest_documents.py and stored as searchable chunks. "
        "The RAG retrieval pipeline was extracted from memory.py into a dedicated "
        "DocumentRetriever class in core/document_retriever.py (see Chapter 10).", S)

    story += section("ChromaDB Collections", S)
    story += make_layer_table(
        ["Collection", "Purpose"],
        [
            ["talon_memory", "User preferences, habits, and eviction insights"],
            ["talon_documents", "Ingested document chunks (PDFs, DOCX, etc.)"],
            ["talon_notes", "User notes with semantic search"],
            ["talon_rules", "Behavioral rules (trigger/action pairs, synced from SQLite)"],
        ],
        S, col_widths=[4*cm, PAGE_W - 2*MARGIN - 4*cm - 10],
    )

    story += section("Embedding &amp; Reranking Pipeline", S)
    story += body(
        "Talon uses the BAAI/bge-base-en-v1.5 SentenceTransformer model for embeddings. "
        "Documents are embedded without a prefix; queries receive the BGE asymmetric "
        "retrieval prefix. After initial vector retrieval, the BAAI/bge-reranker-base "
        "cross-encoder reranks results by scoring (query, chunk) pairs jointly, "
        "significantly improving recall when query vocabulary differs from document vocabulary.", S)

    story += section("RAG Retrieval Modes", S)
    story += make_layer_table(
        ["Mode", "Threshold", "Max Chunks", "Features"],
        [
            ["Ambient", "dist &lt;= 0.55", "2",
             "Auto-triggered when documents exist; minimal latency impact"],
            ["Explicit", "dist &lt;= 1.8", "8 (after rerank from pool of 12)",
             "Multi-query expansion, $contains text fallback, RRF fusion, "
             "Jaccard dedup, cross-encoder reranking, multi-hop entity extraction"],
            ["Synthesis", "dist &lt;= 1.8", "8",
             "Triggered by compare/list-all patterns; wide retrieval, no multi-hop"],
        ],
        S, col_widths=[2*cm, 2.5*cm, 2.5*cm, PAGE_W - 2*MARGIN - 7*cm - 10],
    )
    story += note(
        "_check_documents_exist() caches ChromaDB count(). No RAG calls "
        "are made on fresh installs (empty DB).", S)

    story += section("Vision-Enhanced PDF Ingestion (--vision flag)", S)
    story += body(
        "When ingest_documents.py is run with --vision, each PDF page is rendered "
        "at 100 DPI to a pixmap, sent to the vision LLM (Qwen3-VL via mmproj), "
        "and the model produces a structured plain-text description. "
        "Art/blank pages are skipped if raw text length is under 15 words. "
        "The stored chunk format combines the vision description with raw fitz text:", S)
    story += code_block(
        "[VISION: <model description of page content>]\n\n"
        "RAW TEXT:\n"
        "<fitz extracted text>", S)

    story += section("Chunk Metadata", S)
    story += make_info_table([
        ("filename", "Source document filename"),
        ("page_number", "0-based page index (citations show page_number + 1)"),
        ("type", "'vision_enhanced' or 'text_only'"),
        ("ingested_at", "ISO timestamp"),
    ], S)

    story += section("SQLite Schema", S)
    story += make_layer_table(
        ["Table", "Key Columns", "Purpose"],
        [
            ["commands", "id, timestamp, command_text, success, response",
             "Every command processed by the assistant"],
            ["actions", "id, command_id FK, action_json, result, success",
             "Individual actions within a command execution"],
            ["notes", "id, timestamp, content, tags, chroma_id",
             "User notes (dual-stored with ChromaDB)"],
            ["rules", "id, trigger_phrase, action_text, enabled, chroma_id",
             "Behavioral rules (source of truth, synced to ChromaDB)"],
            ["corrections", "id, timestamp, prev_command, correction",
             "User corrections for learning"],
            ["security_alerts", "id, timestamp, control, pattern_id, action_taken",
             "Audit log for security filter events"],
        ],
        S, col_widths=[2.5*cm, 5.5*cm, PAGE_W - 2*MARGIN - 8*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 3: LORA TRAINING
    # =========================================================================
    story += chapter_heading(3, "LoRA Training", S, doc)

    story += section("Training Pair Harvesting", S)
    story += body(
        "Talon silently collects high-quality instruction-response pairs during normal "
        "use via training_harvester.py. These pairs are written in Alpaca format to "
        "data/training_pairs.jsonl and are ready to feed into Unsloth or axolotl "
        "for LoRA fine-tuning of the local model.", S)

    story += section("Two Capture Sources", S)
    story += make_info_table([
        ("correction",
         "When the user corrects a wrong answer, the original command is paired "
         "with the correct re-executed response. Teaches the model from its mistakes."),
        ("web_search",
         "When the model's training knowledge was stale and web search found "
         "the truth, the search command is paired with the synthesised correct answer."),
    ], S)

    story += section("Alpaca Format", S)
    story += code_block("""\
{
  "instruction": "What is the current price of Bitcoin?",
  "input": "",
  "output": "Bitcoin is currently trading at $67,450 USD...",
  "source": "web_search",
  "timestamp": "2026-03-10T14:23:00"
}""", S)

    story += section("Deduplication &amp; Controls", S)
    story += body(
        "The harvester performs a simple linear scan of the JSONL file to skip "
        "exact duplicate instructions before appending. Responses shorter than "
        "20 characters are discarded. Harvesting can be disabled via settings.json:", S)
    story += code_block('"training": {"harvest_pairs": false}', S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 4: BEHAVIORAL RULES & SUGGESTIONS
    # =========================================================================
    story += chapter_heading(4, "Behavioral Rules &amp; Suggestions", S, doc)

    story += body(
        "Talon's rule system lets users define persistent trigger-action pairs: "
        "'when I say X, do Y.' Rules are stored in SQLite (source of truth) and "
        "synced to ChromaDB for semantic matching. On startup, if the ChromaDB "
        "talon_rules collection is empty but SQLite has rules, they are rebuilt "
        "automatically.", S)

    story += section("Rule Structure", S)
    story += make_info_table([
        ("trigger_phrase", "The phrase that activates the rule"),
        ("action_text", "What to do (a command string, or 'say X' for verbatim responses)"),
        ("enabled", "Boolean, rules can be toggled without deletion"),
        ("chroma_id", "ChromaDB document ID for semantic lookup"),
    ], S)

    story += section("Rule Matching Fast-Path", S)
    story += body(
        "To avoid querying ChromaDB on every command, the assistant caches a "
        "_rules_exist boolean from SQLite COUNT(). This cache is invalidated "
        "whenever a rule is added, deleted, or toggled. If no rules exist, "
        "the entire rule-matching block is skipped.", S)

    story += section("Injection Defence on Rule Actions", S)
    story += body(
        "Rule actions are scanned for injection markers before execution. "
        "Patterns covering format markers, instruction overrides, jailbreak "
        "attempts, and persona hijacks will cause a rule to be rejected.", S)

    story += section("Execution Flow", S)
    story += make_step_table([
        ("1", "User says 'goodnight talon'", ""),
        ("2", "_check_rules() finds rule: trigger='goodnight'", "Semantic ChromaDB match"),
        ("3", "Action is a command, not verbatim", "action='turn off all lights'"),
        ("4", "process_command('turn off all lights', _executing_rule=True)", ""),
        ("5", "_executing_rule=True prevents re-matching, buffer pollution, promise interception", ""),
        ("6", "Routes to hue_lights talent, executes", "Lights off"),
    ], S)

    story += section("Preference Detection", S)
    story += body(
        "_detect_preference() scans every command for keywords like 'prefer', 'like', "
        "'favorite', 'always', 'usually', 'remember'. When found, the preference is "
        "stored in the talon_memory ChromaDB collection for future context injection.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 5: SESSION MEMORY & REFLECTIONS
    # =========================================================================
    story += chapter_heading(5, "Session Memory &amp; Reflections", S, doc)

    story += section("Conversation Buffer", S)
    story += body(
        "The conversation_buffer is a deque(maxlen=16) holding the last 16 turns "
        "(8 user + 8 Talon). It is injected into LLM prompts on the conversation "
        "path only — talent calls, planner sub-steps, and rule executions do not "
        "add to it. The buffer resets on app restart. Buffer management is now "
        "owned by ConversationEngine (core/conversation.py).", S)

    story += section("Session Summarizer", S)
    story += body(
        "_async_summarize_session() runs in a background daemon thread every 6 turns "
        "(3 exchanges). It compresses the entire conversation_buffer into a 1-2 sentence "
        "summary stored in _session_summary. This summary replaces the raw buffer dump "
        "in the prompt — instead of injecting all 16 raw turns, Talon injects "
        "[Session so far: ...] + the last 4 verbatim turns, keeping context compact. "
        "Output is scanned by SecurityFilter before storage.", S)

    story += section("Eviction Consolidation", S)
    story += body(
        "_consolidate_evicted_turn() runs as a background thread whenever a turn is "
        "evicted from the full buffer. It asks the LLM if the exchange reveals a "
        "user preference, habit, or fact worth remembering long-term. If yes, the "
        "insight is stored in talon_memory ChromaDB. Output is scanned by both "
        "SecurityFilter.check_output() and check_semantic() before writing, "
        "preventing cross-session semantic injection via stored memory.", S)

    story += section("Buffer Turn Flow", S)
    story += make_step_table([
        ("1", "User sends command", ""),
        ("2", "ConversationEngine.buffer_turn(command, response)", "Consolidated entry point"),
        ("3", "_maybe_evict_consolidate()", "If buffer full: pop oldest pair, spawn consolidation thread"),
        ("4", "Append user + talon turns to buffer", ""),
        ("5", "Every 6 turns: spawn _async_summarize_session()", "Background compression"),
        ("6", "Summary: SecurityFilter.check_output() + check_semantic()", "Before storage"),
    ], S)

    story += section("Session Reflection", S)
    story += body(
        "_reflect_on_session() is triggered by explicit reflection commands. "
        "It fetches all SQLite commands since session start, sends a structured "
        "JSON analysis prompt to the LLM, and returns a summary with observed "
        "preferences, failure analysis, and shortcut suggestions. The reflection "
        "is stored in talon_memory for next-session context injection.", S)

    story += section("Last-Session Context Injection", S)
    story += body(
        "At startup, inject_last_session_context() loads the most recent session "
        "reflection from memory. For the first 3 conversation turns, this context "
        "is prepended to the LLM prompt as [Last session: ...], giving the model "
        "continuity across restarts. The context fades out after 3 turns.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 6: CORE MODULE — assistant.py
    # =========================================================================
    story += chapter_heading(6, "Core Module — assistant.py", S, doc)

    story += body(
        "assistant.py is the central orchestrator of Talon. At ~1587 lines (reduced "
        "from ~1800 after extracting ConversationEngine and DocumentRetriever), it "
        "contains TalonAssistant, all routing logic, correction/approval detection, "
        "and the session memory system. Conversation handling has been extracted to "
        "core/conversation.py and the RAG pipeline to core/document_retriever.py.", S)

    story += section("Key Extractions (April 2026)", S)
    story += make_info_table([
        ("ConversationEngine", "Conversation buffer, session summary, vision detection, "
         "promise interception, RAG injection — now in core/conversation.py"),
        ("DocumentRetriever", "Multi-query expansion, RRF fusion, cross-encoder reranking, "
         "multi-hop — now in core/document_retriever.py"),
        ("wrap_external()", "_wrap_external() injection defence wrapper — now in core/security.py"),
        ("deep_merge()", "_deep_merge() config utility — now in core/config.py"),
    ], S)

    story += section("Class-Level Constants", S)
    story += bullets([
        "_ROUTING_SYSTEM_PROMPT — LLM prompt used to route commands to talents",
        "_RULE_DETECTION_SYSTEM_PROMPT — LLM prompt for detecting new rule definitions",
        "_CONVERSATION / _CONVERSATION_RAG — sentinel objects for routing results",
        "_RULE_ACTION_INJECTION_PATTERNS — strings that invalidate rule actions",
        "_RULE_INDICATORS / _REFLECT_PHRASES — fast-path keyword lists",
        "_CORRECTION_PHRASES / _APPROVAL_PHRASES — for correction and approval detection",
    ], S)

    story += section("Initialisation Sequence", S)
    story += make_step_table([
        ("1", "Load config (settings.json + news_digest.json)", "deep_merge from core/config.py"),
        ("2", "LLMClient (test_connection)", "LLMError on failure"),
        ("3", "MemorySystem (SQLite + ChromaDB)", "WAL mode, PostHog disabled"),
        ("4", "SecurityFilter + seed system prompt phrases", "Leak detection setup"),
        ("5", "VisionSystem", "PIL ImageGrab"),
        ("6", "VoiceSystem (loads Whisper, edge-tts)", "CUDA float16 preferred"),
        ("7", "_discover_talents() — load all BaseTalent subclasses", "pkgutil.iter_modules"),
        ("8", "ConversationEngine(self)", "Owns buffer, summary, docs cache"),
        ("9", "_migrate_legacy_credentials()", "One-time keyring migration"),
        ("10", "Scheduler.start() if schedule config exists", "Background thread"),
    ], S)

    story += section("LLM Routing", S)
    story += body(
        "_route_with_llm() builds a routing prompt listing all enabled talents (their "
        "name, description, and up to 5 examples) and asks the LLM to return the single "
        "best handler name. A keyword/example cross-check validates the LLM's choice. "
        "If the LLM is unreachable, _find_talent_by_keywords() provides a degraded "
        "keyword-only fallback.", S)

    story += section("LLMError Exception Handling", S)
    story += body(
        "As of April 2026, all three LLM backends (KoboldCpp, llama.cpp, OpenAI) raise "
        "LLMError instead of returning error strings. 36 callers across the codebase "
        "were updated to catch LLMError and handle failures gracefully. This replaces "
        "the previous pattern of checking response strings for error indicators.", S)

    story += section("Correction Detection", S)
    story += body(
        "_is_correction() checks for correction phrases ('no I meant', 'that was wrong'). "
        "When detected, _handle_correction() strips the correction prefix and re-routes "
        "the corrected intent directly to the appropriate talent, preserving context. "
        "A training pair is harvested from corrections.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 7: CORE MODULE — memory.py
    # =========================================================================
    story += chapter_heading(7, "Core Module — memory.py", S, doc)

    story += body(
        "memory.py provides the MemorySystem class (~1217 lines, reduced from ~1600 after "
        "extracting DocumentRetriever). It is Talon's dual-database persistence "
        "layer. SQLite handles structured command logs, rules, notes, corrections, and "
        "security alerts. ChromaDB handles all semantic vector search operations.", S)

    story += section("April 2026 Changes", S)
    story += bullets([
        "DocumentRetriever extraction — the full RAG pipeline (multi-query, RRF, reranking, "
        "multi-hop) is now in core/document_retriever.py. MemorySystem delegates to it.",
        "WAL mode — SQLite now uses journal_mode=WAL for better concurrent read performance.",
        "Rules ChromaDB sync — on startup, if talon_rules collection is empty but SQLite "
        "has rules, they are rebuilt automatically (guards against ChromaDB data loss).",
        "PostHog telemetry disabled — ChromaDB's built-in PostHog analytics thread is "
        "suppressed via environment variable to prevent background network traffic.",
    ], S)

    story += section("Key Public Methods", S)
    story += make_layer_table(
        ["Method", "Description"],
        [
            ["log_command()", "Write to commands table"],
            ["log_action()", "Write to actions table"],
            ["get_relevant_context()", "Retrieve preferences + optional RAG"],
            ["get_document_context()", "Full RAG pipeline via DocumentRetriever"],
            ["store_preference()", "Write to talon_memory ChromaDB"],
            ["store_note()", "Write to talon_notes + SQLite"],
            ["add_rule / delete_rule / toggle_rule", "Rules CRUD with ChromaDB sync"],
            ["get_last_session_reflection()", "Startup context injection"],
            ["get_relevant_corrections()", "Fetch past corrections for prompt hints"],
        ],
        S, col_widths=[5*cm, PAGE_W - 2*MARGIN - 5*cm - 10],
    )

    story += section("ChromaDB Collection Usage", S)
    story += make_info_table([
        ("talon_memory", "Preferences, habits, eviction insights, session reflections"),
        ("talon_documents", "Ingested document chunks with page metadata"),
        ("talon_notes", "User-created notes (semantic search)"),
        ("talon_rules", "Behavioral rules (synced from SQLite source of truth)"),
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 8: CORE MODULE — llm_client.py
    # =========================================================================
    story += chapter_heading(8, "Core Module — llm_client.py", S, doc)

    story += body(
        "LLMClient (~367 lines) is a multi-backend API wrapper supporting three server "
        "formats: KoboldCpp (native API), llama.cpp (POST /completion), and "
        "OpenAI-compatible (POST /v1/chat/completions). The active format is set "
        "via config['llm']['api_format'].", S)

    story += section("LLMError Exception Class", S)
    story += body(
        "LLMError is a custom exception raised by all three backends when a generation "
        "request fails. It carries a message and an optional HTTP status_code. This "
        "replaces the previous pattern where backends returned error strings that callers "
        "had to distinguish from real responses. 36 call sites were updated to catch "
        "LLMError explicitly.", S)
    story += code_block("""\
class LLMError(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code""", S)

    story += section("Three Backend Implementations", S)
    story += make_layer_table(
        ["Backend", "Endpoint", "Key Details"],
        [
            ["koboldcpp", "POST /api/v1/generate",
             "payload: {prompt, max_length, temperature, top_p, rep_pen, stop_sequence}; "
             "vision: payload['images'] = [base64, ...]"],
            ["llamacpp", "POST /completion",
             "Ready-guard checks server_manager.status; "
             "payload: {prompt, n_predict, repeat_penalty, stop, stream: false}; "
             "vision: payload['image_data'] = [{data: b64, id: 10+i}, ...]"],
            ["openai", "POST /v1/chat/completions",
             "messages: [{role, content}]; vision: content list with text + image_url"],
        ],
        S, col_widths=[2.2*cm, 4*cm, PAGE_W - 2*MARGIN - 6.2*cm - 10],
    )

    story += section("Server Ready-Guard", S)
    story += body(
        "When server_manager is set (builtin llama.cpp mode), _generate_llamacpp() "
        "checks server_manager.status before sending any request. If status is "
        "'starting', it returns a friendly 'still loading' message instead of "
        "hanging until timeout or returning a 503 error.", S)

    story += section("Vision Support", S)
    story += body(
        "All three backends support multimodal vision input. Images are passed as "
        "base64-encoded PNG strings. Multiple images can be sent via images_b64 list. "
        "The legacy screenshot_b64 parameter is normalised into images_b64 at dispatch "
        "time. Vision requests require a model with mmproj loaded.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 9: CORE MODULE — conversation.py (NEW)
    # =========================================================================
    story += chapter_heading(9, "Core Module — conversation.py", S, doc)

    story += body(
        "ConversationEngine (~636 lines) was extracted from assistant.py to separate "
        "conversation handling from command routing. It owns the conversation buffer, "
        "session summary, document-existence cache, and the full _handle_conversation "
        "pipeline including RAG injection, vision detection, attachment handling, and "
        "promise interception. All assistant state is accessed via self._a (the parent "
        "assistant instance).", S)

    story += section("Fast-Paths", S)
    story += body(
        "ConversationEngine handles several categories of queries without an LLM call:", S)
    story += make_layer_table(
        ["Fast-Path", "Trigger Examples", "Response"],
        [
            ["Time queries", "'what time is it', 'current time'",
             "datetime.now() formatted directly"],
            ["Date queries", "'what day is it', 'today's date'",
             "datetime.now() formatted directly"],
            ["System facts", "'home directory', 'username', 'hostname', 'what OS'",
             "os.environ / socket / platform lookups"],
            ["Rule definitions", "'when I say X do Y'",
             "Stores rule via _detect_and_store_rule(), acknowledges"],
        ],
        S, col_widths=[2.5*cm, 4.5*cm, PAGE_W - 2*MARGIN - 7*cm - 10],
    )

    story += section("Vision Detection", S)
    story += body(
        "The engine checks for explicit vision phrases ('on my screen', 'what do you see', "
        "'screenshot', etc.) to determine if a desktop screenshot should be captured. "
        "User-attached images from the GUI file picker are handled separately and take "
        "priority over screenshot capture.", S)

    story += section("Promise Interception", S)
    story += body(
        "detect_promise() contains 12 regex patterns that detect when the conversation "
        "LLM promises an action it cannot deliver ('I'll search the web for X', "
        "'let me open Chrome'). When detected, the implied command is extracted and "
        "returned to assistant.py for re-routing through process_command().", S)

    story += section("Buffer Management", S)
    story += bullets([
        "buffer_turn(command, response) — consolidated entry point for appending turns",
        "_maybe_evict_consolidate() — spawns background thread when oldest pair is evicted",
        "_consolidate_evicted_turn() — LLM asks if evicted turn has long-term insight",
        "_async_summarize_session() — every 6 turns, compresses buffer to 1-2 sentence summary",
        "Both consolidation and summarisation are security-filtered (check_output + check_semantic)",
    ], S)

    story += section("Query Intent Classification", S)
    story += body(
        "_classify_query_intent() performs heuristic classification for RAG routing:", S)
    story += make_info_table([
        ("skip", "Short social phrases, 1-2 word commands without question words"),
        ("ambient", "Default — ambient RAG with tight distance threshold"),
        ("synthesis", "Compare/list-all patterns — wide retrieval, no multi-hop"),
        ("factual", "Question word + document reference word — full explicit RAG with multi-hop"),
    ], S)

    story += section("Conversation Prompt Assembly", S)
    story += body(
        "The engine assembles prompts in a specific order: date/time header, correction "
        "hints, last-session context (first 3 turns only), conversation buffer (summary + "
        "last 4 turns or all 16 raw turns), memory context, document RAG chunks, attached "
        "documents, and finally the user command with appropriate instructions. Capabilities "
        "summary is injected for self-awareness questions.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 10: CORE MODULE — document_retriever.py (NEW)
    # =========================================================================
    story += chapter_heading(10, "Core Module — document_retriever.py", S, doc)

    story += body(
        "DocumentRetriever (~420 lines) was extracted from memory.py to separate storage "
        "(CRUD) from retrieval intelligence. It implements the multi-stage RAG pipeline: "
        "multi-query expansion, text-match fallback, RRF fusion, Jaccard deduplication, "
        "cross-encoder reranking, and multi-hop entity extraction.", S)

    story += section("Constructor", S)
    story += body(
        "DocumentRetriever takes a ChromaDB collection, embedding model name, and "
        "reranker model name. MemorySystem creates it at init and delegates all "
        "get_document_context() calls to it.", S)

    story += section("Retrieval Pipeline", S)
    story += make_step_table([
        ("1", "Primary query: ChromaDB semantic search", "embed_queries() + distance filter"),
        ("2", "Alt-query union (explicit mode)", "LLM-generated alternate phrasings, deduplicated"),
        ("3", "$contains text-match fallback (explicit mode)", "Keyword extraction, case variants"),
        ("4", "RRF fusion", "Reciprocal Rank Fusion of semantic + keyword pools (k=60)"),
        ("5", "Jaccard deduplication", "Remove near-duplicate chunks (threshold 0.85)"),
        ("6", "Cross-encoder reranking (explicit mode)", "bge-reranker-base, min_score=-1.0"),
        ("7", "Multi-hop entity extraction (explicit+factual)", "Parse entity names from top chunks"),
        ("8", "Second-hop query", "Metadata lookup + semantic fallback for related entities"),
        ("9", "Format and return", "Preamble + source citations with page numbers"),
    ], S)

    story += section("RRF Fusion", S)
    story += body(
        "_rrf_fuse(list_a, list_b, k=60) implements Reciprocal Rank Fusion. Semantic hits "
        "(distance &lt; 1.0) and keyword hits (artificial distance 1.0) are ranked separately, "
        "then combined using RRF scores: score = sum(1/(k + rank_i)) for each list. Chunks "
        "appearing in both lists receive contributions from both rankings.", S)

    story += section("Jaccard Deduplication", S)
    story += body(
        "_jaccard_dedup(chunks, threshold=0.85) removes near-duplicate chunks from the same "
        "source document. Word-set Jaccard similarity is computed between chunks sharing the "
        "same filename. Higher-ranked chunks (earlier in list) are always kept.", S)

    story += section("Multi-Hop Entity Extraction", S)
    story += body(
        "When multi_hop=True, entity names are parsed from the top 3 chunks via "
        "_parse_entity_names_from_chunk(). This checks for an 'entity_names' metadata "
        "field first, then falls back to parsing [METADATA: {...}] blocks. A second-hop "
        "query fires for each extracted entity (up to 6), using metadata field lookup "
        "first and semantic search as fallback. Hop chunks are discounted by inflating "
        "their distance before RRF fusion.", S)

    story += section("Explicit RAG Preamble", S)
    story += body(
        "In explicit mode, the preamble instructs the LLM to prioritize document content, "
        "cite source filenames, and report only values explicitly present in the excerpts. "
        "For any specific stat, number, or rule not found in excerpts, the LLM is told to "
        "say it was not found rather than substituting from general knowledge. This prevents "
        "training-knowledge contamination.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 11: CORE MODULE — llm_server.py
    # =========================================================================
    story += chapter_heading(11, "Core Module — llm_server.py", S, doc)

    story += body(
        "LLMServerManager handles the complete lifecycle of a local llama-server.exe "
        "process: auto-download from GitHub, launch, health polling, and graceful shutdown. "
        "Used only in 'builtin' mode (llm_server.mode = 'builtin' in settings.json).", S)

    story += section("Server Lifecycle States", S)
    story += make_flow_table([
        "stopped  -->  starting  -->  running",
        "                |                |",
        "                v                v",
        "             error           stopped",
        "                |",
        "                v",
        "          (on_error callback fired)",
    ], S)

    story += section("Auto-Download Logic", S)
    story += body(
        "download_server() queries the GitHub releases API for the latest llama.cpp "
        "release and selects the best Windows binary using a priority chain:", S)
    story += bullets([
        "Priority 1: avx2 build (best general CPU compatibility)",
        "Priority 2: vulkan build (GPU support, different DLL structure)",
        "Priority 3: noavx build (broadest CPU compatibility)",
        "Priority 4: any Windows ZIP without 'cuda' in the name",
        "Priority 5: any Windows ZIP (last resort)",
    ], S)

    story += section("Health Polling", S)
    story += body(
        "_poll_health() runs in a background thread after process launch. "
        "It polls GET /health every 2 seconds for up to 120 seconds. "
        "When the response contains {'status': 'ok'}, status is set to 'running' "
        "and the on_ready callback is fired. On process death or timeout, "
        "status is set to 'error' and on_error is called with stderr output.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 12: CORE MODULE — security.py
    # =========================================================================
    story += chapter_heading(12, "Core Module — security.py", S, doc)

    story += body(
        "SecurityFilter implements a layered, configurable security system for Talon. "
        "Controls raise the minimum sophistication required for a successful attack. "
        "The real load-bearing security sits in capability isolation (talents can only "
        "call their own APIs) and human confirmation gates. Since April 2026, "
        "wrap_external() lives in this module as a public function.", S)

    story += section("Design Principles", S)
    story += bullets([
        "All controls fail open: when disabled, processing continues silently",
        "Each control is independently configurable (enabled, action, per-pattern toggles)",
        "Hot-reload via reload() — no restart required for config changes",
        "Audit log persists all alert events to SQLite security_alerts table",
    ], S)

    story += section("The Four Controls", S)
    story += make_layer_table(
        ["Control", "Purpose", "Actions"],
        [
            ["Input Filter", "Regex pattern scan on incoming commands", "log or block"],
            ["Output Scan", "System prompt leak, API key exposure, encoded content detection",
             "log or suppress"],
            ["Rate Limiter", "Sliding-window per-minute request counter",
             "log or block; _executing_rule bypasses"],
            ["Confirmation Gates", "Named gate registry for irreversible actions",
             "block (always, unless disabled)"],
        ],
        S, col_widths=[2.8*cm, 6*cm, PAGE_W - 2*MARGIN - 8.8*cm - 10],
    )

    story += section("wrap_external()", S)
    story += body(
        "wrap_external(content, source_label) wraps untrusted external content "
        "(web search results, email bodies, document chunks) in structural markers. "
        "Square brackets inside the content are replaced with parentheses to prevent "
        "delimiter spoofing. Formerly _wrap_external in assistant.py, now a public "
        "function in security.py.", S)

    story += section("Integration Points", S)
    story += make_layer_table(
        ["Location", "Check", "Risk Context"],
        [
            ["process_command() entry", "check_rate_limit() + check_input()", "Injection watermark"],
            ["process_command() exit", "check_output()", "talent / conversation"],
            ["_consolidate_evicted_turn()", "check_output() + check_semantic()", "Cross-session risk (highest)"],
            ["_async_summarize_session()", "check_output() + check_semantic()", "Session-scoped risk"],
            ["update_settings()", "security.reload()", "Hot-reload on settings save"],
        ],
        S, col_widths=[4.5*cm, 4.5*cm, PAGE_W - 2*MARGIN - 9*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 13: CORE MODULE — voice.py
    # =========================================================================
    story += chapter_heading(13, "Core Module — voice.py", S, doc)

    story += body(
        "VoiceSystem handles all audio I/O: speech recognition via faster-whisper (Whisper) "
        "and text-to-speech via edge-tts (Microsoft Azure TTS, offline-capable once cached).", S)

    story += section("Wake Word Loop", S)
    story += make_step_table([
        ("1", "record_audio(chunk_duration=3s)", "Continuous loop"),
        ("2", "Check energy + variance thresholds", "Skip if silence"),
        ("3", "transcribe_audio() via Whisper", ""),
        ("4", "Skip noise_words ('you', 'thank you', etc.)", ""),
        ("5", "If any wake_word in transcription: speak('Ready')", "'talon', 'talent' aliases"),
        ("6", "record_audio(command_duration=5s)", "Command capture window"),
        ("7", "Transcribe and pass to command_callback", ""),
    ], S)

    story += section("Key Details", S)
    story += make_info_table([
        ("Wake words", "'okay talon', 'talon', 'hey talon', 'okay talent', 'talent' "
         "(Whisper mishearing aliases)"),
        ("TTS pronunciation", "'Talon' replaced with 'talun' for correct TAL-un sound"),
        ("TTS interruption", "stop_speaking() sets threading.Event + sd.stop()"),
        ("Whisper loading", "CUDA float16 preferred, CPU int8 fallback; size: configurable (default small)"),
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 14: vision.py & document_extractor.py
    # =========================================================================
    story += chapter_heading(14, "Core Module — vision.py &amp; document_extractor.py", S, doc)

    story += section("vision.py — VisionSystem", S)
    story += make_info_table([
        ("capture_screenshot()", "PIL.ImageGrab full-screen capture, base64 PNG"),
        ("load_image_file(path)", "Any PIL-supported format, converts to RGB, base64 PNG"),
    ], S)
    story += body(
        "Both methods return base64-encoded PNG strings compatible with the LLMClient "
        "vision API (images_b64 list parameter).", S)

    story += section("document_extractor.py — Inline Document Review", S)
    story += body(
        "Extracts plain text from common document formats for one-shot inline LLM review "
        "(not RAG ingestion). When a user attaches a file to the GUI chat, this module "
        "extracts the text and injects it into the LLM prompt via wrap_external().", S)
    story += make_layer_table(
        ["Format", "Library", "Notes"],
        [
            [".pdf", "PyMuPDF (fitz)", "Page-by-page text extraction"],
            [".docx", "python-docx", "Paragraphs and table cells"],
            [".xlsx/.xls", "pandas + openpyxl", "One section per sheet"],
            [".pptx", "python-pptx", "Text from all slide shapes"],
            [".csv", "pandas", "String representation"],
            [".txt/.md/.rst", "Direct read", "No library needed"],
        ],
        S, col_widths=[2*cm, 3*cm, PAGE_W - 2*MARGIN - 5*cm - 10],
    )
    story += note("MAX_CHARS = 32,000 (~8,000 tokens). Documents exceeding this are "
                   "truncated with a note suggesting ingest_documents.py for full RAG access.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 15: logging_config.py & config.py (NEW)
    # =========================================================================
    story += chapter_heading(15, "Core Module — logging_config.py &amp; config.py", S, doc)

    story += section("logging_config.py — Structured Logging", S)
    story += body(
        "setup_logging() initialises Python's logging framework with two handlers, "
        "replacing scattered print() calls with structured, level-filtered output. "
        "Called once at startup from main.py.", S)

    story += make_layer_table(
        ["Handler", "Level", "Format", "Destination"],
        [
            ["Console", "INFO+", "%(message)s (clean, like print())", "sys.stdout"],
            ["Rotating File", "DEBUG+",
             "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
             "data/logs/talon.log"],
        ],
        S, col_widths=[2.5*cm, 1.5*cm, 5.5*cm, PAGE_W - 2*MARGIN - 9.5*cm - 10],
    )

    story += subsection("File Rotation", S)
    story += make_info_table([
        ("Max file size", "5 MB per log file"),
        ("Backup count", "3 rotated backups (talon.log.1, .2, .3)"),
        ("Encoding", "UTF-8"),
    ], S)

    story += subsection("Third-Party Logger Quieting", S)
    story += body(
        "The following loggers are set to WARNING level to reduce noise: "
        "chromadb, httpx, urllib3, sentence_transformers, transformers, torch, "
        "onnxruntime, httpcore.", S)

    story += section("config.py — Configuration Utilities", S)
    story += body(
        "config.py contains a single utility function:", S)
    story += code_block("""\
def deep_merge(base: dict, override: dict) -> dict:
    \"\"\"Recursively merge override into base, returning a new dict.
    Keys in base but missing from override keep their base value.\"\"\"
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) \\
           and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result""", S)
    story += body(
        "This was extracted from assistant.py's _deep_merge() to make it reusable "
        "across the codebase. Used when merging settings.example.json defaults with "
        "user-provided settings.json overrides.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 16: scheduler.py, credential_store.py & chat_store.py
    # =========================================================================
    story += chapter_heading(16,
        "Core Module — scheduler.py, credential_store.py &amp; chat_store.py", S, doc)

    story += section("scheduler.py — Scheduler", S)
    story += body(
        "Lightweight cron-style background task scheduler. Reads a schedule list from "
        "settings.json and fires commands via assistant.process_command() at configured times.", S)
    story += make_info_table([
        ("Poll interval", "Every 20 seconds"),
        ("Fire guard", "At most once per day per (date, time, command) triple"),
        ("Day matching", "3-letter abbreviations: mon, tue, wed, thu, fri, sat, sun"),
        ("Execution", "Each fired command runs in its own daemon thread"),
    ], S)

    story += section("credential_store.py — CredentialStore", S)
    story += body(
        "Wraps the OS keyring library for unified talent credential storage. "
        "Secrets are stored under service name 'talon_assistant' with username "
        "format '{talent_name}.{field_key}'.", S)
    story += make_layer_table(
        ["Method", "Description"],
        [
            ["store_secret()", "Write to OS keyring"],
            ["get_secret()", "Read from OS keyring (returns '' if missing)"],
            ["delete_secret()", "Remove from keyring"],
            ["has_secret()", "Check existence without returning value"],
            ["migrate_legacy_email()", "One-time migration from old 'talon_email' service"],
        ],
        S, col_widths=[4*cm, PAGE_W - 2*MARGIN - 4*cm - 10],
    )

    story += section("chat_store.py — ChatStore", S)
    story += body(
        "Persists conversation sessions to data/conversations/ as JSON files. "
        "Each session is a list of ChatMessage dataclass instances (role, text, timestamp).", S)
    story += make_layer_table(
        ["Method", "Description"],
        [
            ["save_conversation()", "Write JSON session file"],
            ["load_conversation()", "Read and return ChatMessage list"],
            ["list_conversations()", "Sorted list of metadata dicts with preview"],
            ["export_as_text()", "Plain text format export"],
            ["export_as_markdown()", "Markdown with role headers"],
            ["delete_conversation()", "Remove session file"],
        ],
        S, col_widths=[4*cm, PAGE_W - 2*MARGIN - 4*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 17: embeddings, reranker, training_harvester, marketplace
    # =========================================================================
    story += chapter_heading(17,
        "Core Module — embeddings.py, reranker.py, training_harvester.py &amp; marketplace.py", S, doc)

    story += section("embeddings.py", S)
    story += body(
        "Module-level singleton wrapper for the BGE SentenceTransformer embedding model. "
        "Lazy-loaded on first use, cached globally. Runs on CPU to avoid competing "
        "with KoboldCpp/llama.cpp for VRAM.", S)
    story += make_info_table([
        ("embed_documents(texts)", "No prefix; used at ingest/add time"),
        ("embed_queries(texts)", "Adds BGE query prefix for asymmetric retrieval"),
        ("Query prefix", "'Represent this sentence for searching relevant passages: '"),
    ], S)

    story += section("reranker.py", S)
    story += body(
        "Module-level singleton CrossEncoder reranker (BAAI/bge-reranker-base). "
        "Called after initial vector retrieval in explicit RAG mode to rerank "
        "(query, chunk) pairs by joint relevance score.", S)
    story += make_info_table([
        ("API", "rerank(query, chunks, model_name, top_k=8, min_score=-2.0)"),
        ("Input", "list of (filename, text, dist, page_num) tuples"),
        ("Scores", "Raw logits: irrelevant -10 to -3, borderline -3 to 0, relevant > 0"),
        ("Default min_score", "-2.0 (drops noise, preserves borderline matches)"),
    ], S)

    story += section("training_harvester.py", S)
    story += body(
        "append_training_pair(instruction, output, source) appends one Alpaca-format "
        "record to data/training_pairs.jsonl. Sources: 'correction' or 'web_search'. "
        "Deduplicates by exact instruction match via linear file scan.", S)

    story += section("marketplace.py", S)
    story += body(
        "Talent discovery, browsing, and installation from a configured remote repository. "
        "Downloads talent .py files, validates them as BaseTalent subclasses, registers "
        "with the running assistant via load_user_talent(). The GUI marketplace dialog "
        "surfaces search and install without requiring an app restart.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 18: GUI — main_window.py & assistant_bridge.py
    # =========================================================================
    story += chapter_heading(18, "GUI — main_window.py &amp; assistant_bridge.py", S, doc)

    story += section("MainWindow (main_window.py)", S)
    story += body(
        "MainWindow is the top-level PyQt6 QMainWindow. It contains the main chat "
        "interface, talent sidebar, activity log, and menu bar. It is created after "
        "TalonAssistant is fully initialised to avoid segfaults from model loading "
        "conflicting with Qt platform plugin initialisation.", S)

    story += subsection("Layout", S)
    story += make_layer_table(
        ["Region", "Component"],
        [
            ["Left", "TalentSidebar — lists enabled talents with icons"],
            ["Centre", "ChatView — message bubbles for user/assistant exchanges"],
            ["Right", "ActivityLog — live print() output stream"],
            ["Bottom", "TextInput — command input with file attachment button"],
            ["Top", "Menu bar — File (Settings, LLM Server, Export), Help"],
        ],
        S, col_widths=[2*cm, PAGE_W - 2*MARGIN - 2*cm - 10],
    )

    story += section("AssistantBridge (assistant_bridge.py)", S)
    story += body(
        "AssistantBridge is a QObject that decouples the GUI from TalonAssistant. "
        "All assistant calls run in CommandWorker (QThread) to keep the GUI responsive. "
        "Results are delivered back to the GUI via Qt signals.", S)

    story += subsection("Signal Flow", S)
    story += make_step_table([
        ("1", "GUI: bridge.process_command(cmd)", "Main thread"),
        ("2", "Create CommandWorker", "Spawns worker thread"),
        ("3", "Worker: assistant.process_command()", "1-30 seconds"),
        ("4", "response_ready signal", "Worker thread to main thread"),
        ("5", "Update chat view + TTS speak", "Main thread"),
    ], S)

    story += subsection("update_settings() Hot-Swap", S)
    story += body(
        "When settings are saved, the settings_saved signal flows: "
        "SettingsDialog -> AssistantBridge.update_settings(). This method "
        "hot-swaps: LLM parameters, voice settings, audio thresholds, desktop config, "
        "and security filter config (reload()). Some settings require restart "
        "(Whisper model, embedding model, DB paths).", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 19: GUI — workers, tray, theme, interceptor
    # =========================================================================
    story += chapter_heading(19,
        "GUI — workers.py, system_tray.py, theme_manager.py &amp; output_interceptor.py", S, doc)

    story += section("workers.py", S)
    story += make_layer_table(
        ["Worker", "Description"],
        [
            ["InitWorker (QThread)", "Loads TalonAssistant off the GUI thread. Emits "
             "finished(assistant) or error(str). main.py pre-initialises before QApplication."],
            ["CommandWorker (QThread)", "Runs process_command() off the GUI thread. "
             "Emits response_ready(command, response, talent_name, success, result_dict)."],
            ["VoiceCommandWorker", "Voice command recording and transcription off-thread."],
        ],
        S, col_widths=[3.5*cm, PAGE_W - 2*MARGIN - 3.5*cm - 10],
    )

    story += section("system_tray.py — SystemTrayManager", S)
    story += make_info_table([
        ("Tray icon", "Programmatically generated 64x64 blue circle with 'J' glyph"),
        ("Global hotkey", "Ctrl+Shift+J via Win32 RegisterHotKey API (daemon thread)"),
        ("Hotkey mechanism", "WM_HOTKEY message loop (not pynput — removed, crash-on-exit issues)"),
        ("cleanup()", "Posts WM_QUIT to hotkey thread, unhides tray icon"),
    ], S)

    story += section("theme_manager.py — ThemeManager", S)
    story += body(
        "Loads QSS stylesheets from gui/styles/. Supports dark and light themes. "
        "Font size scaling applies a regex-based multiplier to all font-size: Npx "
        "declarations in the QSS. Persists preference to settings.json. "
        "Emits theme_changed signal when switched.", S)

    story += section("output_interceptor.py — OutputInterceptor", S)
    story += body(
        "Replaces sys.stdout to capture all print() output as Qt signals for the "
        "activity log. Thread-safe: write() uses QMetaObject.invokeMethod with "
        "QueuedConnection so the signal always delivers on the GUI thread. "
        "Dual-writes to the original stdout for terminal visibility.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 20: TALENT SYSTEM — base.py & planner.py
    # =========================================================================
    story += chapter_heading(20, "Talent System — base.py &amp; planner.py", S, doc)

    story += section("BaseTalent ABC (base.py, ~427 lines)", S)
    story += body(
        "All Talon talents inherit from BaseTalent. Subclasses must define class-level "
        "attributes and implement execute(). The talent system is discovered automatically "
        "via pkgutil.iter_modules at startup.", S)

    story += subsection("Required Class Attributes", S)
    story += make_layer_table(
        ["Attribute", "Type", "Description"],
        [
            ["name", "str", "Unique identifier (used by router)"],
            ["description", "str", "What this talent does (shown to LLM router)"],
            ["examples", "list[str]", "Natural-language example commands (primary routing)"],
            ["keywords", "list[str]", "Fallback trigger words (degraded mode only)"],
            ["priority", "int", "Sidebar ordering (higher = more prominent)"],
            ["subprocess_isolated", "bool", "Run execute() in child process (for C extensions)"],
            ["required_packages", "list[str]", "Pip packages checked at load time"],
            ["required_config", "list[str]", "settings.json keys (dot-notation) that must exist"],
            ["required_env", "list[str]", "OS env var names that must be set"],
        ],
        S, col_widths=[3.5*cm, 2*cm, PAGE_W - 2*MARGIN - 5.5*cm - 10],
    )

    story += subsection("TalentContext Dataclass", S)
    story += body(
        "TalentContext is a typed dataclass passed to execute(). It supports dict-style "
        "access (context['llm']) for backward compatibility. Fields: llm, memory, vision, "
        "voice, config, memory_context, speak_response, assistant, server_manager, "
        "rag_explicit, command_source, notify, attachments, _planner_substep.", S)

    story += subsection("TalentResult Dataclass", S)
    story += body(
        "TalentResult is a typed dataclass returned from execute(). Dict-style access "
        "supported. Fields: success (bool), response (str), actions_taken (list), "
        "spoken (bool).", S)

    story += subsection("Subprocess Isolation", S)
    story += body(
        "Talents that set subprocess_isolated = True have their execute() run in a child "
        "process via ProcessPoolExecutor. All C-extension state (yfinance, pandas, numpy) "
        "is destroyed when the child exits, preventing GIL/heap corruption in the host. "
        "The child receives a minimal context with only serialisable data (config dict). "
        "Per-talent config from talents.json is passed separately.", S)

    story += subsection("LLMError Handling Pattern", S)
    story += body(
        "Talents should catch LLMError around llm.generate() calls. The _extract_arg() "
        "helper already handles this internally, returning None on LLMError.", S)
    story += code_block("""\
try:
    response = llm.generate(prompt, max_length=200)
except LLMError:
    return TalentResult(success=False,
                        response="LLM unavailable.")""", S)

    story += subsection("_extract_arg() Helper", S)
    story += body(
        "_extract_arg(llm, command, what, options, max_length, temperature, fallback) "
        "is a standardised helper for extracting a single named value from a command "
        "string. One tight deterministic LLM call (temperature=0.0, max_length=20).", S)

    story += section("PlannerTalent (planner.py)", S)
    story += body(
        "PlannerTalent detects multi-step routines and breaks them into sequential "
        "sub-commands, each routed through the normal process_command() pipeline. "
        "Priority 85 — higher than all other built-in talents.", S)

    story += subsection("Planner Flow", S)
    story += make_step_table([
        ("1", "User: 'good morning routine'", ""),
        ("2", "LLM call with _PLANNER_SYSTEM_PROMPT (talent roster)", ""),
        ("3", "Returns: {is_multi_step: true, steps: [...]}", ""),
        ("4", "Execute each step via process_command(step, _executing_rule=True)", ""),
        ("5", "Collect responses, return summary", ""),
    ], S)

    story += subsection("Decline Detection", S)
    story += body(
        "If LLM returns {is_multi_step: false}, PlannerTalent returns success=False "
        "with blank response and no actions_taken. process_command() detects this "
        "decline pattern and falls through to conversation.", S)

    story += subsection("Talent Builder Flow", S)
    story += body(
        "New talents can be built through a conversational workflow: the user describes "
        "what they want, the LLM generates a BaseTalent subclass, and an iterative "
        "code review cycle refines the implementation before saving to talents/user/.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 21: TALENT — email_talent.py
    # =========================================================================
    story += chapter_heading(21, "Talent — email_talent.py", S, doc)

    story += body(
        "The email talent handles compose, send, read, and reply workflows for "
        "SMTP/IMAP email accounts. It demonstrates the scaffold + LLM intelligence "
        "pattern: Talon defines routing, config schema, and API calls; the LLM "
        "supplies email composition intelligence from its training.", S)

    story += section("Capabilities", S)
    story += bullets([
        "Compose and send new emails (to, subject, body inferred by LLM from natural language)",
        "Read inbox (IMAP, configurable folder, unread count, subject/sender preview)",
        "Reply to messages",
        "Draft emails for user review via GUI compose dialog (pending_email flow)",
        "Multi-account support via credential store",
    ], S)

    story += section("Configuration", S)
    story += make_info_table([
        ("email_address", "Sender email"),
        ("smtp_server / smtp_port", "Outgoing mail server"),
        ("imap_server / imap_port", "Incoming mail server"),
        ("password", "Stored in OS keyring, never in plaintext config"),
    ], S)

    story += section("Pending Email Flow", S)
    story += body(
        "When the LLM composes an email, the talent returns a 'pending_email' dict "
        "in the result. AssistantBridge detects this and opens the EmailComposeDialog, "
        "allowing the user to review and edit before sending. This implements the "
        "confirmation gate pattern for external_send actions.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 22: TALENT — reminder.py & notes.py (consolidated)
    # =========================================================================
    story += chapter_heading(22, "Talent — reminder.py &amp; notes.py", S, doc)

    story += section("reminder.py — Timed Reminders", S)
    story += body(
        "The reminder talent manages timed reminders with natural language time parsing. "
        "Reminders are stored in SQLite and checked by a background polling thread (every "
        "30 seconds). When a reminder fires, it triggers a desktop notification via the "
        "system tray and optionally speaks via TTS.", S)
    story += make_info_table([
        ("Time parsing", "LLM extracts: 'in 10 minutes', 'at 3pm', 'tomorrow at 9am'"),
        ("Storage", "SQLite: timestamp, message, fired status"),
        ("Polling", "Background thread every 30 seconds"),
    ], S)

    story += section("notes.py — Persistent Notes", S)
    story += body(
        "Persistent, semantically searchable notes system. Notes are stored in both "
        "SQLite (structured retrieval) and talon_notes ChromaDB (semantic search).", S)
    story += make_layer_table(
        ["Action", "Description"],
        [
            ["Save a note", "Store text with optional tags"],
            ["Find/search notes", "Semantic similarity search via ChromaDB"],
            ["List recent notes", "Retrieve latest N notes from SQLite"],
            ["Delete a note", "Remove from both SQLite and ChromaDB"],
            ["Tag-based filtering", "Filter notes by keyword tags"],
        ],
        S, col_widths=[3*cm, PAGE_W - 2*MARGIN - 3*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 23: hue_lights.py & desktop_control.py (consolidated)
    # =========================================================================
    story += chapter_heading(23, "Talent — hue_lights.py &amp; desktop_control.py", S, doc)

    story += section("hue_lights.py — Philips Hue", S)
    story += body(
        "Controls smart lights via the local Hue Bridge REST API. Uses XY colour space "
        "for colour accuracy, brightness 0-254, scenes, groups, and individual light control.", S)
    story += make_info_table([
        ("Colour space", "CIE 1931 XY chromaticity coordinates (0.0-1.0)"),
        ("bridge_ip", "Local IP address of the Hue Bridge"),
        ("api_key", "Hue API user key (obtained via Bridge pairing)"),
        ("default_room", "Default group/room name for ambiguous commands"),
        ("routing_available", "Returns False when bridge not configured (removes from roster)"),
    ], S)

    story += section("desktop_control.py — Desktop Automation", S)
    story += body(
        "Natural language desktop automation via PyAutoGUI. Supports vision-guided "
        "interaction (LLM sees screenshot and determines click coordinates), keyboard "
        "input, application launching, and window management.", S)
    story += make_layer_table(
        ["Action", "Description"],
        [
            ["click / double_click / right_click", "At position or on element (vision-guided)"],
            ["type_text", "Keyboard input via pyautogui.typewrite()"],
            ["key_press", "Hotkeys and special keys (ctrl+c, win+d, etc.)"],
            ["open_application", "Launch app by name"],
            ["screenshot and describe", "Take screenshot, ask LLM to describe content"],
            ["scroll", "Mouse scroll at position"],
        ],
        S, col_widths=[4*cm, PAGE_W - 2*MARGIN - 4*cm - 10],
    )

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 24: news.py & news_digest.py
    # =========================================================================
    story += chapter_heading(24, "Talent — news.py &amp; news_digest.py", S, doc)

    story += section("news.py — Web Search &amp; News", S)
    story += body(
        "Handles real-time web searches and news queries. Uses DuckDuckGo as the "
        "primary search backend (no API key required) with trafilatura for full "
        "article text extraction.", S)
    story += bullets([
        "Web search — query DuckDuckGo, extract top results",
        "Article fetch — trafilatura scrapes readable text from URLs",
        "RSS lookup — per-domain overrides in settings.json web_browser.rss_feeds",
        "Results wrapped with wrap_external() before LLM injection",
        "Training pairs harvested from successful web search responses",
    ], S)

    story += section("news_digest.py — Morning Digest", S)
    story += body(
        "Generates a structured morning briefing from configured RSS feeds. "
        "Feed configuration lives in config/news_digest.json (separate from "
        "settings.json). The digest is triggered by the scheduler at a configured "
        "time and summarised by the LLM.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 25: rules.py, history.py, clipboard_transform.py
    # =========================================================================
    story += chapter_heading(25,
        "Talent — rules.py, history.py &amp; clipboard_transform.py", S, doc)

    story += section("rules.py — Rule Management", S)
    story += body(
        "User interface for managing behavioral rules: list, enable/disable, delete, "
        "view details. The underlying rule storage and fast-path matching live in "
        "memory.py and assistant.py.", S)

    story += section("history.py — Command History", S)
    story += body(
        "Retrieves and presents command history from SQLite. Commands: 'show my history', "
        "'what did I ask recently', 'search history for X'. Formatted with timestamps, "
        "success indicators, and response previews.", S)

    story += section("clipboard_transform.py — Clipboard Transform", S)
    story += body(
        "Reads clipboard content, sends to LLM with transformation instruction "
        "(summarise, rephrase, translate, fix grammar), writes result back to clipboard. "
        "Examples: 'make this more formal', 'translate clipboard to French'.", S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 26: lora_train.py & signal_remote.py (consolidated)
    # =========================================================================
    story += chapter_heading(26, "Talent — lora_train.py &amp; signal_remote.py", S, doc)

    story += section("lora_train.py — LoRA Fine-Tuning", S)
    story += body(
        "Complete LoRA fine-tuning workflow integrated into Talon. Bridges the "
        "training pair harvester with external training frameworks (Unsloth/axolotl).", S)
    story += make_step_table([
        ("1", "User: 'start lora training'", ""),
        ("2", "Read training_pairs.jsonl", "Validate minimum pair count"),
        ("3", "Write Alpaca-format training config YAML", ""),
        ("4", "Launch training subprocess", "Unsloth or axolotl"),
        ("5", "Monitor stdout for progress (loss, epoch)", ""),
        ("6", "Adapter saved to lora_path", ""),
        ("7", "If builtin mode: offer to reload with new adapter", ""),
    ], S)

    story += section("signal_remote.py — Signal Messenger", S)
    story += body(
        "Integrates with Signal encrypted messenger via signal-cli. Bidirectional: "
        "Talon can send Signal messages and receive/process incoming messages.", S)
    story += make_info_table([
        ("signal_number", "Registered Signal phone number"),
        ("signal_cli_path", "Path to signal-cli executable"),
        ("allowed_senders", "Whitelist of phone numbers that can send commands"),
        ("Background polling", "Uses set_assistant() to start polling thread; "
         "routes incoming messages through process_command()"),
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # CHAPTER 27: TESTING (NEW)
    # =========================================================================
    story += chapter_heading(27, "Testing", S, doc)

    story += body(
        "Talon includes a pytest suite of 97 tests across 7 test files in the tests/ "
        "directory. All tests are fully mocked (no LLM, no ChromaDB, no network) and "
        "run in approximately 12 seconds.", S)

    story += section("Test Files", S)
    story += make_layer_table(
        ["File", "Tests", "Coverage Area"],
        [
            ["test_config.py", "10", "deep_merge() edge cases, nested dicts, overrides"],
            ["test_security.py", "17",
             "Input filter, output scan, rate limiter, confirmation gates, injection patterns"],
            ["test_llm_client.py", "15",
             "All three backends, LLMError raising, vision payload, ready-guard"],
            ["test_base_talent.py", "15",
             "TalentContext/TalentResult dataclasses, _extract_arg, keyword_match, "
             "subprocess isolation, required_packages check"],
            ["test_memory.py", "20",
             "SQLite CRUD, ChromaDB collection ops, rules sync, WAL mode, "
             "preference storage, document context delegation"],
            ["test_conversation.py", "9",
             "Fast-paths (time, date, system facts), intent classification, "
             "buffer management, session summarisation trigger, promise detection"],
            ["test_assistant_routing.py", "11",
             "LLM routing, keyword fallback, talent discovery, correction detection, "
             "rule matching, _executing_rule flag behaviour"],
        ],
        S, col_widths=[4*cm, 1.2*cm, PAGE_W - 2*MARGIN - 5.2*cm - 10],
    )

    story += section("Running Tests", S)
    story += code_block("python -m pytest tests/ -v", S)

    story += section("Test Patterns", S)
    story += bullets([
        "All LLM calls are mocked via unittest.mock.patch on llm.generate()",
        "ChromaDB collections are replaced with MagicMock objects",
        "SQLite uses in-memory databases (:memory:) for isolation",
        "No network, no CUDA, no model loading — tests run on any machine",
        "Fixtures in conftest.py provide pre-configured mock instances",
    ], S)

    story += section("Key Assertions", S)
    story += bullets([
        "LLMError is raised (not returned as string) on all three backends",
        "Rate limiter correctly blocks after N requests in sliding window",
        "Injection patterns are detected and blocked/logged as configured",
        "Talent routing returns correct talent for example commands",
        "Correction detection strips prefix and re-routes correctly",
        "Time/date fast-paths return formatted strings without LLM calls",
        "Buffer summarisation triggers every 6 turns",
        "deep_merge preserves nested base keys missing from override",
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # APPENDIX A: Configuration Reference
    # =========================================================================
    story.append(Paragraph("APPENDIX A", S["chapter_label"]))
    story.append(Paragraph("Configuration Reference", S["h1"]))
    story += divider(S)

    story += body(
        "All configuration lives in config/settings.json (gitignored). "
        "The committed template is config/settings.example.json. "
        "Defaults are merged with user overrides via deep_merge() from core/config.py.", S)

    story += section("llm — LLM Client Settings", S)
    story += make_config_table([
        ("endpoint", "string", "LLM API URL"),
        ("api_format", "string", "'koboldcpp' | 'llamacpp' | 'openai'"),
        ("max_length", "int", "Max tokens to generate per call (default 512)"),
        ("temperature", "float", "Sampling temperature (0.0-2.0, default 0.7)"),
        ("top_p", "float", "Nucleus sampling (0.0-1.0, default 0.9)"),
        ("rep_pen", "float", "Repetition penalty (1.0+, default 1.1)"),
        ("timeout", "int", "HTTP timeout in seconds (default 120)"),
        ("stop_sequences", "list", "Token stop strings"),
        ("prompt_template", "object", "user_prefix, user_suffix, assistant_prefix, vision_prefix"),
    ], S)

    story += section("llm_server — Built-in Server Settings", S)
    story += make_config_table([
        ("mode", "string", "'builtin' | 'external'"),
        ("model_path", "string", "Path to GGUF model file"),
        ("mmproj_path", "string", "Path to multimodal projector (vision)"),
        ("lora_path", "string", "Path to LoRA adapter file"),
        ("port", "int", "Server port (default 8080)"),
        ("n_gpu_layers", "int", "GPU layers (-1 = all, 0 = CPU only)"),
        ("ctx_size", "int", "Context window size (default 12288)"),
        ("threads", "int", "CPU threads (default 4)"),
        ("bin_path", "string", "Directory containing llama-server.exe"),
        ("extra_args", "string", "Additional CLI flags"),
    ], S)

    story += section("audio — Audio Capture Settings", S)
    story += make_config_table([
        ("sample_rate", "int", "Audio sample rate (default 16000)"),
        ("chunk_duration", "int", "Wake word listen window in seconds (default 3)"),
        ("command_duration", "int", "Command capture window in seconds (default 5)"),
        ("energy_threshold", "int", "Min energy to process audio (default 100)"),
        ("variance_threshold", "int", "Min variance to process audio (default 1000)"),
        ("noise_words", "list", "Short words to discard ('you', 'thanks', etc.)"),
    ], S)

    story += section("security — Security Filter Settings", S)

    story += subsection("input_filter", S)
    story += make_config_table([
        ("enabled", "bool", "Enable/disable input pattern scanning"),
        ("action", "string", "'log' | 'block'"),
        ("patterns", "list", "[{id, enabled, builtin, label, pattern (regex)}]"),
    ], S)

    story += subsection("output_scan", S)
    story += make_config_table([
        ("enabled", "bool", "Enable/disable output scanning"),
        ("action", "string", "'log' | 'suppress'"),
        ("checks", "list", "[{id, enabled, builtin, label}] — ids: prompt_leak, api_keys, encoded_content"),
    ], S)

    story += subsection("rate_limit", S)
    story += make_config_table([
        ("enabled", "bool", "Enable/disable rate limiting"),
        ("action", "string", "'log' | 'block'"),
        ("requests_per_minute", "int", "Sliding window limit (default 30)"),
    ], S)

    story += subsection("confirmation_gates", S)
    story += make_config_table([
        ("enabled", "bool", "Enable/disable confirmation gates"),
        ("action", "string", "'block' (gates always block unless disabled)"),
        ("gates", "list", "[{id, enabled, builtin, label}] — ids: destructive_file_ops, rule_writes, external_send"),
    ], S)

    story += subsection("audit_log", S)
    story += make_config_table([
        ("enabled", "bool", "Write alerts to SQLite security_alerts table"),
        ("level", "string", "'minimal' | 'standard' | 'verbose'"),
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # APPENDIX B: Adding a New Talent
    # =========================================================================
    story.append(Paragraph("APPENDIX B", S["chapter_label"]))
    story.append(Paragraph("Adding a New Talent", S["h1"]))
    story += divider(S)

    story += body(
        "Talents are auto-discovered from the talents/ and talents/user/ directories. "
        "To add a new talent, create a .py file in talents/user/ with a BaseTalent subclass. "
        "New talents can also be built through the conversational talent builder flow: "
        "describe what you want, the LLM generates a BaseTalent subclass, and iterative "
        "code review refines the implementation.", S)

    story += section("Step 1: Create the Talent File", S)
    story += code_block("""\
# talents/user/my_talent.py
from talents.base import BaseTalent, TalentResult
from core.llm_client import LLMError

class MyTalent(BaseTalent):
    name        = "my_talent"
    description = "Does something specific when the user asks"
    examples    = [
        "do the specific thing",
        "run my thing for X",
        "activate my feature",
    ]
    keywords    = ["specific", "thing"]   # fallback only
    priority    = 50

    # Set True for talents loading C extensions (yfinance, pandas, etc.)
    subprocess_isolated = False

    # Pip packages beyond base requirements (checked at load time)
    required_packages = []

    def execute(self, command: str, context: dict) -> dict:
        llm    = context["llm"]
        config = context["config"]

        # Extract a parameter using the standardised helper
        target = self._extract_arg(llm, command, "target item")
        if not target:
            return TalentResult(
                success=False,
                response="I couldn't identify what to operate on.",
            )

        # Do the work (wrap LLM calls in LLMError handling)
        try:
            result = llm.generate(f"Process: {target}",
                                  max_length=200)
        except LLMError:
            return TalentResult(
                success=False,
                response="LLM is unavailable right now.",
            )

        return TalentResult(
            success=True,
            response=f"Done: {result}",
            actions_taken=[{"action": {"type": "my_action",
                                       "target": target},
                            "result": result, "success": True}],
        )""", S)

    story += section("Step 2: Add Configuration (Optional)", S)
    story += code_block("""\
def get_config_schema(self) -> dict:
    return {
        "fields": [
            {"key": "api_key",  "label": "API Key",
             "type": "password", "default": ""},
            {"key": "max_items","label": "Max Items",
             "type": "int", "default": 5, "min": 1, "max": 20},
            {"key": "mode",     "label": "Mode",
             "type": "choice", "default": "fast",
             "choices": ["fast", "thorough"]},
        ]
    }""", S)

    story += section("Step 3: Test Routing", S)
    story += body(
        "After placing the file in talents/user/, restart Talon. Test with commands "
        "matching your examples list. If routing fails, add more distinctive example "
        "phrases — the LLM router uses these to determine when your talent should "
        "handle a command.", S)

    story += section("Key Rules for Talent Development", S)
    story += bullets([
        "name must be unique — the LLM router matches by exact name",
        "examples are more important than keywords — the LLM uses examples for routing",
        "Return TalentResult (or dict with success, response, actions_taken, spoken keys)",
        "Use _extract_arg() for simple scalar extraction (temperature=0.0, max_length=20)",
        "Catch LLMError around all llm.generate() calls",
        "Set subprocess_isolated=True for talents with C extensions (numpy, pandas, yfinance)",
        "List pip dependencies in required_packages (auto-checked at load time)",
        "Use context['assistant'].process_command() to trigger other talents from within",
        "Override routing_available to return False when backend is unavailable",
    ], S)

    story.append(PageBreak())

    # =========================================================================
    # APPENDIX C: Extending the Security Filter
    # =========================================================================
    story.append(Paragraph("APPENDIX C", S["chapter_label"]))
    story.append(Paragraph("Extending the Security Filter", S["h1"]))
    story += divider(S)

    story += section("Adding a New Input Pattern", S)
    story += body("Option 1 — Via GUI (Settings > Security tab > Input Filter > Add):", S)
    story += bullets([
        "Click Add, enter a label and a Python regex pattern",
        "The pattern is saved to settings.json and hot-reloaded immediately",
        "Custom patterns have builtin: false — they can be removed, not just disabled",
    ], S)
    story += body("Option 2 — Directly in settings.json:", S)
    story += code_block("""\
"security": {
  "input_filter": {
    "patterns": [
      ...existing patterns...,
      {
        "id":      "my_pattern",
        "enabled": true,
        "builtin": false,
        "label":   "My custom detection",
        "pattern": "(?i)\\\\bmy trigger phrase\\\\b"
      }
    ]
  }
}""", S)

    story += section("Adding a New Output Check", S)
    story += body(
        "Output checks with custom logic require a code change in core/security.py. "
        "Add to the check_output() method:", S)
    story += code_block("""\
# In check_output() inside SecurityFilter:
chk = checks.get("my_custom_check", {})
if chk.get("enabled", False):
    if MY_PATTERN.search(text):
        alert = self._make_alert(
            "output_scan", "my_custom_check",
            "My custom check label",
            text[:300], action, extra=context,
        )
        self._record_alert(alert)
        return action == "suppress", alert""", S)
    story += body("Then add the check to DEFAULT_OUTPUT_CHECKS and settings.example.json.", S)

    story += section("Adding a New Confirmation Gate", S)
    story += body(
        "Add the gate to DEFAULT_CONFIRMATION_GATES in security.py and settings.example.json, "
        "then call security.gate_required('your_gate_id') in the talent or assistant code "
        "before executing the action. If True, surface a confirmation prompt.", S)
    story += code_block("""\
# In a talent's execute() method:
if context.get('assistant') and \\
   context['assistant'].security.gate_required('your_gate_id'):
    return TalentResult(
        success=False,
        response="This action requires confirmation.",
    )""", S)

    story += section("Security Filter API Reference", S)
    story += make_layer_table(
        ["Method", "Description"],
        [
            ["reload(config)", "Call after any config change; safe from any thread"],
            ["set_system_prompt_phrases(phrases)", "Seed leak detector with 5-15 distinctive phrases"],
            ["check_input(text)", "Returns (blocked, alert) — call at command entry point"],
            ["check_output(text, context)", "Returns (suppressed, alert) — before returning LLM output"],
            ["check_rate_limit()", "Returns (blocked, alert) — once per process_command()"],
            ["gate_required(gate_id)", "Returns bool — before any irreversible action"],
            ["wrap_external(content, source_label)", "Public function — wraps untrusted content"],
            ["recent_alerts", "Last 100 SecurityAlert objects (in-memory, current session)"],
        ],
        S, col_widths=[5.5*cm, PAGE_W - 2*MARGIN - 5.5*cm - 10],
    )

    # =========================================================================
    # Build the PDF
    # =========================================================================
    def first_page(canvas, doc):
        draw_cover_page(canvas, doc)

    def later_pages(canvas, doc):
        draw_header_footer(canvas, doc)

    doc.build(story, onFirstPage=first_page, onLaterPages=later_pages)
    return output_path


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    path = build_document()
    print(f"PDF generated: {path}")
