"""Build architecture.pdf from docs/architecture.md (max 3 pages).

Layout (US Letter, 13mm margins):
- DejaVu Sans (regular + bold) so real glyphs render: em-dash, arrows,
  >=, bullet. No ASCII mangling of the source text.
- Title block + thin rule, numbered section headings in dark slate-blue,
  justified body text, hanging-indent bullets.
- Paragraphs are joined from the source file's hard-wrapped lines
  (blank line = paragraph boundary; indented lines continue a bullet),
  so text fills the full column width instead of breaking at ~72 chars.

Usage: .venv/bin/python docs/build_architecture_pdf.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "architecture.md"
OUT = ROOT / "architecture.pdf"

W, H = 215.9, 279.4          # US Letter, mm
MARGIN = 13
TITLE_COLOR = (31, 56, 100)  # dark slate-blue for headings
GREY = (90, 90, 90)

REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Body", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 6, "FrontierAtlas - Technical Architecture", align="R")
        self.ln(8)
        self.set_text_color(0, 0, 0)

    def footer(self):
        self.set_y(-12)
        self.set_font("Body", "", 8)
        self.set_text_color(*GREY)
        self.cell(0, 8, f"Page {self.page_no()} of {{nb}}", align="C")
        self.set_text_color(0, 0, 0)


def parse_blocks(text: str):
    """Yield ("h", title) / ("p", text) / ("b", text) blocks.

    Consecutive non-blank lines join into one paragraph; a line
    starting with "- " starts a bullet; indented lines continue it.
    """
    blocks = []
    para: list[str] = []
    bullet: list[str] = []

    def flush_para():
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    def flush_bullet():
        if bullet:
            blocks.append(("b", " ".join(bullet)))
            bullet.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush_para()
            flush_bullet()
            continue
        if line.startswith("# "):            # document title, rendered separately
            flush_para()
            flush_bullet()
            continue
        m = re.match(r"^## (\d+)\. (.+)", line)
        if m:
            flush_para()
            flush_bullet()
            blocks.append(("h", f"{m.group(1)}. {m.group(2)}"))
            continue
        if line.startswith("- "):
            flush_para()
            flush_bullet()
            bullet.append(line[2:].strip())
            continue
        if line.startswith("  ") and bullet:  # indented continuation of a bullet
            bullet.append(line.strip())
            continue
        if bullet:
            flush_bullet()
        para.append(line.strip())
    flush_para()
    flush_bullet()
    return blocks


def render():
    pdf = Doc("P", "mm", (W, H))
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=12)
    pdf.set_margins(MARGIN, 12, MARGIN)
    pdf.add_font("Body", "", REG)
    pdf.add_font("Body", "B", BOLD)
    pdf.add_page()

    # Title block
    pdf.set_font("Body", "B", 16)
    pdf.set_text_color(*TITLE_COLOR)
    pdf.cell(0, 9, "FrontierAtlas - Technical Architecture", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Body", "", 9.5)
    pdf.set_text_color(*GREY)
    pdf.cell(0, 6, "GraphOne / FrontierAtlas AI Engineer demo task - Rajendra Prasad Joshi - 23 Nov 2026",
             new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_draw_color(*TITLE_COLOR)
    pdf.set_line_width(0.6)
    pdf.line(MARGIN, pdf.get_y(), W - MARGIN, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)

    for kind, text in parse_blocks(MD.read_text()):
        # Strip backticks (code spans) but keep the code text itself.
        text = re.sub(r"`([^`]+)`", r"\1", text)
        if kind == "h":
            pdf.ln(1.5)
            pdf.set_font("Body", "B", 11.5)
            pdf.set_text_color(*TITLE_COLOR)
            pdf.cell(0, 6.5, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)
            pdf.set_text_color(0, 0, 0)
        elif kind == "p":
            pdf.set_font("Body", "", 9.5)
            pdf.multi_cell(0, 5.2, text, align="J", markdown=True)
            pdf.ln(1.2)
        else:  # bullet with hanging indent
            pdf.set_font("Body", "B", 9.5)
            pdf.cell(5, 5.2, "\u2022")
            pdf.set_font("Body", "", 9.5)
            pdf.multi_cell(0, 5.2, text, align="J", markdown=True)
            pdf.ln(0.8)

    if pdf.page_no() > 3:
        print(f"ERROR: {pdf.page_no()} pages - exceeds the 3-page limit", file=sys.stderr)
        sys.exit(1)
    pdf.output(str(OUT))
    print(f"pages: {pdf.page_no()} -> {OUT}")


if __name__ == "__main__":
    render()
