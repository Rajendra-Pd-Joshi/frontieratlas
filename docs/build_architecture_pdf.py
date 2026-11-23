"""Build architecture.pdf (max 3 pages) from docs/architecture.md.

Usage: .venv/bin/python docs/build_architecture_pdf.py
Writes architecture.pdf next to this script's project root.
Fails (exit 1) if the result exceeds 3 pages — trim the markdown, not
the font, if that happens.
"""
import re
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parent.parent
MD = ROOT / "docs" / "architecture.md"
OUT = ROOT / "architecture.pdf"

W, H = 215.9, 279.4  # US Letter, mm
MARGIN = 13


class Doc(FPDF):
    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, "FrontierAtlas - Technical Architecture", align="R")
        self.ln(8)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()} of {{nb}}", align="C")


def clean(text: str) -> str:
    # Built-in Helvetica is latin-1 only: replace unicode punctuation.
    return (text.replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u201c", '"').replace("\u201d", '"')
                .replace("\u2018", "'").replace("\u2019", "'")
                .replace("\u2265", ">=").replace("\u2264", "<=")
                .replace("\u2192", "->").replace("\u2026", "..."))


def render():
    pdf = Doc("P", "mm", (W, H))
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(True, margin=12)
    pdf.set_margins(MARGIN, 12, MARGIN)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 8, "FrontierAtlas - Technical Architecture", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, "GraphOne / FrontierAtlas AI Engineer demo task - Rajendra Prasad Joshi - 23 Nov 2026", new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    for line in MD.read_text(encoding="utf-8").splitlines():
        line = line.rstrip()
        if not line:
            pdf.ln(0.5)
            continue
        m = re.match(r"^# (.+)", line)
        if m:
            continue  # title already rendered
        m = re.match(r"^## (\d+)\. (.+)", line)
        if m:
            pdf.ln(1)
            pdf.set_font("Helvetica", "B", 10.5)
            pdf.cell(0, 6, clean(f"{m.group(1)}. {m.group(2)}"), new_x="LMARGIN", new_y="NEXT")
            pdf.ln(0.5)
            continue
        m = re.match(r"^- \*\*(.+?)\*\*:? ?(.*)$", line)
        if m:
            pdf.set_font("Helvetica", "", 9)
            x = pdf.get_x()
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(3, 5, "-")
            pdf.write(5, clean(m.group(1) + ": "))
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4.5, clean(m.group(2)))
            pdf.ln(0.5)
            continue
        m = re.match(r"^- (.+)$", line)
        if m:
            pdf.set_font("Helvetica", "", 9)
            pdf.cell(3, 5, "-")
            pdf.multi_cell(0, 4.5, clean(m.group(1)))
            pdf.ln(0.5)
            continue
        # Plain paragraph; strip markdown emphasis for the built-in font.
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        text = re.sub(r"`([^`]+)`", r"\1", text)
        pdf.set_font("Helvetica", "", 9)
        pdf.multi_cell(0, 4.5, clean(text))
        pdf.ln(0.5)

    pdf.output(str(OUT))
    print(f"pages: {pdf.pages_count} -> {OUT}")
    if pdf.pages_count > 3:
        print("ERROR: exceeds the 3-page limit — trim docs/architecture.md")
        sys.exit(1)


if __name__ == "__main__":
    render()
