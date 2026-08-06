"""
report.py — Generates a shareable PDF report from a DeepFER session.

Turns an EmotionSession's on-screen dashboard into a standalone PDF:
summary KPIs, emotion distribution chart, satisfaction breakdown chart,
and the notable-moments list — something that can be emailed or saved,
not just viewed once in the browser.
"""

import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")  # no GUI backend needed — we only render to bytes
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle,
)

EMOTION_COLORS_HEX = {
    "happy": "#22C55E", "surprise": "#38BDF8", "neutral": "#94A3B8",
    "sad": "#3B82F6", "angry": "#EF4444", "fear": "#C026D3", "disgust": "#14B8A6",
}


def _render_bar_chart(data: dict, title: str) -> io.BytesIO:
    """Render a simple bar chart (emotion distribution or satisfaction
    breakdown) to an in-memory PNG buffer, ready to embed in the PDF.
    """
    labels = list(data.keys())
    values = list(data.values())
    bar_colors = [EMOTION_COLORS_HEX.get(label.lower(), "#6C5CE7") for label in labels]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(labels, values, color=bar_colors)
    ax.set_ylabel("Percent")
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf


def generate_pdf_report(session, report_title: str = "DeepFER Session Report") -> bytes:
    """Build a complete PDF report from a session's data.

    Args:
        session: an analytics.EmotionSession with data already recorded.
        report_title: shown as the document's main heading.

    Returns:
        PDF file content as bytes — pass directly to st.download_button.
    """
    summary = session.summary()
    moments = session.notable_moments()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                             topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DeepFERTitle", parent=styles["Title"], textColor=colors.HexColor("#341f97"),
    )
    subtitle_style = ParagraphStyle(
        "DeepFERSubtitle", parent=styles["Normal"], textColor=colors.grey, fontSize=10,
    )

    story = []

    # ── Header ──
    story.append(Paragraph(report_title, title_style))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", subtitle_style
    ))
    story.append(Spacer(1, 20))

    # ── KPI summary table ──
    kpi_data = [
        ["Total predictions", str(summary["total_predictions"])],
        ["Dominant emotion", summary["dominant_emotion"].capitalize() if summary["dominant_emotion"] else "—"],
        ["Satisfaction score", f"{summary['satisfaction_score']:+.1f} (range -100 to +100)"],
    ]
    kpi_table = Table(kpi_data, colWidths=[2.5 * inch, 3.5 * inch])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#1A1D27")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2A2E3D")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 24))

    # ── Emotion distribution chart ──
    if summary["emotion_distribution"]:
        story.append(Paragraph("Emotion Distribution", styles["Heading2"]))
        chart_buf = _render_bar_chart(summary["emotion_distribution"], "Emotion Distribution (%)")
        story.append(RLImage(chart_buf, width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 16))

    # ── Satisfaction breakdown chart ──
    if summary["satisfaction_distribution"]:
        story.append(Paragraph("Satisfaction Breakdown", styles["Heading2"]))
        chart_buf = _render_bar_chart(summary["satisfaction_distribution"], "Satisfaction Breakdown (%)")
        story.append(RLImage(chart_buf, width=6 * inch, height=3 * inch))
        story.append(Spacer(1, 16))

    # ── Notable moments ──
    if moments:
        story.append(Paragraph("Notable Moments", styles["Heading2"]))
        story.append(Paragraph(
            "High-confidence, non-neutral moments worth reviewing:", styles["Normal"]
        ))
        story.append(Spacer(1, 6))
        moment_rows = [["Time", "Person", "Emotion", "Confidence"]]
        for m in moments[:20]:
            moment_rows.append([
                f"{m['timestamp']:.1f}s", f"Person {m['face_id']}",
                m["emotion"].capitalize(), f"{m['confidence'] * 100:.0f}%",
            ])
        moment_table = Table(moment_rows, colWidths=[1.2 * inch, 1.5 * inch, 1.8 * inch, 1.5 * inch])
        moment_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#6C5CE7")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#2A2E3D")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F2FF")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(moment_table)

    story.append(Spacer(1, 30))
    story.append(Paragraph(
        "Generated by DeepFER — Facial Emotion Recognition & Satisfaction Analytics",
        ParagraphStyle("Footer", parent=styles["Normal"], textColor=colors.grey, fontSize=8),
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
