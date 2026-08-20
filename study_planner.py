import io
from typing import List, Dict, Any, Optional
from groq import Groq

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

from config import LLM_MODEL, get_groq_api_key


def get_groq_client() -> Groq:
    """Returns a Groq API client configured with the API key."""
    api_key = get_groq_api_key()
    return Groq(api_key=api_key or "PASTE_YOUR_GROQ_API_KEY_HERE")




def format_duration(hours: int, minutes: int) -> str:
    """Formats hours and minutes into a clean human-readable duration string."""
    parts = []
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts:
        return "0 minutes"
    return " ".join(parts)


def generate_study_plan(
    topic: str,
    target_level: str,
    day_schedules: List[Dict[str, Any]]
) -> str:
    """
    Generates a personalized study plan based strictly on the user's manual topic,
    target level, and manually selected days with exact hour & minute allocations.
    """
    schedule_summary_lines = []
    for item in day_schedules:
        day_name = item["day"]
        dur_str = format_duration(item.get("hours", 0), item.get("minutes", 0))
        total_mins = item.get("total_minutes", (item.get("hours", 0) * 60) + item.get("minutes", 0))
        schedule_summary_lines.append(f"- {day_name}: {dur_str} ({total_mins} total minutes)")

    schedule_text = "\n".join(schedule_summary_lines)

    level_guidance = ""
    if target_level == "Beginner":
        level_guidance = "Focus on foundational definitions, core principles, simple analogies, and fundamental practice."
    elif target_level == "Advanced":
        level_guidance = "Focus on complex edge-cases, system architecture, performance optimization, deep problem-solving, and advanced analysis."
    else:
        level_guidance = "Focus on conceptual clarity, standard textbook problems, practical scenarios, and exam-level question practice."

    prompt = f"""
Create a highly structured, realistic, and personalized college study plan.

TOPIC:
{topic}

TARGET LEVEL:
{target_level} ({level_guidance})

STUDY SCHEDULE (Days and Allocated Time):
{schedule_text}

CRITICAL RULES:
1. Create study sessions ONLY for the specific days listed in the schedule above. Do NOT add unselected days (e.g. if only Monday and Wednesday are listed, output sessions for Monday and Wednesday ONLY).
2. The study tasks for each day MUST strictly fit the allocated duration for that specific day.
3. For each scheduled day, include:
   - Day Header (e.g., Day 1 — Monday | Duration: 2 hours 30 minutes)
   - Specific Subtopic Focus
   - Core Study Tasks (concepts to learn)
   - Practice / Hands-on Activity (exercises, problem solving)
   - Quick Self-Check / Revision Question
4. Keep the output clean, highly organized, and structured in Markdown format.
"""

    client = get_groq_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content or ""



def generate_study_plan_pdf(
    student_name: str,
    topic: str,
    target_level: str,
    day_schedules: List[Dict[str, Any]],
    plan_content: str
) -> bytes:
    """
    Generates a clean, professional PDF document of the personalized study plan using ReportLab.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is required to generate study plan PDFs. Please install it using 'pip install reportlab'.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#0f172a'),
        alignment=0,
        spaceAfter=4
    )

    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#475569'),
        spaceAfter=12
    )

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#0f172a')
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("AI STUDY ASSISTANT", title_style))
    story.append(Paragraph("Personalized Study Plan & Revision Schedule", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceBefore=2, spaceAfter=14))

    # Meta Overview Table
    meta_data = [
        [
            Paragraph("Student Name:", meta_label_style),
            Paragraph(student_name or "Student", meta_val_style),
            Paragraph("Target Level:", meta_label_style),
            Paragraph(target_level, meta_val_style)
        ],
        [
            Paragraph("Study Topic:", meta_label_style),
            Paragraph(topic, meta_val_style),
            Paragraph("Active Study Days:", meta_label_style),
            Paragraph(f"{len(day_schedules)} days selected", meta_val_style)
        ]
    ]

    meta_table = Table(meta_data, colWidths=[1.3 * inch, 2.3 * inch, 1.3 * inch, 2.1 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 12))

    # Schedule Allocation Summary
    story.append(Paragraph("Weekly Time Allocations", section_header_style))
    schedule_rows = [["Day of Week", "Allocated Duration", "Total Minutes"]]
    for s in day_schedules:
        dur = format_duration(s.get("hours", 0), s.get("minutes", 0))
        mins = s.get("total_minutes", (s.get("hours", 0) * 60) + s.get("minutes", 0))
        schedule_rows.append([s["day"], dur, f"{mins} mins"])

    sched_table = Table(schedule_rows, colWidths=[2.2 * inch, 2.8 * inch, 2.0 * inch])
    sched_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0f172a')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
    ]))
    story.append(sched_table)
    story.append(Spacer(1, 14))

    # Detailed Generated Plan Section
    story.append(Paragraph("Structured Daily Study Schedule", section_header_style))
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#cbd5e1'), spaceBefore=2, spaceAfter=8))

    # Parse and format the markdown plan lines into robust PDF flowables (tables, lists, headers, clean XML)
    try:
        from pdf_generator import _markdown_to_flowables
        plan_flowables = _markdown_to_flowables(plan_content, styles)
        story.extend(plan_flowables)
    except Exception:
        # Fallback to safe plain text paragraphs if custom parser encounters any unexpected syntax
        for line in plan_content.split("\n"):
            line_clean = line.strip()
            if not line_clean:
                story.append(Spacer(1, 4))
            else:
                from pdf_generator import _escape_xml
                story.append(Paragraph(_escape_xml(line_clean), body_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes