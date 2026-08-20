import io
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        HRFlowable
    )
    from reportlab.lib.units import inch
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False


def derive_pdf_filename(title_or_query: str, prefix: str = "") -> str:
    """Generates a clean, safe filename from a title or question prompt."""
    clean = re.sub(r'[^\w\s-]', '', title_or_query).strip()
    clean = re.sub(r'[-\s]+', '_', clean)
    if not clean:
        clean = "AI_Study_Notes"
    clean = clean[:45]
    if prefix:
        return f"{prefix}_{clean}.pdf"
    return f"{clean}.pdf"


def _escape_xml(text: str) -> str:
    """Safely escapes XML special characters while preserving basic formatting."""
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    return text


def _convert_inline_markdown(text: str) -> str:
    """Converts bold, italic, line breaks, and inline code markdown into ReportLab XML formatting."""
    # Standardize all variations of <br>, <br/>, <br />
    text = re.sub(r'<\s*br\s*/?\s*>', '___BR_TOKEN___', text, flags=re.IGNORECASE)
    # Strip any stray opening/closing tags that ReportLab doesn't support or unclosed tags
    text = re.sub(r'<\s*/?\s*(?:p|div|span|strong|em|para)\s*>', '', text, flags=re.IGNORECASE)
    # Escape special XML characters
    text = _escape_xml(text)
    # Restore safe self-closing <br/> for ReportLab
    text = text.replace('___BR_TOKEN___', '<br/>')
    # Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    # Italic *text* or _text_
    text = re.sub(r'(?<!\*)\*([^*]+?)\*(?!\*)', r'<i>\1</i>', text)
    # Inline code `code`
    text = re.sub(r'`([^`]+?)`', r'<font face="Courier" color="#4f46e5">\1</font>', text)
    return text


def _markdown_to_flowables(markdown_text: str, styles: Any) -> List[Any]:
    """Parses a markdown string into a list of styled ReportLab Flowables."""
    flowables = []
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=4
    )
    
    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13.5,
        leading=17,
        textColor=colors.HexColor('#0f172a'),
        spaceBefore=8,
        spaceAfter=3
    )
    
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=6,
        spaceAfter=3
    )
    
    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor('#334155'),
        spaceBefore=5,
        spaceAfter=2
    )

    code_style = ParagraphStyle(
        'DocCode',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=3
    )
    
    lines = markdown_text.split('\n')
    i = 0
    in_code_block = False
    code_block_lines = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Handle code blocks ```
        if stripped.startswith('```'):
            if in_code_block:
                if code_block_lines:
                    code_rows = [[Paragraph(_escape_xml(cl) or '&nbsp;', code_style)] for cl in code_block_lines]
                    code_table = Table(code_rows, colWidths=[6.8 * inch])
                    code_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
                        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                        ('TOPPADDING', (0, 0), (-1, -1), 1),
                        ('BOTTOMPADDING', (0, 0), (-1, -1), 1),
                        ('LEFTPADDING', (0, 0), (-1, -1), 8),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                    ]))
                    flowables.append(Spacer(1, 3))
                    flowables.append(code_table)
                    flowables.append(Spacer(1, 4))
                code_block_lines = []
                in_code_block = False
            else:
                in_code_block = True
                code_block_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_block_lines.append(line)
            i += 1
            continue
            
        if not stripped:
            flowables.append(Spacer(1, 3))
            i += 1
            continue
            
        # Table detection (starts and contains |)
        if stripped.startswith('|') and '|' in stripped[1:]:
            table_raw_rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                row_line = lines[i].strip()
                if not re.match(r'^\|[\s\-:|]+\|$', row_line):
                    cells = [c.strip() for c in row_line.strip('|').split('|')]
                    table_raw_rows.append(cells)
                i += 1
                
            if table_raw_rows:
                max_cols = max(len(r) for r in table_raw_rows) if table_raw_rows else 1
                col_width = (6.8 * inch) / max(1, max_cols)
                
                table_flowable_data = []
                for row_idx, row in enumerate(table_raw_rows):
                    row_cells = []
                    for c in row:
                        c_fmt = _convert_inline_markdown(c)
                        font_style = h3_style if row_idx == 0 else body_style
                        row_cells.append(Paragraph(c_fmt, font_style))
                    while len(row_cells) < max_cols:
                        row_cells.append(Paragraph("", body_style))
                    table_flowable_data.append(row_cells)
                    
                table_flowable = Table(table_flowable_data, colWidths=[col_width] * max_cols)
                table_flowable.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                    ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 5),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                ]))
                flowables.append(Spacer(1, 3))
                flowables.append(table_flowable)
                flowables.append(Spacer(1, 4))
            continue
            
        # Headings
        if stripped.startswith('#### '):
            flowables.append(Paragraph(_convert_inline_markdown(stripped[5:]), h3_style))
        elif stripped.startswith('### '):
            flowables.append(Paragraph(_convert_inline_markdown(stripped[4:]), h2_style))
        elif stripped.startswith('## '):
            flowables.append(Paragraph(_convert_inline_markdown(stripped[3:]), h2_style))
        elif stripped.startswith('# '):
            flowables.append(Paragraph(_convert_inline_markdown(stripped[2:]), h1_style))
        elif stripped.startswith('- ') or stripped.startswith('* '):
            bullet_text = _convert_inline_markdown(stripped[2:])
            flowables.append(Paragraph(f"&bull; {bullet_text}", body_style))
        elif re.match(r'^\d+\.\s', stripped):
            num_match = re.match(r'^(\d+\.)\s(.*)', stripped)
            if num_match:
                prefix_num = num_match.group(1)
                item_text = _convert_inline_markdown(num_match.group(2))
                flowables.append(Paragraph(f"<b>{prefix_num}</b> {item_text}", body_style))
            else:
                flowables.append(Paragraph(_convert_inline_markdown(stripped), body_style))
        elif stripped.startswith('> '):
            quote_text = _convert_inline_markdown(stripped[2:])
            quote_table = Table([[Paragraph(quote_text, body_style)]], colWidths=[6.8 * inch])
            quote_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('LINELEFT', (0, 0), (0, -1), 3, colors.HexColor('#6366f1')),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            flowables.append(quote_table)
        elif stripped.startswith('---') or stripped.startswith('***'):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=3, spaceAfter=5))
        else:
            flowables.append(Paragraph(_convert_inline_markdown(stripped), body_style))
            
        i += 1
        
    return flowables


def generate_response_pdf(
    title: str,
    content: str,
    student_name: str = "Student",
    subject: str = "AI Study Assistant"
) -> bytes:
    """
    Generates a clean, professional PDF document from a single AI Tutor response or study topic.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is required for PDF generation. Please install reportlab.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'DocMainTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=6
    )

    meta_label_style = ParagraphStyle(
        'MetaLabel',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9,
        leading=11.5,
        textColor=colors.HexColor('#0f172a')
    )

    meta_val_style = ParagraphStyle(
        'MetaVal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=11.5,
        textColor=colors.HexColor('#334155')
    )

    story = []

    # Header section
    story.append(Paragraph(_escape_xml(title or "AI Study Notes"), title_style))
    story.append(Paragraph(f"Subject: {_escape_xml(subject)} &bull; AI Study Assistant", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceBefore=2, spaceAfter=8))

    # Meta banner
    timestamp_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    meta_data = [
        [
            Paragraph("Prepared For:", meta_label_style),
            Paragraph(_escape_xml(student_name or "Student"), meta_val_style),
            Paragraph("Date / Time:", meta_label_style),
            Paragraph(timestamp_str, meta_val_style)
        ]
    ]
    meta_table = Table(meta_data, colWidths=[1.2 * inch, 2.3 * inch, 1.1 * inch, 2.2 * inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 7),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # Main content flowables
    content_flowables = _markdown_to_flowables(content, styles)
    story.extend(content_flowables)

    # Footer disclaimer
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#cbd5e1'), spaceBefore=4, spaceAfter=5))
    footer_style = ParagraphStyle(
        'DocFooter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#94a3b8'),
        alignment=1
    )
    story.append(Paragraph("AI Study Assistant &bull; Educational Guidance Notes", footer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def generate_chat_pdf(
    session_title: str,
    subject: str,
    student_name: str,
    messages: List[Dict[str, Any]]
) -> bytes:
    """
    Generates a complete chronological PDF transcript of an AI Tutor chat session.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab is required for PDF generation. Please install reportlab.")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'ChatTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=16,
        leading=20,
        textColor=colors.HexColor('#0f172a'),
        spaceAfter=3
    )

    subtitle_style = ParagraphStyle(
        'ChatSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=6
    )

    user_bubble_header = ParagraphStyle(
        'UserHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=2
    )

    assistant_bubble_header = ParagraphStyle(
        'AssistantHeader',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#065f46'),
        spaceAfter=2
    )

    story = []

    # Title & Metadata
    story.append(Paragraph(_escape_xml(session_title or "AI Tutor Conversation"), title_style))
    story.append(Paragraph(f"Subject Focus: {_escape_xml(subject)} &bull; Complete Chat Transcript", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#6366f1'), spaceBefore=2, spaceAfter=8))

    timestamp_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    info_p = Paragraph(f"<b>Student:</b> {_escape_xml(student_name)} &nbsp;&bull;&nbsp; <b>Date:</b> {timestamp_str} &nbsp;&bull;&nbsp; <b>Total Messages:</b> {len(messages)}", subtitle_style)
    story.append(info_p)
    story.append(Spacer(1, 8))

    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "user":
            story.append(Spacer(1, 4))
            story.append(Paragraph(f"👤 <b>{_escape_xml(student_name)}</b>", user_bubble_header))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#bfdbfe'), spaceBefore=2, spaceAfter=4))
            user_flowables = _markdown_to_flowables(content, styles)
            story.extend(user_flowables)
            story.append(Spacer(1, 6))
        else:
            story.append(Spacer(1, 4))
            story.append(Paragraph("🤖 <b>AI Tutor</b>", assistant_bubble_header))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#6ee7b7'), spaceBefore=2, spaceAfter=4))
            assistant_flowables = _markdown_to_flowables(content, styles)
            story.extend(assistant_flowables)
            story.append(Spacer(1, 6))

        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#e2e8f0'), spaceBefore=2, spaceAfter=6))

    # Footer disclaimer
    story.append(Spacer(1, 8))
    footer_style = ParagraphStyle(
        'ChatFooter',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#94a3b8'),
        alignment=1
    )
    story.append(Paragraph("AI Study Assistant &bull; Educational Chat Transcript", footer_style))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
