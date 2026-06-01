# PPTX Design Specification
## CSE 253R Assignment 2 — Music Generation

**Purpose:** This file specifies visual design, layout rules, and `python-pptx`
implementation notes for a builder agent that will create a 20-slide PPTX deck.
Content comes from `slides/SLIDES_CONTENT.md`. Do NOT deviate from the content
or ordering specified there.

---

## 1. Slide Dimensions & Aspect Ratio

| Property | Value |
|----------|-------|
| Width | 13.333 in (33.867 cm) |
| Height | 7.5 in (19.05 cm) |
| Aspect ratio | 16:9 |

```python
from pptx.util import Inches
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
```

---

## 2. Color Palette

High-contrast, simple palette. All colors specified as RGB hex.

| Role | Hex | Usage |
|------|-----|-------|
| Background | `#1A1A2E` | Every slide fill |
| Primary text | `#F0F0F0` | Headings, body text |
| Secondary text | `#B0B0C0` | Captions, footnotes, table row labels |
| Task 1 accent | `#E74C3C` | Red — headings, borders, highlights for Task 1 slides (1–7) |
| Task 4 accent | `#3498DB` | Blue — headings, borders, highlights for Task 4 slides (8–14) |
| Positive / correct | `#2ECC71` | Green cells in results tables |
| Negative / incorrect | `#E74C3C` | Red cells in results tables |
| Neutral | `#95A5A6` | Divider lines, muted labels |
| Table header bg | `#2A2A4A` | Slightly lighter than slide bg |
| Table alt row bg | `#22223A` | Subtle zebra striping |

```python
from pptx.util import Pt
from pptx.dml.color import RGBColor

BG        = RGBColor(0x1A, 0x1A, 0x2E)
TEXT      = RGBColor(0xF0, 0xF0, 0xF0)
TEXT_DIM  = RGBColor(0xB0, 0xB0, 0xC0)
RED       = RGBColor(0xE7, 0x4C, 0x3C)
BLUE      = RGBColor(0x34, 0x98, 0xDB)
GREEN     = RGBColor(0x2E, 0xCC, 0x71)
NEUTRAL   = RGBColor(0x95, 0xA5, 0xA6)
TBL_HDR   = RGBColor(0x2A, 0x2A, 0x4A)
TBL_ALT   = RGBColor(0x22, 0x22, 0x3A)
```

---

## 3. Typography

Use system-safe sans-serif. Calibri is the `python-pptx` default and renders
well on projectors.

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Slide title | Calibri | 36 pt | Bold | `#F0F0F0` |
| Section subtitle | Calibri | 28 pt | Semi-bold | Accent (red or blue) |
| Body text | Calibri | 22 pt | Regular | `#F0F0F0` |
| Bullet text | Calibri | 20 pt | Regular | `#F0F0F0` |
| Table header | Calibri | 18 pt | Bold | `#F0F0F0` |
| Table body | Calibri | 16 pt | Regular | `#F0F0F0` |
| Footer / caption | Calibri | 14 pt | Regular | `#B0B0C0` |
| Code / monospace | Consolas | 16 pt | Regular | `#F0F0F0` |
| Large callout number | Calibri | 48 pt | Bold | Accent |

Rule: never go below 14 pt. Projector readability is the top priority.

---

## 4. Margins & Safe Zone

| Edge | Margin |
|------|--------|
| Left | 0.7 in |
| Right | 0.7 in |
| Top (below title) | 1.5 in |
| Bottom (above footer) | 0.6 in |

Content area: 11.933 in wide × 5.4 in tall (starting at x=0.7, y=1.5).

Footer strip: y=7.0, height=0.4 in, full width. Contains slide number (right),
"CSE 253R" (left), accent-colored thin line above.

---

## 5. Slide Layouts

### 5a. Title Slide (Slide 1 only)

```
+--------------------------------------------------+
|                                                    |
|          [Title — 36pt bold, centered]             |
|          [Subtitle — 24pt, accent color]           |
|                                                    |
|     [Task 1 label — red]  |  [Task 4 label — blue] |
|                                                    |
|     [Team names — 14pt dim]   [Spring 2026]        |
+--------------------------------------------------+
```

- Title vertically centered at ~40% height
- Two accent-colored rounded rectangles side by side for the task labels
- No footer bar on this slide

### 5b. Section Title (Slides 8, 19)

```
+--------------------------------------------------+
|                                                    |
|          [Section Title — 36pt bold]               |
|          [Subtitle — 22pt, accent color]           |
|                                                    |
|          [Simple graphic / flow arrow]             |
|                                                    |
+--------------------------------------------------+
```

- Title centered vertically and horizontally
- Accent-colored horizontal rule (2pt) below subtitle

### 5c. Two-Column (Slides 2, 4, 10, 17)

```
+--------------------------------------------------+
| [Title — top left, 36pt]                           |
+------------------------+--------------------------+
|                        |                           |
|   Left Column          |    Right Column           |
|   (bullet list or      |    (bullet list or        |
|    table or card)      |     table or card)        |
|                        |                           |
+------------------------+--------------------------+
| [Bottom banner — full width, 16pt]                 |
+--------------------------------------------------+
```

- Column widths: 5.7 in each, 0.5 in gutter
- Columns start at y=1.5 in
- Each column may have its own accent-colored header bar (4pt tall rounded rect)
- Bottom banner: full-width rectangle at y=6.2, height=0.7 in, bg `#22223A`

### 5d. Table Slide (Slides 3, 6, 7, 9, 12, 14, 15, 16)

```
+--------------------------------------------------+
| [Title — top left]                                 |
+--------------------------------------------------+
|                                                    |
|   [Table — centered, width ≤ 11 in]               |
|                                                    |
+--------------------------------------------------+
| [Bottom note — 14pt dim]                           |
+--------------------------------------------------+
```

- Table centered horizontally
- Header row: bg `#2A2A4A`, text bold white, accent-colored bottom border (2pt)
- Body rows: alternate `#1A1A2E` and `#22223A`
- Cell padding: 0.08 in all sides
- No visible cell borders except header bottom — clean look
- Max columns: 6. If more, reduce font to 14pt.

### 5e. Image + Text (Slides 3, 7, 12, 14)

```
+--------------------------------------------------+
| [Title — top left]                                 |
+---------------------------+-----------------------+
|                           |                        |
|   Text / table / bullets  |    [Image — right]     |
|   (width ~6.5 in)        |    (width ~4.5 in)     |
|                           |                        |
+---------------------------+-----------------------+
| [Bottom note — 14pt dim]                           |
+--------------------------------------------------+
```

- Image placed right, vertically centered in content area
- Image has 2pt accent-colored border (rounded rect behind it)
- If no image available, leave a placeholder rectangle with dashed border
  and label "See workbook.ipynb" in 16pt dim text

### 5f. Diagram / Flowchart (Slides 5, 11, 13)

```
+--------------------------------------------------+
| [Title — top left]                                 |
+--------------------------------------------------+
|                                                    |
|   [Boxes + arrows — centered, built from shapes]   |
|                                                    |
+--------------------------------------------------+
| [Side annotation — right margin, 16pt dim]         |
+--------------------------------------------------+
```

- Use `python-pptx` shapes: rounded rectangles + connectors
- Box fill: accent color at 20% opacity (use solid lighter shade instead,
  since pptx transparency is tricky)
- Box border: accent color, 2pt
- Arrow connectors: white, 2pt, with arrowhead
- Labels inside boxes: 16pt bold, white

### 5g. Demo / List Slide (Slide 20)

```
+--------------------------------------------------+
| [Title — top left]                                 |
+--------------------------------------------------+
|                                                    |
|   Task 1 — Symbolic        Task 4 — Continuous    |
|   1. Real Bach ...          4. Pretrained ...      |
|   2. Markov ...             5. Fine-tuned ...      |
|   3. LSTM ...               6. ...                 |
|                                                    |
+--------------------------------------------------+
```

- Two-column numbered list, large text (20pt)
- Each task header in its accent color
- Simple layout, no complex visuals

---

## 6. Image Embedding Plan

These images must be generated from the notebook **before** building the PPTX.
None currently exist in the repo — the builder should run the relevant notebook
cells or export them first.

| Image File | Slide | Placement | Size | Notes |
|-----------|-------|-----------|------|-------|
| `training_curves.png` | 7 (Task 1 Results) | Right column, image+text layout | 4.5 in wide | LSTM train/val loss curves. Red accent border. |
| `eval_summary.png` | 7 (Task 1 Results) | Right column below training_curves, OR as a standalone if space is tight | 4.5 in wide | 5-metric comparison bar chart. Red accent border. |
| `eval_task4_genre_accuracy.png` | 14 (Task 4 Results) | Right column or below results table | 4.5 in wide | Pretrained vs fine-tuned accuracy bar chart. Blue accent border. |

Optional images (embed if available, skip if not):

| Image File | Slide | Notes |
|-----------|-------|-------|
| `eda_piano_roll.png` | 3 | Piano roll visualization, right column |
| `eda_fma_overview.png` | 9 | FMA genre distribution, right column |
| `eda_fma_comparison.png` | 10 | Fine-tune vs scratch comparison visual |
| `eda_musicgen_arch.png` | 11 | MusicGen architecture diagram |

Image insertion pattern:

```python
from pptx.util import Inches
slide.shapes.add_picture(
    "training_curves.png",
    left=Inches(8.0),
    top=Inches(1.8),
    width=Inches(4.5),
)
```

---

## 7. Accent Color Assignment by Slide

| Slides | Accent | Hex |
|--------|--------|-----|
| 1 (title) | Both (split) | Red left, Blue right |
| 2 (overview) | Both (split columns) | Red left, Blue right |
| 3–7 | Task 1 | `#E74C3C` red |
| 8–14 | Task 4 | `#3498DB` blue |
| 15 | Task 1 related work | `#E74C3C` red |
| 16 | Task 4 related work | `#3498DB` blue |
| 17 (comparison) | Both (split columns) | Red left, Blue right |
| 18 (conclusion) | Both (two cards) | Red card, Blue card |
| 19 (thank you) | Neutral | `#95A5A6` |
| 20 (demo) | Both (split columns) | Red left, Blue right |

---

## 8. python-pptx Implementation Notes

### 8a. Slide Background

Apply dark background to every slide. Do not use master/layout backgrounds —
set each slide individually for reliability.

```python
from pptx.oxml.ns import qn
from lxml import etree

def set_slide_bg(slide, color=BG):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color
```

### 8b. Blank Layout

Use a blank slide layout (index 6 in most templates) and build all content
from shapes. This avoids placeholder conflicts.

```python
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
```

### 8c. Adding Text Boxes

```python
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

txBox = slide.shapes.add_textbox(
    Inches(0.7), Inches(0.4), Inches(11.9), Inches(0.9)
)
tf = txBox.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Slide Title Here"
p.font.size = Pt(36)
p.font.bold = True
p.font.color.rgb = TEXT
p.alignment = PP_ALIGN.LEFT
```

### 8d. Building Tables

```python
rows, cols = 6, 4
table_shape = slide.shapes.add_table(
    rows, cols,
    Inches(0.7), Inches(1.5), Inches(11.0), Inches(4.0)
)
table = table_shape.table

for col_idx in range(cols):
    cell = table.cell(0, col_idx)
    cell.text = headers[col_idx]
    # Style header
    for paragraph in cell.text_frame.paragraphs:
        paragraph.font.size = Pt(18)
        paragraph.font.bold = True
        paragraph.font.color.rgb = TEXT
    # Header background
    cell.fill.solid()
    cell.fill.fore_color.rgb = TBL_HDR

# Remove default borders for clean look
from pptx.oxml.ns import qn
for cell in table.iter_cells():
    tcPr = cell._tc.get_or_add_tcPr()
    for border_name in ['a:lnL', 'a:lnR', 'a:lnT', 'a:lnB']:
        ln = tcPr.find(qn(border_name))
        if ln is not None:
            tcPr.remove(ln)
        ln = etree.SubElement(tcPr, qn(border_name))
        ln.set('w', '0')
        noFill = etree.SubElement(ln, qn('a:noFill'))
```

### 8e. Shape-Based Diagrams (Slides 5, 11, 13)

Use rounded rectangles for boxes and connectors for arrows.

```python
from pptx.enum.shapes import MSO_SHAPE

box = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE,
    Inches(x), Inches(y), Inches(w), Inches(h)
)
box.fill.solid()
box.fill.fore_color.rgb = RGBColor(0x2A, 0x3A, 0x5E)  # Muted accent
box.line.color.rgb = BLUE
box.line.width = Pt(2)

tf = box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "T5 Text Encoder"
p.font.size = Pt(16)
p.font.bold = True
p.font.color.rgb = TEXT
p.alignment = PP_ALIGN.CENTER
```

For arrows between boxes, use straight connectors:

```python
connector = slide.shapes.add_connector(
    MSO_CONNECTOR_TYPE.STRAIGHT,
    Inches(x1), Inches(y1), Inches(x2), Inches(y2)
)
connector.line.color.rgb = TEXT
connector.line.width = Pt(2)
```

### 8f. Conditional / Accent Result Cells

For the results tables (slides 7, 14) that need green/red cell coloring:

```python
def color_cell(table, row, col, bg_color):
    cell = table.cell(row, col)
    cell.fill.solid()
    cell.fill.fore_color.rgb = bg_color
```

### 8g. Footer Strip

Add to every slide except slide 1:

```python
def add_footer(slide, slide_num, accent_color=NEUTRAL):
    # Thin accent line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.7), Inches(6.95), Inches(11.9), Inches(0.03)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = accent_color
    line.line.fill.background()

    # Left label
    left = slide.shapes.add_textbox(
        Inches(0.7), Inches(7.0), Inches(3.0), Inches(0.4)
    )
    p = left.text_frame.paragraphs[0]
    p.text = "CSE 253R — Assignment 2"
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DIM

    # Right slide number
    right = slide.shapes.add_textbox(
        Inches(11.5), Inches(7.0), Inches(1.2), Inches(0.4)
    )
    p = right.text_frame.paragraphs[0]
    p.text = str(slide_num)
    p.font.size = Pt(12)
    p.font.color.rgb = TEXT_DIM
    p.alignment = PP_ALIGN.RIGHT
```

### 8h. Image Fallback

If an image file does not exist at build time, insert a placeholder:

```python
import os

def add_image_or_placeholder(slide, img_path, left, top, width, accent):
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, left, top, width=width)
    else:
        box = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, left, top, width, Inches(3.0)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0x22, 0x22, 0x3A)
        box.line.color.rgb = accent
        box.line.width = Pt(1)
        box.line.dash_style = MSO_LINE_DASH_STYLE.DASH
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"[{os.path.basename(img_path)}]\nSee workbook.ipynb"
        p.font.size = Pt(14)
        p.font.color.rgb = TEXT_DIM
        p.alignment = PP_ALIGN.CENTER
```

---

## 9. General Style Rules

1. **No clutter.** Maximum 6 bullet points per column. Maximum 7 table rows
   visible at once. If more content exists, split across slides or summarize.
2. **No animations.** PPTX should have no entrance/exit animations. Static
   slides only.
3. **Consistent alignment.** All titles start at (0.7 in, 0.4 in). All content
   starts at y=1.5 in. All footers at y=7.0 in.
4. **Bold key numbers.** In results tables, make the winning/best metric value
   bold. Use green cell fill for correct predictions, red for incorrect.
5. **Whitespace.** Leave at least 0.3 in between text blocks. Do not fill
   every pixel — breathing room improves readability.
6. **Image borders.** All embedded images get a 2pt accent-colored border
   (place a slightly larger rounded rect behind the image).
7. **No gradients.** Solid fills only. Gradients render inconsistently across
   PowerPoint versions.
8. **Slide numbers.** Bottom-right, 12pt, dim gray. Start numbering at 1.
