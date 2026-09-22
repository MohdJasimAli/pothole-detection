"""
Build the shareable code documents:
    docs/Pothole_Detection_Main_Code.docx   (Word - editable for team review)
    docs/Pothole_Detection_Main_Code.pdf    (PDF  - fixed formatting)
"""
import datetime
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(ROOT, "docs")
os.makedirs(DOCS, exist_ok=True)

FILES = [
    ("config.py",
     "Configuration: paths, thresholds, size/depth/severity levels, API vars"),
    ("main.py",
     "MAIN PIPELINE - detect -> size -> depth -> severity -> report"),
    ("src/detector.py",
     "YOLOv8 detector + DetectionResult data structure (primary backend)"),
    ("src/classical_detector.py",
     "Classical CV detector: Otsu+adaptive threshold, morphology, contours"),
    ("src/preprocessing.py",
     "Image preprocessing: CLAHE contrast, denoise, lighting/shadow removal"),
    ("src/depth_estimator.py",
     "Depth estimation: MiDaS DPT + synthetic fallback, depth categories"),
    ("src/size_estimator.py",
     "Pixel -> cm size estimation (camera model) + Small/Medium/Large"),
    ("src/severity.py",
     "Severity scoring (0-100) + matrix classification + priority levels"),
    ("src/utils.py",
     "Helpers: logging, image IO, drawing boxes, IoU, report generation"),
    ("api/app.py",
     "Flask REST API: /api/analyze, /api/detect, /api/reports, /dashboard"),
    ("train.py",
     "YOLOv8 training script (dataset config, train, validate, export)"),
    ("tests/test_detector.py",
     "Unit tests - 25 tests covering config, preprocessing, size, severity"),
]

OVERVIEW = """
The system detects potholes in road images and estimates their size, depth
and severity to support smart road maintenance.

Pipeline (5 steps, implemented in main.py -> PotholeAnalysisPipeline):
  1. DETECT  - YOLOv8 object detection (or the classical CV fallback when
               PyTorch is unavailable). Output: bounding boxes + confidence.
  2. SIZE    - Convert pixel box to real-world cm using a calibrated baseline
               or a reference object; classify Small / Medium / Large.
  3. DEPTH   - MiDaS monocular depth map sampled inside each box; classify
               Shallow (<2cm) / Moderate (2-5cm) / Deep (>5cm).
  4. SEVERITY- Weighted score 0-100 from size + depth -> Critical / High /
               Medium / Low with a maintenance response deadline.
  5. REPORT  - Annotated image (severity-coloured boxes) + JSON report with
               per-pothole metrics, summary counts and backend info.

Severity -> response time:
  Critical (>=80): 24 hours | High (60-79): 1 week
  Medium (35-59): 1 month  | Low (<35): 3 months

Verified performance:
  - 24 tests passed, 1 skipped (integration test requires torch)
  - POST /api/analyze on a 3-pothole image: HTTP 200 in ~0.85s local,
    ~3.7s on the free cloud instance; all 3 potholes detected with
    independent size/depth/severity.
"""

RUN = """
Run locally:
    pip install -r requirements.txt
    pytest tests/ -q
    python main.py --image data/samples/multi_potholes.jpg
    python main.py --web          # then open http://localhost:5000/dashboard

Deploy (already done):
    Live dashboard: https://pothole-detection-api-h1cp.onrender.com/dashboard
    Render auto-deploys on every push to main (build runs the test suite).

Train a custom YOLOv8 model:
    pip install torch ultralytics
    python train.py --create-config
    python train.py --data dataset.yaml --epochs 100 --model yolov8s.pt
"""


def read(rel):
    path = os.path.join(ROOT, rel)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        return f.read()


REPL = {"\u2192": "->", "\u2265": ">=", "\u2264": "<=", "\u2248": "~",
        "\u2014": "-", "\u2013": "-", "\u2022": "*", "\u2713": "v",
        "\u00d7": "x", "\u00b2": "2", "\u2500": "-", "\u2502": "|",
        "\u2514": "+", "\u251c": "+", "\u2190": "<-", "\u2026": "..."}




def ascii_safe(s):
    for k, v in REPL.items():
        s = s.replace(k, v)
    return s.encode("ascii", "replace").decode("ascii")


def build_docx():
    """Word document: editable by the team (comments/track changes)."""
    try:
        from docx import Document
        from docx.shared import Pt
    except ImportError:
        print("python-docx not available - skipping .docx")
        return None
    doc = Document()
    doc.add_heading("Road Pothole Detection & Severity Analysis", 0)
    sub = doc.add_paragraph("Main Code - for project team review")
    sub.runs[0].italic = True
    doc.add_paragraph("Generated: %s" % datetime.date.today().strftime("%d %b %Y"))
    doc.add_paragraph(
        "Live demo: https://pothole-detection-api-h1cp.onrender.com/dashboard")
    doc.add_paragraph(
        "Repository: https://github.com/MohdJasimAli/pothole-detection")
    doc.add_heading("Overview", 1)
    doc.add_paragraph(OVERVIEW.strip())
    doc.add_heading("How to Run", 1)
    doc.add_paragraph(RUN.strip())
    doc.add_heading("File Index", 1)
    for path, desc in FILES:
        par = doc.add_paragraph(style="List Bullet")
        run = par.add_run(path)
        run.bold = True
        par.add_run(" - " + desc)
    for path, desc in FILES:
        doc.add_page_break()
        doc.add_heading(path, 1)
        ip = doc.add_paragraph(desc)
        ip.runs[0].italic = True
        for line in read(path).splitlines():
            cp = doc.add_paragraph()
            pf = cp.paragraph_format
            pf.space_before = Pt(0)
            pf.space_after = Pt(0)
            r = cp.add_run(line if line.strip() else " ")
            r.font.name = "Consolas"
            r.font.size = Pt(8)
    out = os.path.join(DOCS, "Pothole_Detection_Main_Code.docx")
    doc.save(out)
    print("wrote", out, "|", os.path.getsize(out) // 1024, "KB")
    return out


def build_pdf():
    """PDF document: fixed formatting, one file per section."""
    try:
        import pymupdf as fitz
    except ImportError:
        try:
            import fitz
        except ImportError:
            print("pymupdf not available - skipping .pdf")
            return None
    doc = fitz.open()
    W, H, M, LH, WRAP = 595, 842, 40, 9.3, 112
    state = {"page": doc.new_page(width=W, height=H), "y": M}

    def put(text, size=7.4, bold=False):
        text = ascii_safe(text)
        while True:
            chunk, text = text[:WRAP], text[WRAP:]
            if state["y"] > H - M:
                state["page"] = doc.new_page(width=W, height=H)
                state["y"] = M
            if chunk.strip():
                state["page"].insert_text(
                    (M, state["y"]), chunk,
                    fontname="cobo" if bold else "cour", fontsize=size)
            state["y"] += LH
            if not text:
                break

    def newpage():
        state["page"] = doc.new_page(width=W, height=H)
        state["y"] = M

    put("Road Pothole Detection & Severity Analysis", size=15, bold=True)
    state["y"] += LH
    put("Main Code - for project team review", size=10, bold=True)
    put("Generated: %s" % datetime.date.today().strftime("%d %b %Y"))
    put("Live demo: https://pothole-detection-api-h1cp.onrender.com/dashboard")
    put("Repository: https://github.com/MohdJasimAli/pothole-detection")
    state["y"] += LH
    put("OVERVIEW", size=9, bold=True)
    for ln in OVERVIEW.strip().splitlines():
        put(ln)
    state["y"] += LH
    put("HOW TO RUN", size=9, bold=True)
    for ln in RUN.strip().splitlines():
        put(ln)
    state["y"] += LH
    put("FILE INDEX", size=9, bold=True)
    for path, desc in FILES:
        put("  %s - %s" % (path, desc))
    for path, desc in FILES:
        newpage()
        put("=" * 100, bold=True)
        put("%s  |  %s" % (path, desc), size=9, bold=True)
        put("=" * 100, bold=True)
        for line in read(path).splitlines():
            put(line if line.strip() else "")
    out = os.path.join(DOCS, "Pothole_Detection_Main_Code.pdf")
    doc.save(out)
    print("wrote", out, "|", os.path.getsize(out) // 1024, "KB",
          "|", doc.page_count, "pages")
    return out


if __name__ == "__main__":
    missing = [p for p, _ in FILES
               if not os.path.exists(os.path.join(ROOT, p))]
    if missing:
        print("WARNING missing files:", missing)
    total = sum(len(read(p).splitlines()) for p, _ in FILES)
    print("source files:", len(FILES), "| total lines:", total)
    build_docx()
    build_pdf()