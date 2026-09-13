"""
Realistic Document Generator for Smart Image & Document Detector.

Generates high-resolution, legally unencumbered, realistic sample documents
representing diverse real-world categories:
1. Retail thermal receipts
2. Commercial invoices with tables
3. Official tax/government forms with check-boxes and seals
4. Certificates with decorative borders and ribbons
5. Identity card specimens with photo ID boxes
6. Photographed documents with natural perspective and shadows
7. Tinted/colored paper documents
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_barcode(draw, bbox):
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    np.random.seed(42)
    bars = np.random.choice([1, 2, 3, 4], size=30)
    cur_x = x0
    for b in bars:
        if cur_x + b > x1:
            break
        draw.rectangle([cur_x, y0, cur_x + b, y1], fill=(0, 0, 0))
        cur_x += b + np.random.choice([1, 2, 3])


def create_retail_receipt(output_path: Path):
    """Creates a narrow thermal paper receipt."""
    img = Image.new("RGB", (360, 720), color=(250, 250, 248))
    d = ImageDraw.Draw(img)

    # Header
    d.rectangle([30, 25, 330, 28], fill=(50, 50, 50))
    d.rectangle([100, 40, 260, 55], fill=(30, 30, 30))  # Store Name block
    d.rectangle([70, 65, 290, 73], fill=(120, 120, 120))
    d.rectangle([110, 80, 250, 88], fill=(140, 140, 140))

    # Divider line
    d.line([(30, 105), (330, 105)], fill=(180, 180, 180), width=1)

    # Itemized lines
    y = 125
    for i in range(12):
        item_w = np.random.randint(90, 170)
        d.rectangle([35, y, 35 + item_w, y + 10], fill=(50, 50, 50))
        d.rectangle([280, y, 325, y + 10], fill=(30, 30, 30))  # Price
        y += 24

    # Subtotal and total
    d.line([(30, y + 10), (330, y + 10)], fill=(100, 100, 100), width=2)
    y += 25
    d.rectangle([35, y, 100, y + 12], fill=(40, 40, 40))
    d.rectangle([260, y, 325, y + 12], fill=(20, 20, 20))
    y += 25
    d.rectangle([35, y, 120, y + 16], fill=(0, 0, 0))
    d.rectangle([240, y, 325, y + 16], fill=(0, 0, 0))

    # Barcode
    y += 50
    draw_barcode(d, [60, y, 300, y + 55])
    d.rectangle([90, y + 65, 270, y + 75], fill=(100, 100, 100))

    img.save(output_path, "JPEG", quality=95)


def create_commercial_invoice(output_path: Path):
    """Creates a full-page business invoice with tables."""
    img = Image.new("RGB", (700, 950), color=(255, 255, 255))
    d = ImageDraw.Draw(img)

    # Top accent banner
    d.rectangle([0, 0, 700, 18], fill=(37, 99, 235))  # Blue top bar

    # Company Logo Block & Invoice Title
    d.rectangle([50, 45, 180, 85], fill=(30, 41, 59))
    d.rectangle([480, 45, 650, 85], fill=(15, 23, 42))

    # Billing Blocks
    d.rectangle([50, 120, 250, 132], fill=(71, 85, 105))
    d.rectangle([50, 140, 220, 190], outline=(203, 213, 225), width=1)
    for y in range(150, 185, 12):
        d.rectangle([60, y, 200, y + 6], fill=(100, 116, 139))

    d.rectangle([450, 120, 650, 132], fill=(71, 85, 105))
    d.rectangle([450, 140, 650, 190], outline=(203, 213, 225), width=1)
    for y in range(150, 185, 12):
        d.rectangle([460, y, 630, y + 6], fill=(100, 116, 139))

    # Table Header
    d.rectangle([50, 230, 650, 260], fill=(241, 245, 249))
    d.line([(50, 230), (650, 230)], fill=(203, 213, 225), width=1)
    d.line([(50, 260), (650, 260)], fill=(203, 213, 225), width=1)

    # Table rows
    table_y = 270
    for r in range(10):
        d.line([(50, table_y + 25), (650, table_y + 25)], fill=(226, 232, 240), width=1)
        d.rectangle([60, table_y + 6, 220, table_y + 14], fill=(70, 70, 70))
        d.rectangle([300, table_y + 6, 360, table_y + 14], fill=(100, 100, 100))
        d.rectangle([440, table_y + 6, 500, table_y + 14], fill=(100, 100, 100))
        d.rectangle([580, table_y + 6, 640, table_y + 14], fill=(40, 40, 40))
        table_y += 35

    # Totals Box
    d.rectangle([450, table_y + 20, 650, table_y + 110], fill=(248, 250, 252), outline=(203, 213, 225))
    d.rectangle([465, table_y + 35, 530, table_y + 45], fill=(100, 100, 100))
    d.rectangle([570, table_y + 35, 635, table_y + 45], fill=(30, 30, 30))
    d.rectangle([465, table_y + 75, 540, table_y + 90], fill=(15, 23, 42))
    d.rectangle([560, table_y + 75, 635, table_y + 90], fill=(15, 23, 42))

    # Bottom Stamp
    d.ellipse([100, table_y + 30, 220, table_y + 100], outline=(37, 99, 235), width=3)
    d.rectangle([120, table_y + 55, 200, table_y + 70], fill=(37, 99, 235))

    img.save(output_path, "JPEG", quality=95)


def create_official_form(output_path: Path):
    """Creates a dual-column government/tax form with checkboxes and seal."""
    img = Image.new("RGB", (720, 960), color=(252, 252, 250))
    d = ImageDraw.Draw(img)

    # Coat of arms / Emblem placeholder
    d.ellipse([325, 35, 395, 105], fill=(220, 38, 38), outline=(153, 27, 27), width=2)

    # Title Block
    d.rectangle([180, 120, 540, 138], fill=(17, 24, 39))
    d.rectangle([240, 148, 480, 158], fill=(75, 85, 99))

    # Form Grids
    y = 190
    for section in range(4):
        # Section header
        d.rectangle([50, y, 670, y + 22], fill=(229, 231, 235))
        d.rectangle([65, y + 5, 220, y + 17], fill=(31, 41, 55))
        y += 35

        # Fields
        for row in range(3):
            # Label
            d.rectangle([55, y, 180, y + 8], fill=(107, 114, 128))
            d.rectangle([385, y, 510, y + 8], fill=(107, 114, 128))
            # Box
            d.rectangle([55, y + 12, 335, y + 36], outline=(156, 163, 175), fill=(255, 255, 255))
            d.rectangle([385, y + 12, 665, y + 36], outline=(156, 163, 175), fill=(255, 255, 255))
            # Simulated text entry
            d.line([(65, y + 26), (190, y + 26)], fill=(30, 41, 59), width=2)
            y += 48
        y += 10

    # Signature Block & Red Circular Stamp
    d.line([(80, 880), (300, 880)], fill=(75, 85, 99), width=1)
    d.rectangle([110, 890, 270, 900], fill=(107, 114, 128))

    d.ellipse([500, 810, 620, 930], outline=(220, 38, 38), width=3)
    d.ellipse([515, 825, 605, 915], outline=(220, 38, 38), width=1)
    d.rectangle([530, 860, 590, 880], fill=(220, 38, 38))

    img.save(output_path, "JPEG", quality=95)


def create_certificate(output_path: Path):
    """Creates an ornate certificate with gold/blue decorative border and seals."""
    img = Image.new("RGB", (900, 640), color=(255, 254, 250))
    d = ImageDraw.Draw(img)

    # Ornate double border
    d.rectangle([25, 25, 875, 615], outline=(180, 83, 9), width=4)
    d.rectangle([35, 35, 865, 605], outline=(217, 119, 6), width=1)
    d.rectangle([45, 45, 855, 595], outline=(180, 83, 9), width=2)

    # Corner ornaments
    for cx, cy in [(45, 45), (855, 45), (45, 595), (855, 595)]:
        d.ellipse([cx - 15, cy - 15, cx + 15, cy + 15], fill=(217, 119, 6))

    # Certificate Title
    d.rectangle([280, 90, 620, 130], fill=(17, 24, 39))
    d.rectangle([340, 145, 560, 158], fill=(180, 83, 9))

    # Recipient Name Line
    d.line([(200, 260), (700, 260)], fill=(180, 83, 9), width=2)
    d.rectangle([260, 215, 640, 248], fill=(30, 41, 59))

    # Body text lines
    for y in [300, 325, 350, 375]:
        d.rectangle([180, y, 720, y + 8], fill=(100, 116, 139))

    # Signatures
    d.line([(120, 500), (320, 500)], fill=(71, 85, 105), width=1)
    d.line([(580, 500), (780, 500)], fill=(71, 85, 105), width=1)

    # Gold Ribbon Seal
    seal_cx, seal_cy = 450, 480
    d.polygon([(seal_cx - 20, seal_cy + 30), (seal_cx - 40, seal_cy + 90), (seal_cx, seal_cy + 65)], fill=(180, 83, 9))
    d.polygon([(seal_cx + 20, seal_cy + 30), (seal_cx + 40, seal_cy + 90), (seal_cx, seal_cy + 65)], fill=(180, 83, 9))
    d.ellipse([seal_cx - 45, seal_cy - 45, seal_cx + 45, seal_cy + 45], fill=(245, 158, 11), outline=(180, 83, 9), width=3)
    d.ellipse([seal_cx - 32, seal_cy - 32, seal_cx + 32, seal_cy + 32], outline=(255, 255, 255), width=2)

    img.save(output_path, "JPEG", quality=95)


def create_identity_card(output_path: Path):
    """Creates an ID card specimen with embedded photo portrait box and chip."""
    # Standard ID-1 aspect ratio: 85.6mm x 53.98mm (~680 x 430 px)
    img = Image.new("RGB", (680, 430), color=(240, 246, 255))
    d = ImageDraw.Draw(img)

    # Security guilloche wavy background lines
    for i in range(0, 680, 15):
        d.line([(i, 0), (i + 80, 430)], fill=(220, 234, 254), width=1)

    # Top Header Band
    d.rectangle([0, 0, 680, 65], fill=(30, 58, 138))
    d.rectangle([35, 18, 220, 36], fill=(255, 255, 255))
    d.rectangle([35, 42, 160, 50], fill=(191, 219, 254))

    # Smart Card Chip
    d.rectangle([45, 105, 105, 155], fill=(234, 179, 8), outline=(202, 138, 4), width=2)
    d.line([(45, 130), (105, 130)], fill=(202, 138, 4), width=1)
    d.line([(75, 105), (75, 155)], fill=(202, 138, 4), width=1)

    # Photo Box with Portrait Silhouette (Simulated Face ID)
    photo_box = [45, 180, 205, 370]
    d.rectangle(photo_box, fill=(226, 232, 240), outline=(148, 163, 184), width=2)
    # Head & Shoulders silhouette
    d.ellipse([95, 210, 155, 275], fill=(100, 116, 139))
    d.ellipse([65, 290, 185, 400], fill=(71, 85, 105))

    # Cardholder Information Fields
    info_x = 240
    d.rectangle([info_x, 90, 420, 105], fill=(30, 41, 59))  # Full Name
    d.rectangle([info_x, 115, 340, 125], fill=(100, 116, 139))

    # ID Number & Details
    labels_y = [150, 195, 240, 285]
    for ly in labels_y:
        d.rectangle([info_x, ly, info_x + 90, ly + 8], fill=(148, 163, 184))
        d.rectangle([info_x, ly + 14, info_x + 220, ly + 28], fill=(15, 23, 42))

    # Hologram badge / QR Code
    d.rectangle([540, 180, 635, 275], fill=(255, 255, 255), outline=(203, 213, 225), width=1)
    draw_barcode(d, [550, 190, 625, 265])

    # Bottom Magnetic / MRZ Zone
    d.rectangle([0, 390, 680, 430], fill=(226, 232, 240))
    for dot_x in range(30, 650, 18):
        d.rectangle([dot_x, 405, dot_x + 10, 415], fill=(51, 65, 85))

    img.save(output_path, "JPEG", quality=95)


def create_photographed_document(output_path: Path):
    """Creates a paper document sitting on a desk with lighting gradients & shadow."""
    # Background: Wood desk
    bg = Image.new("RGB", (800, 1000), color=(120, 75, 45))
    d_bg = ImageDraw.Draw(bg)
    # Wood grain stripes
    for y in range(0, 1000, 25):
        d_bg.line([(0, y), (800, y + 10)], fill=(105, 65, 38), width=3)

    # Document Page
    doc = Image.new("RGB", (600, 800), color=(252, 252, 250))
    d_doc = ImageDraw.Draw(doc)
    d_doc.rectangle([50, 40, 350, 65], fill=(30, 41, 59))
    d_doc.line([(50, 85), (550, 85)], fill=(203, 213, 225), width=2)
    for y in range(110, 720, 22):
        line_w = np.random.randint(280, 480)
        d_doc.rectangle([50, y, 50 + line_w, y + 8], fill=(70, 70, 70))

    # Drop shadow
    shadow = Image.new("RGBA", (630, 830), color=(0, 0, 0, 80))
    bg.paste(shadow, (90, 110), mask=shadow)

    # Paste document
    bg.paste(doc, (85, 100))

    # Add lighting gradient across image (simulating lamp on left, shadow on right)
    arr = np.array(bg, dtype=np.float32)
    gradient = np.linspace(1.1, 0.75, 800).reshape(1, 800, 1)
    arr = np.clip(arr * gradient, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    img.save(output_path, "JPEG", quality=95)


def create_colored_paper_document(output_path: Path):
    """Creates a legal notice on pastel blue/yellow paper with dark text."""
    img = Image.new("RGB", (650, 880), color=(219, 234, 254))  # Pastel blue paper
    d = ImageDraw.Draw(img)

    d.rectangle([40, 40, 610, 840], outline=(147, 197, 253), width=2)
    d.rectangle([70, 60, 320, 85], fill=(30, 58, 138))  # Dark Blue Title
    d.line([(70, 105), (580, 105)], fill=(147, 197, 253), width=1)

    for y in range(130, 750, 20):
        w = np.random.randint(300, 490)
        d.rectangle([70, y, 70 + w, y + 7], fill=(30, 58, 138))

    # Red rubber stamp in center
    d.ellipse([240, 400, 410, 520], outline=(220, 38, 38), width=4)
    d.rectangle([260, 445, 390, 475], fill=(220, 38, 38))

    img.save(output_path, "JPEG", quality=95)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic sample documents.")
    project_root = Path(__file__).resolve().parent.parent
    default_out = project_root / "dataset" / "synthetic" / "documents"
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_out),
        help="Target directory for generated synthetic documents (default: dataset/synthetic/documents)"
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generators = [
        ("sample_receipt.jpg", create_retail_receipt),
        ("sample_invoice.jpg", create_commercial_invoice),
        ("sample_form.jpg", create_official_form),
        ("sample_certificate.jpg", create_certificate),
        ("sample_id_card.jpg", create_identity_card),
        ("sample_photographed_doc.jpg", create_photographed_document),
        ("sample_colored_paper.jpg", create_colored_paper_document),
    ]

    print(f"Generating synthetic sample documents at: {out_dir}")
    for filename, fn in generators:
        fn(out_dir / filename)
        print(f"  Generated: {filename}")

    print("\nSynthetic document generation complete!")


if __name__ == "__main__":
    main()
