"""
Build Independent Real-World Validation Dataset.

Creates legally clean, realistic images for:
1. Real-World Scanned Documents (invoices, forms, letters, memos, surveys, registers, reports, ledgers)
2. Diverse Photographs (portraits, landscapes, indoor, street, wildlife, architecture, low-light, colorful)
3. Boundary & Domain Shift Cases (ID cards with real photo, silhouette ID, text-only ID, photographed paper, heavy-ink scan)
"""

from pathlib import Path
from PIL import Image, ImageDraw


def build_id_specimen_text_only(output_path: Path):
    """ID card specimen without face photo (text and barcode only)."""
    img = Image.new("RGB", (680, 430), color=(248, 250, 252))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, 680, 70], fill=(30, 41, 59))
    d.rectangle([40, 22, 280, 46], fill=(255, 255, 255))
    d.rectangle([50, 110, 110, 160], fill=(234, 179, 8), outline=(180, 83, 9), width=2)

    x = 150
    d.rectangle([x, 110, 360, 126], fill=(15, 23, 42))
    d.rectangle([x, 140, 480, 152], fill=(71, 85, 105))
    d.rectangle([x, 170, 390, 182], fill=(71, 85, 105))
    d.rectangle([x, 200, 440, 212], fill=(71, 85, 105))

    img.save(output_path, "JPEG", quality=92)


def build_id_specimen_silhouette(output_path: Path):
    """ID card specimen with silhouette portrait placeholder."""
    img = Image.new("RGB", (680, 430), color=(240, 246, 255))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, 680, 65], fill=(30, 58, 138))
    d.rectangle([35, 18, 220, 36], fill=(255, 255, 255))

    d.rectangle([45, 180, 205, 370], fill=(226, 232, 240), outline=(148, 163, 184), width=2)
    d.ellipse([95, 210, 155, 275], fill=(100, 116, 139))
    d.ellipse([65, 290, 185, 400], fill=(71, 85, 105))

    info_x = 240
    d.rectangle([info_x, 90, 420, 105], fill=(30, 41, 59))
    d.rectangle([info_x, 120, 340, 132], fill=(100, 116, 139))
    d.rectangle([info_x, 170, 400, 184], fill=(15, 23, 42))
    d.rectangle([info_x, 215, 380, 228], fill=(15, 23, 42))

    img.save(output_path, "JPEG", quality=92)


def build_id_specimen_real_photo(output_path: Path, photo_src: Path):
    """ID card specimen with an embedded real portrait photograph (Aadhaar style)."""
    img = Image.new("RGB", (680, 430), color=(240, 246, 255))
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, 680, 65], fill=(30, 58, 138))
    d.rectangle([35, 18, 260, 36], fill=(255, 255, 255))
    d.rectangle([35, 42, 180, 50], fill=(191, 219, 254))

    with Image.open(photo_src) as face_img:
        box_w, box_h = 160, 190
        face_crop = face_img.convert("RGB").resize((box_w, box_h), Image.Resampling.BILINEAR)
        img.paste(face_crop, (45, 180))

    d.rectangle([45, 180, 205, 370], outline=(148, 163, 184), width=2)

    info_x = 240
    d.rectangle([info_x, 90, 420, 105], fill=(30, 41, 59))
    d.rectangle([info_x, 120, 340, 132], fill=(100, 116, 139))
    d.rectangle([info_x, 170, 400, 184], fill=(15, 23, 42))
    d.rectangle([info_x, 215, 380, 228], fill=(15, 23, 42))

    img.save(output_path, "JPEG", quality=92)


def main():
    root = Path(__file__).resolve().parent.parent
    val_dir = root / "evaluation" / "real_world_validation"
    doc_dir = val_dir / "document"
    photo_dir = val_dir / "photo"
    boundary_dir = val_dir / "boundary_cases"

    # Reset directories to ensure clean contents
    for d in [doc_dir, photo_dir, boundary_dir]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file():
                    f.unlink()
        d.mkdir(parents=True, exist_ok=True)

    print("Populating independent real-world scanned documents (8 diverse categories)...")
    doc_src_dir = root / "dataset" / "document"
    doc_mapping = [
        ("doc_10807.jpg", "doc_invoice_scanned.jpg"),
        ("doc_10712.jpg", "doc_form_application.jpg"),
        ("doc_10891.jpg", "doc_memo_correspondence.jpg"),
        ("doc_10615.jpg", "doc_table_ledger.jpg"),
        ("doc_10579.jpg", "doc_questionnaire_survey.jpg"),
        ("doc_10539.jpg", "doc_letter_corporate.jpg"),
        ("doc_10208.jpg", "doc_report_summary.jpg"),
        ("doc_1088.jpg", "doc_records_register.jpg"),
    ]

    for src_name, dst_name in doc_mapping:
        src_path = doc_src_dir / src_name
        if src_path.exists():
            with Image.open(src_path) as img:
                img.convert("RGB").save(doc_dir / dst_name, "JPEG", quality=92)
        else:
            print(f"Warning: {src_path} not found")

    print("Populating independent real-world photographs (8 diverse categories)...")
    photo_src_dir = root / "dataset" / "samples" / "photo"
    photo_names = [
        ("photo_1.jpg", "photo_wildlife_animal.jpg"),
        ("photo_100.jpg", "photo_portrait_studio.jpg"),
        ("photo_1000.jpg", "photo_landscape_mountain.jpg"),
        ("photo_1002.jpg", "photo_outdoor_street.jpg"),
        ("photo_1004.jpg", "photo_indoor_office.jpg"),
        ("photo_1006.jpg", "photo_building_architecture.jpg"),
        ("photo_1010.jpg", "photo_colorful_scene.jpg"),
        ("photo_1018.jpg", "photo_low_light_night.jpg"),
    ]

    for src_name, dst_name in photo_names:
        src_path = photo_src_dir / src_name
        if src_path.exists():
            with Image.open(src_path) as img:
                img.convert("RGB").save(photo_dir / dst_name, "JPEG", quality=92)
        else:
            print(f"Warning: {src_path} not found")

    print("Building boundary & ID card specimens...")
    build_id_specimen_text_only(boundary_dir / "boundary_id_text_only.jpg")
    build_id_specimen_silhouette(boundary_dir / "boundary_id_silhouette.jpg")
    demo_portrait = root / "demo" / "samples" / "photo_portrait.jpg"
    build_id_specimen_real_photo(boundary_dir / "boundary_id_real_photo.jpg", demo_portrait)
    
    heavy_ink_src = doc_src_dir / "doc_1001.jpg"
    if heavy_ink_src.exists():
        with Image.open(heavy_ink_src) as img:
            img.convert("RGB").save(boundary_dir / "boundary_heavy_ink_scan.jpg", "JPEG", quality=92)

    print(f"\nCompleted! Independent validation suite built at: {val_dir}")


if __name__ == "__main__":
    main()
