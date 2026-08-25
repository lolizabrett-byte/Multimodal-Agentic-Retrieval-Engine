from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

TILE_WIDTH = 320
TILE_HEIGHT = 180
LABEL_HEIGHT = 32
COLUMNS = 4


def write_contact_sheet(tiles: list[tuple[Path, str]], output: Path) -> Path | None:
    """Ghép danh sách (đường_dẫn_ảnh, nhãn) thành 1 ảnh lưới JPG tại `output`.

    Trả về `output` khi ghi thành công, `None` khi `tiles` rỗng (không sinh
    file hỏng).
    """
    if not tiles:
        return None
    rows = max(1, (len(tiles) + COLUMNS - 1) // COLUMNS)
    sheet = Image.new(
        "RGB", (COLUMNS * TILE_WIDTH, rows * (TILE_HEIGHT + LABEL_HEIGHT)), "black"
    )
    draw = ImageDraw.Draw(sheet)
    for index, (image_path, label) in enumerate(tiles):
        x = (index % COLUMNS) * TILE_WIDTH
        y = (index // COLUMNS) * (TILE_HEIGHT + LABEL_HEIGHT)
        with Image.open(Path(image_path)) as image:
            tile = ImageOps.fit(
                image.convert("RGB"),
                (TILE_WIDTH, TILE_HEIGHT),
                method=Image.Resampling.LANCZOS,
            )
        sheet.paste(tile, (x, y))
        draw.text((x + 4, y + TILE_HEIGHT + 6), label, fill="white")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="JPEG", quality=90, subsampling=0)
    return output
