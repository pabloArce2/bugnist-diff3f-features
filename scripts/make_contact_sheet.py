import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile existing preview PNGs (or any images) into one labeled row/column grid image."
    )
    parser.add_argument(
        "--cell",
        nargs=3,
        action="append",
        metavar=("ROW", "COL", "IMAGE"),
        required=True,
        help="One grid cell: row label, column label, path to the image to place there. "
        "Row/column order follows first appearance.",
    )
    parser.add_argument("--out", required=True, help="Output grid PNG path.")
    parser.add_argument("--cell-size", type=int, default=420, help="Each cell's square thumbnail size in pixels.")
    parser.add_argument("--col-header-height", type=int, default=34)
    parser.add_argument("--row-label-width", type=int, default=150)
    parser.add_argument("--gap", type=int, default=6)
    parser.add_argument("--title", default=None, help="Optional title drawn above the grid.")
    return parser.parse_args()


def thumbnail_in_box(path, box_size):
    image = Image.open(path).convert("RGB")
    image.thumbnail((box_size, box_size), Image.LANCZOS)
    canvas = Image.new("RGB", (box_size, box_size), (248, 248, 248))
    offset = ((box_size - image.width) // 2, (box_size - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def centered_x(draw, text, font, left, box_width):
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    return left + max(0, (box_width - text_w) // 2)


def main():
    args = parse_args()

    rows = []
    cols = []
    cells = {}
    for row, col, image_path in args.cell:
        if row not in rows:
            rows.append(row)
        if col not in cols:
            cols.append(col)
        cells[(row, col)] = image_path

    font = ImageFont.load_default()
    title_h = 30 if args.title else 0
    grid_left = args.row_label_width + args.gap
    grid_top = title_h + args.col_header_height + args.gap

    width = grid_left + len(cols) * (args.cell_size + args.gap)
    height = grid_top + len(rows) * (args.cell_size + args.gap)
    grid = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    if args.title:
        draw.text((args.gap, 6), args.title, fill=(20, 20, 20), font=font)

    for c, col in enumerate(cols):
        x = grid_left + c * (args.cell_size + args.gap)
        text_x = centered_x(draw, col, font, x, args.cell_size)
        draw.text((text_x, title_h + 10), col, fill=(20, 20, 20), font=font)

    for r, row in enumerate(rows):
        y = grid_top + r * (args.cell_size + args.gap)
        draw.text((args.gap, y + args.cell_size // 2 - 6), row, fill=(20, 20, 20), font=font)

    for (row, col), image_path in cells.items():
        r = rows.index(row)
        c = cols.index(col)
        x = grid_left + c * (args.cell_size + args.gap)
        y = grid_top + r * (args.cell_size + args.gap)
        thumb = thumbnail_in_box(image_path, args.cell_size)
        grid.paste(thumb, (x, y))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
