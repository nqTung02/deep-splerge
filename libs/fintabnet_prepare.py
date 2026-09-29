"""Prepare Deep Splerge split-model data from FinTabNet cell annotations."""

import html
import json
import pickle
import re
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path

import cv2
import numpy as np


IMAGE_PATTERN = re.compile(
    r"^(?P<ticker>[A-Za-z0-9]+)_(?P<year>\d{4})_page_(?P<page>\d+)_table_(?P<table>\d+)$"
)


class FinTabNetHTMLParser(HTMLParser):
    """Recover logical cell spans from FinTabNet's tokenized HTML."""

    def __init__(self):
        super().__init__()
        self.cells = []
        self.grid = {}
        self.row = -1
        self.column = 0

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.row += 1
            self.column = 0
            while (self.row, self.column) in self.grid:
                self.column += 1
            return
        if tag not in ("td", "th"):
            return
        attributes = dict(attrs)
        rowspan = int(attributes.get("rowspan", 1))
        colspan = int(attributes.get("colspan", 1))
        while (self.row, self.column) in self.grid:
            self.column += 1
        cell_index = len(self.cells)
        cell = {
            "row_start": self.row,
            "row_end": self.row + rowspan - 1,
            "col_start": self.column,
            "col_end": self.column + colspan - 1,
            "rowspan": rowspan,
            "colspan": colspan,
            "is_merged": rowspan > 1 or colspan > 1,
        }
        self.cells.append(cell)
        for row in range(cell["row_start"], cell["row_end"] + 1):
            for column in range(cell["col_start"], cell["col_end"] + 1):
                self.grid[(row, column)] = cell_index
        self.column += colspan


def parse_structure(record):
    parser = FinTabNetHTMLParser()
    tokens = record.get("html", {}).get("structure", {}).get("tokens", [])
    parser.feed("".join(tokens))
    rows = max((cell["row_end"] for cell in parser.cells), default=-1) + 1
    columns = max((cell["col_end"] for cell in parser.cells), default=-1) + 1
    return parser.cells, rows, columns


def build_image_map(image_dir):
    """Map FinTabNet source page/table keys to local table-crop paths."""
    image_dir = Path(image_dir)
    candidates = sorted(
        (path for path in image_dir.rglob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda path: (path.stem, path.suffix.lower() != ".png", str(path)),
    )
    image_map = {}
    for path in candidates:
        match = IMAGE_PATTERN.match(path.stem)
        if not match:
            continue
        values = match.groupdict()
        key = (
            f"{values['ticker']}/{values['year']}/page_{values['page']}.pdf",
            int(values["table"]),
        )
        image_map.setdefault(key, path.resolve())
    return image_map


def scan_annotations(cell_jsonl, image_map, max_tables=None):
    """Preflight image coverage and preserve page-local table numbering."""
    page_counts = defaultdict(int)
    matches = []
    total = 0
    with Path(cell_jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if max_tables is not None and total >= max_tables:
                break
            record = json.loads(line)
            filename = str(record.get("filename", ""))
            table_index = page_counts[filename]
            page_counts[filename] += 1
            image_path = image_map.get((filename, table_index))
            if image_path is not None:
                matches.append((line_number, image_path))
            total += 1
    return matches, {
        "annotations": total,
        "matched_images": len(matches),
        "missing_images": total - len(matches),
        "coverage": len(matches) / total if total else 0.0,
    }


def _clean_text(tokens):
    parts = []
    for token in tokens or []:
        clean = re.sub(r"<[^>]+>", "", str(token)).strip()
        if clean:
            parts.append(html.unescape(clean))
    return " ".join(parts)


def _convert_bbox(bbox, table_bbox, width, height):
    if not bbox or len(bbox) != 4 or not table_bbox or len(table_bbox) != 4:
        return None
    table_width = max(float(table_bbox[2]) - float(table_bbox[0]), 1.0)
    table_height = max(float(table_bbox[3]) - float(table_bbox[1]), 1.0)
    scale_x = width / table_width
    scale_y = height / table_height
    converted = [
        (float(bbox[0]) - float(table_bbox[0])) * scale_x,
        (float(table_bbox[3]) - float(bbox[3])) * scale_y,
        (float(bbox[2]) - float(table_bbox[0])) * scale_x,
        (float(table_bbox[3]) - float(bbox[1])) * scale_y,
    ]
    converted[0] = max(0.0, min(float(width), converted[0]))
    converted[2] = max(0.0, min(float(width), converted[2]))
    converted[1] = max(0.0, min(float(height), converted[1]))
    converted[3] = max(0.0, min(float(height), converted[3]))
    if converted[2] <= converted[0] or converted[3] <= converted[1]:
        return None
    return converted


def _boundary_candidates(cells, axis, boundary):
    start_key = "col_start" if axis == "x" else "row_start"
    end_key = "col_end" if axis == "x" else "row_end"
    low_index = 0 if axis == "x" else 1
    high_index = 2 if axis == "x" else 3
    before = [
        cell["bbox"][high_index]
        for cell in cells
        if cell["bbox"] is not None and cell[end_key] == boundary - 1
    ]
    after = [
        cell["bbox"][low_index]
        for cell in cells
        if cell["bbox"] is not None and cell[start_key] == boundary
    ]
    if before and after:
        return (float(np.median(before)) + float(np.median(after))) / 2.0
    values = before or after
    return float(np.median(values)) if values else None


def _interpolate_boundaries(values, extent):
    values[0], values[-1] = 0.0, float(extent)
    known = [index for index, value in enumerate(values) if value is not None]
    for left, right in zip(known, known[1:]):
        if right - left <= 1:
            continue
        step = (values[right] - values[left]) / (right - left)
        for index in range(left + 1, right):
            values[index] = values[left] + step * (index - left)
    rounded = np.rint(np.asarray(values, dtype=np.float64)).astype(np.int32)
    rounded = np.clip(rounded, 0, extent)
    # Separator coordinates must remain ordered even when annotation boxes are noisy.
    rounded = np.maximum.accumulate(rounded)
    rounded[-1] = extent
    return rounded


def logical_boundaries(cells, rows, columns, width, height):
    x_values = [None] * (columns + 1)
    y_values = [None] * (rows + 1)
    for boundary in range(1, columns):
        x_values[boundary] = _boundary_candidates(cells, "x", boundary)
    for boundary in range(1, rows):
        y_values[boundary] = _boundary_candidates(cells, "y", boundary)
    return (
        _interpolate_boundaries(x_values, width),
        _interpolate_boundaries(y_values, height),
    )


def _separator_vector(occupancy, boundaries, max_separator_width):
    extent = occupancy.shape[0]
    occupied = np.flatnonzero(occupancy)
    result = np.zeros(extent, dtype=np.uint8)
    half_width = max(1, max_separator_width // 2)
    for boundary in boundaries[1:-1]:
        boundary = int(boundary)
        before = occupied[occupied < boundary]
        after = occupied[occupied > boundary]
        start = int(before[-1] + 1) if before.size else max(0, boundary - half_width)
        end = int(after[0]) if after.size else min(extent, boundary + half_width + 1)
        if end <= start or end - start > max_separator_width:
            start = max(0, boundary - half_width)
            end = min(extent, boundary + half_width + 1)
        result[start:end] = 255
    return result


def convert_record(record, image, max_separator_width=32):
    """Convert one FinTabNet annotation and crop into Deep Splerge artifacts."""
    height, width = image.shape[:2]
    structure_cells, rows, columns = parse_structure(record)
    if rows < 1 or columns < 1 or not structure_cells:
        raise ValueError("FinTabNet record has no logical table cells")
    table_bbox = record.get("bbox") or record.get("table_bbox")
    raw_cells = record.get("html", {}).get("cells", [])
    cells = []
    ocr = []
    for index, structure in enumerate(structure_cells):
        raw = raw_cells[index] if index < len(raw_cells) else {}
        converted = _convert_bbox(raw.get("bbox"), table_bbox, width, height)
        cell = {**structure, "bbox": converted}
        cells.append(cell)
        text = _clean_text(raw.get("tokens", []))
        if text and converted is not None:
            x0, y0, x1, y1 = [int(round(value)) for value in converted]
            ocr.append([len(text), text, x0, y0, x1, y1])

    x_boundaries, y_boundaries = logical_boundaries(
        cells, rows, columns, width, height)
    text_mask = np.zeros((height, width), dtype=np.uint8)
    for item in ocr:
        cv2.rectangle(text_mask, (item[2], item[3]), (item[4], item[5]), 255, -1)
    # Text inside a spanning cell must not suppress the logical separator label.
    separator_occupancy = text_mask.copy()
    for cell in cells:
        if cell["is_merged"] and cell["bbox"] is not None:
            x0, y0, x1, y1 = [int(round(value)) for value in cell["bbox"]]
            cv2.rectangle(separator_occupancy, (x0, y0), (x1, y1), 0, -1)

    row_labels = _separator_vector(
        np.any(separator_occupancy != 0, axis=1), y_boundaries,
        max_separator_width)
    col_labels = _separator_vector(
        np.any(separator_occupancy != 0, axis=0), x_boundaries,
        max_separator_width)
    return {
        "row_labels": row_labels,
        "col_labels": col_labels,
        "ocr": ocr,
        "rows": rows,
        "columns": columns,
        "row_boundaries": y_boundaries,
        "column_boundaries": x_boundaries,
    }


def _write_vector(path, values):
    path.write_text("".join(f"{int(value)}\n" for value in values), encoding="utf-8")


def prepare_fintabnet(
    cell_jsonl,
    image_dir,
    out_dir,
    max_tables=None,
    allow_missing_images=False,
    max_separator_width=32,
):
    cell_jsonl = Path(cell_jsonl)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    image_map = build_image_map(image_dir)
    matches, coverage = scan_annotations(cell_jsonl, image_map, max_tables)
    if coverage["missing_images"] and not allow_missing_images:
        raise RuntimeError(
            "Incomplete FinTabNet image coverage: "
            f"{coverage['matched_images']}/{coverage['annotations']} annotations matched. "
            "Provide the correct table crops or pass --allow-missing-images for a subset."
        )

    output_images = out_dir / "table_images"
    output_labels = out_dir / "table_split_labels"
    output_ocr = out_dir / "table_ocr"
    for path in (output_images, output_labels, output_ocr):
        path.mkdir(parents=True, exist_ok=True)

    selected = {line_number: image_path for line_number, image_path in matches}
    summary = dict(coverage)
    summary.update({"written_tables": 0, "rows": 0, "columns": 0})
    used_names = set()
    with cell_jsonl.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            image_path = selected.get(line_number)
            if image_path is None:
                continue
            record = json.loads(line)
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Unable to read table image: {image_path}")
            table_name = image_path.stem
            if table_name in used_names:
                raise RuntimeError(f"Duplicate output table name: {table_name}")
            used_names.add(table_name)
            converted = convert_record(record, image, max_separator_width)
            if not cv2.imwrite(str(output_images / f"{table_name}.png"), image):
                raise RuntimeError(f"Unable to write prepared image: {table_name}")
            _write_vector(
                output_labels / f"{table_name}_row.txt", converted["row_labels"])
            _write_vector(
                output_labels / f"{table_name}_col.txt", converted["col_labels"])
            with (output_ocr / f"{table_name}.pkl").open("wb") as handle_out:
                pickle.dump(converted["ocr"], handle_out)
            summary["written_tables"] += 1
            summary["rows"] += converted["rows"]
            summary["columns"] += converted["columns"]
            print(
                f"[{summary['written_tables']}/{coverage['matched_images']}] "
                f"{table_name}: {converted['rows']}x{converted['columns']}"
            )

    summary.update({
        "source_jsonl": str(cell_jsonl.resolve()),
        "image_dir": str(image_dir.resolve()),
        "output_dir": str(out_dir.resolve()),
        "max_separator_width": int(max_separator_width),
    })
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
