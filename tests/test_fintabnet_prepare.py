import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from libs.fintabnet_prepare import (
    convert_fintabnet_c,
    convert_record,
    load_fintabnet_c_words,
    parse_fintabnet_c_xml,
    prepare_fintabnet,
    prepare_fintabnet_c,
)


def make_record():
    return {
        "filename": "ABC/2020/page_1.pdf",
        "split": "train",
        "bbox": [0, 0, 100, 80],
        "html": {
            "structure": {"tokens": [
                "<table>", "<tr>", '<td colspan="2">', "</td>", "</tr>",
                "<tr>", "<td>", "</td>", "<td>", "</td>", "</tr>",
                "</table>",
            ]},
            "cells": [
                {"bbox": [0, 40, 100, 80], "tokens": ["Header"]},
                {"bbox": [0, 0, 50, 40], "tokens": ["Left"]},
                {"bbox": [50, 0, 100, 40], "tokens": ["Right"]},
            ],
        },
    }


def write_fintabnet_c_sample(root, stem="ABC_2020_page_1_table_0"):
    for directory in ("images", "train", "val", "test", "words"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    image = np.full((80, 100), 255, dtype=np.uint8)
    cv2.imwrite(str(root / "images" / f"{stem}.jpg"), image)
    objects = [
        ("table", (0, 0, 100, 80)),
        ("table row", (0, 0, 100, 40)),
        ("table row", (0, 40, 100, 80)),
        ("table column", (0, 0, 50, 80)),
        ("table column", (50, 0, 100, 80)),
        ("table spanning cell", (0, 0, 100, 40)),
    ]
    object_xml = "".join(
        "<object><name>{}</name><bndbox><xmin>{}</xmin><ymin>{}</ymin>"
        "<xmax>{}</xmax><ymax>{}</ymax></bndbox></object>".format(name, *bbox)
        for name, bbox in objects
    )
    (root / "train" / f"{stem}.xml").write_text(
        "<annotation><filename>{}.jpg</filename>"
        "<size><width>100</width><height>80</height><depth>3</depth></size>"
        "{}</annotation>".format(stem, object_xml),
        encoding="utf-8",
    )
    words = [
        {"text": "Header", "bbox": [10, 10, 90, 25]},
        {"text": "Left", "bbox": [5, 50, 35, 65]},
        {"text": "Right", "bbox": [60, 50, 95, 65]},
    ]
    (root / "words" / f"{stem}_words.json").write_text(
        json.dumps(words), encoding="utf-8")
    return stem, image


class FinTabNetPreparationTests(unittest.TestCase):
    def test_record_conversion_builds_expected_grid_and_vectors(self):
        image = np.full((80, 100), 255, dtype=np.uint8)
        result = convert_record(make_record(), image, max_separator_width=8)
        self.assertEqual((result["rows"], result["columns"]), (2, 2))
        np.testing.assert_array_equal(result["row_boundaries"], [0, 40, 80])
        np.testing.assert_array_equal(result["column_boundaries"], [0, 50, 100])
        self.assertEqual(result["row_labels"].shape, (80,))
        self.assertEqual(result["col_labels"].shape, (100,))
        self.assertGreater(np.count_nonzero(result["row_labels"]), 0)
        self.assertGreater(np.count_nonzero(result["col_labels"]), 0)
        self.assertEqual([item[1] for item in result["ocr"]], ["Header", "Left", "Right"])

    def test_end_to_end_preparation_writes_deep_splerge_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            image_path = images / "ABC_2020_page_1_table_0.png"
            cv2.imwrite(str(image_path), np.full((80, 100), 255, dtype=np.uint8))
            annotations = root / "cells.jsonl"
            annotations.write_text(json.dumps(make_record()) + "\n", encoding="utf-8")
            output = root / "prepared"

            summary = prepare_fintabnet(annotations, images, output)

            self.assertEqual(summary["written_tables"], 1)
            self.assertEqual(summary["missing_images"], 0)
            self.assertTrue((output / "table_images" / f"{image_path.stem}.png").is_file())
            row_path = output / "table_split_labels" / f"{image_path.stem}_row.txt"
            col_path = output / "table_split_labels" / f"{image_path.stem}_col.txt"
            self.assertEqual(len(row_path.read_text().splitlines()), 80)
            self.assertEqual(len(col_path.read_text().splitlines()), 100)
            with (output / "table_ocr" / f"{image_path.stem}.pkl").open("rb") as handle:
                self.assertEqual(len(pickle.load(handle)), 3)

    def test_missing_images_fail_before_output_is_created(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            annotations = root / "cells.jsonl"
            annotations.write_text(json.dumps(make_record()) + "\n", encoding="utf-8")
            output = root / "prepared"
            with self.assertRaises(RuntimeError):
                prepare_fintabnet(annotations, images, output)
            self.assertFalse(output.exists())

    def test_fintabnet_c_parses_official_pascal_voc_and_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stem, image = write_fintabnet_c_sample(root)
            annotation = parse_fintabnet_c_xml(root / "train" / f"{stem}.xml")
            words = load_fintabnet_c_words(
                root / "words" / f"{stem}_words.json", 100, 80)
            result = convert_fintabnet_c(annotation, words, image, 8)

            self.assertEqual((result["rows"], result["columns"]), (2, 2))
            np.testing.assert_array_equal(result["row_boundaries"], [0, 40, 80])
            np.testing.assert_array_equal(result["column_boundaries"], [0, 50, 100])
            self.assertEqual([item[1] for item in result["ocr"]], [
                "Header", "Left", "Right"])
            self.assertGreater(np.count_nonzero(result["row_labels"]), 0)
            self.assertGreater(np.count_nonzero(result["col_labels"]), 0)

    def test_fintabnet_c_end_to_end_uses_extracted_tree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "FinTabNet.c-Structure"
            stem, _image = write_fintabnet_c_sample(root)
            output = Path(directory) / "prepared"

            summary = prepare_fintabnet_c(root, "train", output)

            self.assertEqual(summary["annotations"], 1)
            self.assertEqual(summary["written_tables"], 1)
            self.assertEqual(summary["coverage"], 1.0)
            self.assertTrue((output / "table_images" / f"{stem}.png").is_file())
            self.assertEqual(
                len((output / "table_split_labels" / f"{stem}_row.txt")
                    .read_text().splitlines()),
                80,
            )
            with (output / "table_ocr" / f"{stem}.pkl").open("rb") as handle:
                self.assertEqual(len(pickle.load(handle)), 3)

    def test_fintabnet_c_fails_before_writing_when_words_are_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "FinTabNet.c-Structure"
            stem, _image = write_fintabnet_c_sample(root)
            (root / "words" / f"{stem}_words.json").unlink()
            output = Path(directory) / "prepared"

            with self.assertRaises(RuntimeError):
                prepare_fintabnet_c(root, "train", output)
            self.assertFalse(output.exists())

    def test_fintabnet_c_cli_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "FinTabNet.c-Structure"
            stem, _image = write_fintabnet_c_sample(root)
            output = Path(directory) / "prepared"
            repository = Path(__file__).resolve().parents[1]

            completed = subprocess.run(
                [
                    sys.executable,
                    str(repository / "prepare_data.py"),
                    "--dataset-format", "fintabnet-c",
                    "--fintabnet-c-root", str(root),
                    "--split", "train",
                    "--out_dir", str(output),
                ],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertIn("FinTabNet.c preparation summary", completed.stdout)
            self.assertTrue((output / "table_images" / f"{stem}.png").is_file())


if __name__ == "__main__":
    unittest.main()
