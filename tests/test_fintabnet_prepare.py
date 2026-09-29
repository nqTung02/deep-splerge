import json
import pickle
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from libs.fintabnet_prepare import convert_record, prepare_fintabnet


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


if __name__ == "__main__":
    unittest.main()
