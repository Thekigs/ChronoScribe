"""Offline regression tests for the standard-CSV ChronoScribe SFT builder."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.build_chronoscribe_sft import (
    RESPONSE_SYSTEM_PROMPT,
    ROUTER_SYSTEM_PROMPT,
    build_chronoscribe_sft_records,
    read_standard_timeseries_csv,
    split_paired_records,
    validate_chronoscribe_sft_records,
)


class ChronoScribeSFTBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temporary_directory.name) / "synthetic_training_source.csv"
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp", "load", "temperature"])
            for index in range(480):
                writer.writerow([
                    f"2026-01-{(index // 24) + 1:02d}T{index % 24:02d}:00:00",
                    20 + (index % 24) * 0.5 + (index // 96) * 0.1,
                    10 + ((index * 3) % 17) * 0.25,
                ])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def _records(self):
        series = read_standard_timeseries_csv(self.csv_path)
        return build_chronoscribe_sft_records(
            series,
            dataset_name="synthetic_provenance_disjoint",
            domain="energy",
            horizons=(24,),
            context_candidates=(64, 128),
            stride=72,
            max_windows_per_series=3,
        )

    def test_records_match_current_controller_contract(self) -> None:
        records = self._records()
        validate_chronoscribe_sft_records(records)
        self.assertEqual(len(records), 12)
        directive_records = [record for record in records if record["task_type"] == "directive_generation"]
        response_records = [record for record in records if record["task_type"] == "evidence_conditioned_response"]
        self.assertEqual(len(directive_records), len(response_records))
        for record in directive_records:
            messages = record["messages"]
            self.assertEqual(messages[0]["content"], ROUTER_SYSTEM_PROMPT)
            target = json.loads(messages[-1]["content"])
            self.assertEqual(set(target), {"use_chronos2", "prediction_length", "context_length"})
            self.assertNotIn("predicted_values", messages[-1]["content"])
            self.assertNotIn("future_values", messages[1]["content"])
            self.assertFalse(record["metadata"]["future_values_serialized"])
        for record in response_records:
            messages = record["messages"]
            self.assertEqual(messages[0]["content"], RESPONSE_SYSTEM_PROMPT)
            target = json.loads(messages[-1]["content"])
            self.assertEqual(set(target), {"trend", "analysis"})
            evidence = record["metadata"]["evidence"]
            self.assertEqual(len(evidence["p10"]), 24)
            self.assertEqual(len(evidence["p10"]), len(evidence["p50"]))
            self.assertEqual(len(evidence["p50"]), len(evidence["p90"]))
            self.assertNotIn("P10-P90", messages[1]["content"])
            self.assertIn("预测:24步", messages[1]["content"])

    def test_split_is_deterministic_and_keeps_pairs_together(self) -> None:
        records = self._records()
        train_a, validation_a = split_paired_records(records, validation_fraction=0.25, seed=42)
        train_b, validation_b = split_paired_records(records, validation_fraction=0.25, seed=42)
        self.assertEqual([item["id"] for item in train_a], [item["id"] for item in train_b])
        self.assertEqual([item["id"] for item in validation_a], [item["id"] for item in validation_b])
        train_pairs = {item["pair_id"] for item in train_a}
        validation_pairs = {item["pair_id"] for item in validation_a}
        self.assertFalse(train_pairs.intersection(validation_pairs))
        self.assertTrue(all(sum(item["pair_id"] == pair for item in train_a) == 2 for pair in train_pairs))
        self.assertTrue(
            all(sum(item["pair_id"] == pair for item in validation_a) == 2 for pair in validation_pairs)
        )


if __name__ == "__main__":
    unittest.main()
