"""Импорт проверяется на временных файлах; исходный data/ и out/ не меняются."""

import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import ingestion
import run


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.mapping = {field: field for field in ingestion.REQUIRED_FIELDS}
        self.metadata = {"name": "Проверка импорта", "source": "Тестовая выписка",
                         "coverage": "unknown", "currency": "KZT", "seed_gids": []}

    def csv_file(self, rows, header=None, delimiter=",", filename="upload.csv", bom=False):
        buffer = StringIO(newline="")
        writer = csv.writer(buffer, delimiter=delimiter)
        writer.writerow(header or list(self.mapping))
        writer.writerows(rows)
        path = self.root / filename
        path.write_text(buffer.getvalue(), encoding="utf-8-sig" if bom else "utf-8", newline="")
        return path

    def build(self, path, **changes):
        return ingestion.build_dataset(path, changes.get("mapping", self.mapping),
                                       changes.get("metadata", self.metadata),
                                       changes.get("data_dir", self.root / "dataset"))

    def test_inspection_preserves_text_ids_and_detects_supported_delimiters(self):
        for delimiter in (",", ";", "\t"):
            with self.subTest(delimiter=delimiter):
                path = self.csv_file([["000100000000000000001", "account/2", "1 234,50", "01.07.2026"]],
                                     ["Отправитель", "Получатель", "Сумма", "Дата"], delimiter, bom=True)
                info = ingestion.inspect_upload(path, "выписка.CSV")
                self.assertEqual(info["row_count"], 1)
                self.assertEqual(info["preview"][0]["Отправитель"], "000100000000000000001")
                self.assertEqual(info["suggested_mapping"],
                                 {"src": "Отправитель", "dst": "Получатель", "sum_kzt": "Сумма", "date": "Дата"})
                self.assertFalse((self.root / "dataset").exists())

    def test_identical_real_transactions_remain_and_unknown_coverage_is_explicit(self):
        path = self.csv_file([["000100000000000000001", "@recipient", "1\u00a0234,50", "01.07.2026"]] * 2)
        summary = self.build(path, metadata=self.metadata | {"seed_gids": ["case-only", "000100000000000000001"]})
        edges, nodes, tx = run.load(self.root / "dataset")
        run.validate_inputs(edges, nodes, tx)
        self.assertEqual((summary["row_count"], summary["edge_count"], summary["node_count"]), (2, 1, 3))
        self.assertEqual(summary["total_kzt"], 2469.0)
        self.assertEqual(edges.iloc[0].n_tx, 2)
        self.assertEqual(tx.src.tolist(), ["000100000000000000001"] * 2)
        self.assertEqual(int(nodes.is_seed.sum()), 2)
        self.assertFalse(nodes.depth_known.any())
        self.assertFalse(nodes.outgoing_coverage_known.any())
        self.assertFalse(nodes.seed_incoming_incomplete.any())
        document = json.loads((self.root / "dataset/metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(document["mapping"], self.mapping)
        self.assertEqual(document["summary"], summary)
        self.assertEqual(document["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_duplicate_transaction_id_reports_source_row_and_column(self):
        path = self.csv_file([["a", "b", "10", "2026-07-01", "same"],
                              ["b", "a", "20", "2026-07-02", "same"]],
                             list(self.mapping) + ["operation"])
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path, mapping=self.mapping | {"transaction_id": "operation"})
        self.assertEqual(raised.exception.errors[0]["row"], 3)
        self.assertEqual(raised.exception.errors[0]["column"], "operation")
        self.assertIn("строке 2", raised.exception.errors[0]["message"])
        self.assertFalse((self.root / "dataset").exists())

    def test_bad_values_have_row_and_column_errors_without_partial_dataset(self):
        for amount in ("NaN", "inf", "0", "-1", "12,345", "1 2", "1,234.56"):
            with self.subTest(amount=amount):
                path = self.csv_file([["a", "b", amount, "2026-07-01"]])
                with self.assertRaises(ingestion.ImportValidationError) as raised:
                    self.build(path)
                self.assertEqual(raised.exception.errors[0]["column"], "sum_kzt")
                self.assertEqual(raised.exception.errors[0]["row"], 2)
        path = self.csv_file([["", "b", "12", "2026-02-30"]])
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path)
        self.assertEqual({error["column"] for error in raised.exception.errors}, {"src", "date"})
        self.assertFalse((self.root / "dataset").exists())

    def test_parquet_integer_identifiers_do_not_become_floats_during_iteration(self):
        path = self.root / "upload"
        pd.DataFrame({"src": [2**63 - 1], "dst": [2**63 - 2],
                      "sum_kzt": [12.50], "date": [pd.Timestamp("2026-07-01")]}).to_parquet(path)
        info = ingestion.inspect_upload(path, "upload.parquet")
        self.assertEqual(info["preview"][0]["src"], str(2**63 - 1))
        self.build(path)
        tx = pd.read_parquet(self.root / "dataset/transactions.parquet")
        self.assertEqual(tx.src.tolist(), [str(2**63 - 1)])

    def test_parquet_float_identifiers_are_rejected_even_when_integral(self):
        path = self.root / "upload.parquet"
        pd.DataFrame({"src": [1.0], "dst": ["a"], "sum_kzt": [12.50],
                      "date": [pd.Timestamp("2026-07-01")]}).to_parquet(path)
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path)
        self.assertEqual(raised.exception.errors[0]["row"], 1)
        self.assertEqual(raised.exception.errors[0]["column"], "src")
        self.assertIn("float", str(raised.exception))

    def test_mapping_and_currency_must_be_confirmed_and_consistent(self):
        path = self.csv_file([["a", "b", "1.00", "2026-07-01", "USD"]], list(self.mapping) + ["currency"])
        for mapping in ({}, self.mapping | {"src": "absent"}, self.mapping | {"dst": "src"}):
            with self.subTest(mapping=mapping), self.assertRaises(ingestion.ImportValidationError):
                self.build(path, mapping=mapping)
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path, mapping=self.mapping | {"currency": "currency"})
        self.assertEqual(raised.exception.errors[0]["column"], "currency")
        self.assertEqual(raised.exception.errors[0]["row"], 2)
        with self.assertRaises(ingestion.ImportValidationError):
            self.build(path, metadata=self.metadata | {"currency": "USD"})

    def test_period_bounds_reject_rows_instead_of_silently_filtering(self):
        path = self.csv_file([["a", "b", "1", "2026-07-01"], ["a", "b", "1", "2026-07-03"]])
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path, metadata=self.metadata | {"period_to": "2026-07-02"})
        self.assertEqual(raised.exception.errors[0]["row"], 3)
        self.assertEqual(raised.exception.errors[0]["column"], "date")
        with self.assertRaises(ingestion.ImportValidationError):
            self.build(path, metadata=self.metadata | {"period_from": "2026-07-03", "period_to": "2026-07-01"})

    def test_aware_dates_normalize_to_utc_and_mixed_timezones_require_clarification(self):
        path = self.csv_file([["a", "b", "1", "2026-07-01T10:00:00+05:00"],
                              ["b", "a", "2", "2026-07-01T06:00:00Z"]])
        summary = self.build(path)
        tx = pd.read_parquet(self.root / "dataset/transactions.parquet")
        self.assertEqual(tx.date.iloc[0], pd.Timestamp("2026-07-01T05:00:00Z"))
        self.assertTrue(any("UTC" in warning for warning in summary["warnings"]))
        path = self.csv_file([["a", "b", "1", "2026-07-01T10:00:00+05:00"],
                              ["b", "a", "2", "2026-07-01"]])
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path, data_dir=self.root / "mixed")
        self.assertIn("часовым поясом", raised.exception.errors[0]["message"])

    def test_format_header_and_encoding_errors_are_understandable(self):
        path = self.csv_file([["a", "b", "1", "2026-07-01"]])
        with self.assertRaises(ingestion.ImportValidationError):
            ingestion.inspect_upload(path, "upload.xlsx")
        path.write_bytes(b"src;dst;amount;date\n\xff;b;1;2026-07-01\n")
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            ingestion.inspect_upload(path, "upload.csv")
        self.assertIn("UTF-8", str(raised.exception))
        path.write_text("src,src,sum_kzt,date\na,b,1,2026-07-01\n", encoding="utf-8")
        with self.assertRaises(ingestion.ImportValidationError):
            ingestion.inspect_upload(path, "upload.csv")

    def test_resource_limits_and_error_count_are_bounded(self):
        path = self.csv_file([["a", "b", "1", "2026-07-01"], ["b", "c", "1", "2026-07-01"]])
        with patch.object(ingestion, "MAX_ROWS", 1), self.assertRaises(ingestion.ImportValidationError):
            ingestion.inspect_upload(path, "upload.csv")
        with patch.object(ingestion, "MAX_NODES", 2), self.assertRaises(ingestion.ImportValidationError):
            self.build(path)
        with patch.object(ingestion, "MAX_FILE_BYTES", 1), self.assertRaises(ingestion.ImportValidationError):
            ingestion.inspect_upload(path, "upload.csv")
        path = self.csv_file([["a", "b", "bad", "2026-07-01"]] * 120)
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            self.build(path)
        self.assertEqual(len(raised.exception.errors), 100)

    def test_parquet_metadata_limits_and_nested_fields_are_checked_before_loading(self):
        path = self.root / "upload.parquet"
        pd.DataFrame({"src": ["a"], "dst": ["b"], "sum_kzt": [10],
                      "date": [pd.Timestamp("2026-07-01")]}).to_parquet(path)
        for limit in ("MAX_COLUMNS", "MAX_UNCOMPRESSED_BYTES", "MAX_ROWS"):
            with self.subTest(limit=limit), patch.object(ingestion, limit, 0), \
                    patch.object(ingestion.pd, "read_parquet") as reader, \
                    self.assertRaises(ingestion.ImportValidationError):
                ingestion.inspect_upload(path, "upload.parquet")
            reader.assert_not_called()
        pd.DataFrame({"values": [[1, 2]]}).to_parquet(path)
        with self.assertRaises(ingestion.ImportValidationError) as raised:
            ingestion.inspect_upload(path, "upload.parquet")
        self.assertIn("Вложенные", str(raised.exception))
        csv_path = self.csv_file([["a", "b", "10", "2026-07-01"]])
        with patch.object(ingestion, "MAX_COLUMNS", 3), self.assertRaises(ingestion.ImportValidationError):
            ingestion.inspect_upload(csv_path, "upload.csv")

    def test_preview_is_bounded_without_truncating_uploaded_values(self):
        path = self.root / "upload.parquet"
        pd.DataFrame({"src": ["a"], "dst": ["b"], "sum_kzt": [10],
                      "date": [pd.Timestamp("2026-07-01")], "comment": ["x" * 1000]}).to_parquet(path)
        info = ingestion.inspect_upload(path, "upload.parquet")
        self.assertEqual(len(info["preview"][0]["comment"]), 300)
        self.assertTrue(info["preview"][0]["comment"].endswith("…"))
        self.assertEqual(len(pd.read_parquet(path).iloc[0]["comment"]), 1000)

    def test_existing_dataset_cannot_be_overwritten(self):
        path = self.csv_file([["a", "b", "1", "2026-07-01"]])
        self.build(path)
        before = (self.root / "dataset/transactions.parquet").read_bytes()
        with self.assertRaises(ingestion.ImportValidationError):
            self.build(path)
        self.assertEqual((self.root / "dataset/transactions.parquet").read_bytes(), before)

    def test_unknown_outgoing_does_not_become_consolidator_or_terminal(self):
        path = self.csv_file([[f"payer{i}", "recipient", "100000", "2026-07-01"] for i in range(5)])
        self.build(path, metadata=self.metadata | {"seed_gids": ["recipient"]})
        edges, nodes, tx = run.load(self.root / "dataset")
        roles = run.assign_roles(run.compute_features(edges, nodes))
        target = roles.set_index("gid").loc["recipient"]
        self.assertEqual(target.role, "peripheral")
        self.assertFalse(roles.truncated_by_depth.any())
        self.assertFalse(target.incoming_incomplete)
        text = run.evidence(target)
        self.assertIn("полнота исходящих неизвестна", text)
        self.assertNotIn("занижены", text)
        self.assertNotIn("глубина 0", text)

    def test_complete_period_allows_terminal_without_inventing_depth(self):
        path = self.csv_file([["a", "b", "100000", "2026-07-01"]])
        self.build(path, metadata=self.metadata | {"coverage": "complete"})
        edges, nodes, _ = run.load(self.root / "dataset")
        roles = run.assign_roles(run.compute_features(edges, nodes))
        target = roles.set_index("gid").loc["b"]
        self.assertEqual(target.role, "terminal")
        self.assertIn("за наблюдаемый период", run.evidence(target))
        self.assertNotIn("глубина", run.evidence(target))

    def test_single_self_transfer_can_be_analysed_offline(self):
        path = self.csv_file([["account/1", "account/1", "1000", "2026-07-01"]])
        self.build(path)
        edges, nodes, tx = run.load(self.root / "dataset")
        run.validate_inputs(edges, nodes, tx)
        roles = run.add_priority(run.assign_roles(run.compute_features(edges, nodes)))
        roles["evidence"] = [run.evidence(row) for row in roles.itertuples(index=False)]
        run.write_outputs(roles, edges, self.root / "result", offline=True)
        self.assertEqual(len(pd.read_csv(self.root / "result/nodes_roles.csv")), 1)
        self.assertEqual(roles.hubs.iloc[0], 1.0)
        self.assertEqual(roles.authorities.iloc[0], 1.0)


if __name__ == "__main__":
    unittest.main()
