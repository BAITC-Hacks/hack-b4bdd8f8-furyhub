"""Offline regression checks for structured, cached LLM explanations.

Every API client is replaced with a mock; no test contacts OpenAI.
"""

from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

import llm_layer


ROLES = ("consolidator", "transit", "distributor", "terminal", "coordinator", "peripheral")
GID = "100000000343175100"
OTHER_GID = "100000000343175101"


def node_payload(**changes):
    node = {
        "gid": GID, "role": "coordinator", "role_score": 0.8,
        "cluster_id": 0, "priority_score": 0.7,
        "evidence": "8 входящих связей; признаки координации", "depth": 0,
        "is_seed": True, "in_deg": 8, "out_deg": 3,
        "in_kzt": 123456.78, "out_kzt": 90000.0,
        "in_tx": 12, "out_tx": 5, "pagerank": 0.01,
        "pass_through": 0.729, "truncated_by_depth": False,
        "turnover_kzt": 213456.78, "betweenness": 0.03,
        "hubs": 0.02, "authorities": 0.04,
        "seed_payers": 1, "seed_payers_share": 0.125,
        "role_rule": "coordinator", "priority_score_raw": 0.6,
    }
    node.update(changes)
    return {
        "node": node,
        "counterparties": {
            "incoming": [{"gid": OTHER_GID, "role": "transit", "sum_kzt": 123456.78, "n_tx": 12}],
            "outgoing": [{"gid": OTHER_GID, "role": "transit", "sum_kzt": 90000.0, "n_tx": 5}],
        },
    }


def cluster_payload(cluster_id=0):
    return {
        "cluster_id": cluster_id, "n_nodes": 8, "n_seed": 1,
        "sum_kzt_internal": 100000.0, "turnover_kzt": 250000.0,
        "turnover_share": 0.125,
        "role_counts": dict(zip(ROLES, (1, 1, 1, 3, 1, 1))),
        "top_nodes": [node_payload()["node"]],
    }


def schema_selection(request, choose_last=False):
    """Return a schema-valid choice without relying on implementation codes."""
    schema = request["text"]["format"]["schema"]
    return {
        name: {
            field: definition["enum"][-1 if choose_last else 0]
            for field, definition in item["properties"].items()
        }
        for name, item in schema["properties"].items()
    }


def completed_response(request, choose_last=False):
    return SimpleNamespace(
        status="completed", output_text=json.dumps(schema_selection(request, choose_last)),
        output=[],
    )


class LLMLayerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.env_path = self.root / "absent.env"

    @contextmanager
    def api(self, *, key=True, create=None, dotenv=None):
        constructor = Mock()
        client = constructor.return_value
        client.responses.create.side_effect = create or (lambda **kwargs: completed_response(kwargs))
        environment = {"OPENAI_API_KEY": "unit-test-key"} if key else {}
        with patch.dict(os.environ, environment, clear=True), \
                patch.object(llm_layer, "OpenAI", constructor), \
                patch.object(llm_layer, "dotenv_values", return_value=dotenv or {}):
            yield constructor, client.responses.create

    def layer(self, subdir="out", model="test-model"):
        return llm_layer.LLMLayer(self.root / subdir, env_path=self.env_path, model=model)

    def assert_hypothesis(self, text):
        self.assertTrue(text.startswith("Гипотеза для проверки:"), text)
        terminators = re.findall(r"(?<!\d)[.!?](?!\d)|(?<=\d)[.!?](?!\d)", text)
        self.assertEqual(len(terminators), 2, text)

    def test_no_key_returns_two_sentence_fallback_and_writes_cache(self):
        with self.api(key=False) as (constructor, create):
            texts = self.layer().hypotheses([cluster_payload()])
        self.assertEqual(len(texts), 1)
        self.assert_hypothesis(texts[0])
        self.assertIn("seed", texts[0])
        constructor.assert_not_called()
        create.assert_not_called()
        cache = json.loads((self.root / "out/llm_cache.json").read_text(encoding="utf-8"))
        self.assertEqual(cache["version"], llm_layer.CACHE_VERSION)
        self.assertTrue(cache["entries"])
        self.assertTrue(all(entry["source"] == "fallback" for entry in cache["entries"].values()))

    def test_client_timeout_retries_and_strict_structured_outputs(self):
        with self.api() as (constructor, create):
            texts = self.layer().hypotheses([cluster_payload()])
        self.assert_hypothesis(texts[0])
        kwargs = constructor.call_args.kwargs
        self.assertEqual(kwargs["timeout"], 30.0)
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertEqual(kwargs["api_key"], "unit-test-key")
        request = create.call_args.kwargs
        self.assertEqual(request["model"], "test-model")
        self.assertIs(request["store"], False)
        output_format = request["text"]["format"]
        self.assertEqual(output_format["type"], "json_schema")
        self.assertEqual(output_format["name"], "aml_cluster")
        self.assertIs(output_format["strict"], True)
        self.assertIs(output_format["schema"]["additionalProperties"], False)
        self.assertIn(GID, json.dumps(request["input"], ensure_ascii=False))

    def test_dotenv_key_is_read_without_changing_environment(self):
        with self.api(key=False, dotenv={"OPENAI_API_KEY": "dotenv-test-key"}) as (constructor, create):
            self.layer().hypotheses([cluster_payload()])
            self.assertNotIn("OPENAI_API_KEY", os.environ)
        self.assertEqual(constructor.call_args.kwargs["api_key"], "dotenv-test-key")
        self.assertEqual(create.call_count, 1)

    def test_other_valid_model_selection_changes_hypothesis(self):
        with self.api(create=lambda **kwargs: completed_response(kwargs)):
            first = self.layer("first").hypotheses([cluster_payload()])[0]
        with self.api(create=lambda **kwargs: completed_response(kwargs, choose_last=True)):
            second = self.layer("second").hypotheses([cluster_payload()])[0]
        self.assertNotEqual(first, second)
        self.assert_hypothesis(first)
        self.assert_hypothesis(second)

    def test_network_error_breaks_all_subsequent_batches(self):
        payloads = [cluster_payload(i) for i in range(51)]
        with self.api(create=OSError("offline")) as (_, create):
            layer = self.layer()
            result = layer.hypotheses(payloads)
            card = layer.node_card(node_payload())
        self.assertEqual(len(result), 51)
        self.assertTrue(card)
        self.assertEqual(create.call_count, 1)
        for text in result:
            self.assert_hypothesis(text)

    def test_valid_responses_use_batches_of_at_most_twenty_five(self):
        with self.api() as (_, create):
            texts = self.layer().hypotheses([cluster_payload(i) for i in range(51)])
        self.assertEqual(len(texts), 51)
        self.assertEqual(create.call_count, 3)
        sizes = [len(call.kwargs["text"]["format"]["schema"]["properties"]) for call in create.call_args_list]
        self.assertEqual(sizes, [25, 25, 1])

    def test_valid_llm_cache_is_reused_with_and_without_key(self):
        with self.api() as (_, create):
            first = self.layer().hypotheses([cluster_payload()])
            second = self.layer().hypotheses([cluster_payload()])
        self.assertEqual(create.call_count, 1)
        self.assertEqual(first, second)
        with self.api(key=False) as (_, create):
            offline = self.layer().hypotheses([cluster_payload()])
        self.assertEqual(offline, first)
        create.assert_not_called()

    def test_cache_invalidates_for_amount_model_and_top_node_metric(self):
        original = cluster_payload()
        changed_amount = deepcopy(original)
        changed_amount["sum_kzt_internal"] += 100
        changed_top = deepcopy(original)
        changed_top["top_nodes"][0]["priority_score"] += 0.1
        with self.api() as (_, create):
            self.layer().hypotheses([original])
            self.layer().hypotheses([changed_amount])
            self.layer().hypotheses([changed_top])
            self.layer(model="different-model").hypotheses([original])
        self.assertEqual(create.call_count, 4)

    def test_fallback_cache_is_retried_when_api_becomes_available(self):
        with self.api(key=False):
            self.layer().hypotheses([cluster_payload()])
        with self.api() as (_, create):
            self.layer().hypotheses([cluster_payload()])
        self.assertEqual(create.call_count, 1)
        cache = json.loads((self.root / "out/llm_cache.json").read_text(encoding="utf-8"))
        self.assertTrue(any(entry["source"] == "llm" for entry in cache["entries"].values()))

    def test_corrupt_cache_does_not_prevent_explanations(self):
        out = self.root / "out"
        out.mkdir()
        (out / "llm_cache.json").write_text("{not valid json", encoding="utf-8")
        with self.api() as (_, create):
            texts = self.layer().hypotheses([cluster_payload()])
        self.assertEqual(create.call_count, 1)
        self.assert_hypothesis(texts[0])
        json.loads((out / "llm_cache.json").read_text(encoding="utf-8"))

    def test_refused_incomplete_invalid_and_invented_choices_fall_back(self):
        with self.api(key=False):
            expected = self.layer("baseline").hypotheses([cluster_payload()])

        def invented(**request):
            values = schema_selection(request)
            next(iter(values.values()))["purpose"] = "преступление_999999"
            return SimpleNamespace(status="completed", output_text=json.dumps(values), output=[])

        scenarios = [
            lambda **request: SimpleNamespace(status="incomplete", output_text=json.dumps(schema_selection(request)), output=[]),
            lambda **request: SimpleNamespace(status="completed", output_text="", output=[{"type": "refusal", "refusal": "refused"}]),
            lambda **request: SimpleNamespace(status="completed", output_text="not json", output=[]),
            invented,
        ]
        for index, responder in enumerate(scenarios):
            with self.subTest(index=index), self.api(create=responder) as (_, create):
                layer = self.layer(f"bad_{index}")
                actual = layer.hypotheses([cluster_payload()])
                layer.hypotheses([cluster_payload(1)])
            self.assertEqual(actual, expected)
            self.assertEqual(create.call_count, 1)
            self.assertNotIn("999999", actual[0])

    def test_missing_openai_library_uses_fallback(self):
        with self.api(), patch.object(llm_layer, "OpenAI", None):
            texts = self.layer().hypotheses([cluster_payload()])
        self.assert_hypothesis(texts[0])

    def test_node_card_preserves_gid_and_data_limitations(self):
        with self.api(key=False):
            layer = self.layer()
            seed = layer.node_card(node_payload())
            truncated = layer.node_card(node_payload(
                gid=OTHER_GID, role="peripheral", is_seed=False,
                depth=4, out_deg=0, out_kzt=0, truncated_by_depth=True,
            ))
            isolated_payload = node_payload(
                role="peripheral", in_deg=0, out_deg=0, in_kzt=0,
                out_kzt=0, in_tx=0, out_tx=0, pass_through=float("nan"),
            )
            isolated_payload["counterparties"] = {"incoming": [], "outgoing": []}
            isolated = layer.node_card(isolated_payload)
        self.assertIn(GID, seed)
        self.assertIn(OTHER_GID, seed)
        self.assertIn("coordinator", seed)
        self.assertIn("seed", seed.lower())
        self.assertRegex(seed.lower(), r"занижен|неполн|ограничен")
        self.assertRegex(truncated.lower(), r"обход|обрезан|глубин|границ")
        self.assertIn("0", isolated)
        self.assertNotIn("nan", isolated.lower())

    def test_node_uses_structured_output_and_caches(self):
        with self.api() as (_, create):
            first = self.layer().node_card(node_payload())
            second = self.layer().node_card(node_payload())
        self.assertEqual(create.call_count, 1)
        self.assertEqual(first, second)
        self.assertIn(GID, first)
        self.assertEqual(create.call_args.kwargs["text"]["format"]["name"], "aml_node")

    def test_other_valid_model_selection_changes_node_card(self):
        with self.api(create=lambda **kwargs: completed_response(kwargs)):
            first = self.layer("first").node_card(node_payload())
        with self.api(create=lambda **kwargs: completed_response(kwargs, choose_last=True)):
            second = self.layer("second").node_card(node_payload())
        self.assertNotEqual(first, second)

    def test_node_hints_batch_preserves_order_and_shares_node_card_cache(self):
        payloads = [node_payload(gid=str(int(GID) + index), is_seed=index == 0)
                    for index in range(25)]
        with self.api() as (_, create):
            layer = self.layer()
            hints = layer.node_hints(payloads)
            cards = [layer.node_card(payload) for payload in payloads]
        self.assertEqual(create.call_count, 1)
        self.assertEqual(len(hints), 25)
        self.assertNotEqual(hints[0]["attention"], hints[1]["attention"])
        for hint, card in zip(hints, cards):
            self.assertEqual(set(hint), {"attention", "next_request"})
            self.assertIn(hint["attention"], card)
            self.assertIn(hint["next_request"], card)
        self.assertEqual(create.call_args.kwargs["text"]["format"]["name"], "aml_node")

    def test_node_hints_offline_and_empty_batch(self):
        with self.api(key=False) as (constructor, create):
            layer = self.layer()
            self.assertEqual(layer.node_hints([]), [])
            hints = layer.node_hints([
                node_payload(), node_payload(is_seed=False, truncated_by_depth=True),
            ])
        self.assertEqual(hints[0]["next_request"], llm_layer.REQUESTS["incoming"])
        self.assertEqual(hints[1]["next_request"], llm_layer.REQUESTS["outgoing"])
        constructor.assert_not_called()
        create.assert_not_called()

    def test_node_payload_keeps_legacy_fields_exact_ids_and_counterparty_order(self):
        original = node_payload()["node"]
        peer_ids = [str(int(OTHER_GID) + index) for index in range(7)]
        nodes = pd.DataFrame([original] + [
            node_payload(gid=peer, role="transit")["node"] for peer in peer_ids
        ])
        edges = pd.DataFrame([
            {"src": int(peer), "dst": int(GID), "sum_kzt": 100.0, "n_tx": index + 1}
            for index, peer in reversed(list(enumerate(peer_ids)))
        ] + [{"src": int(GID), "dst": int(OTHER_GID), "sum_kzt": 90000.0, "n_tx": 5}])
        saved_nodes, saved_edges = nodes.copy(deep=True), edges.copy(deep=True)
        payload = llm_layer.node_payload(nodes, edges, int(GID))
        legacy_fields = [
            "gid", "role", "role_score", "cluster_id", "priority_score", "depth",
            "is_seed", "in_deg", "out_deg", "in_kzt", "out_kzt", "in_tx", "out_tx",
            "pass_through", "truncated_by_depth", "seed_payers", "role_rule",
        ]
        self.assertEqual(payload["node"], {field: original[field] for field in legacy_fields})
        self.assertEqual(payload["counterparties"]["incoming"], [
            {"gid": peer, "role": "transit", "sum_kzt": 100.0, "n_tx": index + 1}
            for index, peer in enumerate(peer_ids[:5])
        ])
        self.assertEqual(payload["counterparties"]["outgoing"], [
            {"gid": OTHER_GID, "role": "transit", "sum_kzt": 90000.0, "n_tx": 5}
        ])
        pd.testing.assert_frame_equal(nodes, saved_nodes)
        pd.testing.assert_frame_equal(edges, saved_edges)
        with self.assertRaisesRegex(ValueError, "не найден"):
            llm_layer.node_payload(nodes, edges, "missing")

    def test_short_timeout_and_budget_are_available_for_html_generation(self):
        with self.api() as (constructor, create), \
                patch.object(llm_layer, "monotonic", side_effect=[0.0, 2.0, 11.0]):
            layer = llm_layer.LLMLayer(self.root / "out", env_path=self.env_path,
                                       timeout_seconds=10.0, budget_seconds=10.0)
            hints = layer.node_hints([node_payload(gid=str(int(GID) + i)) for i in range(26)])
        self.assertEqual(len(hints), 26)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(constructor.call_args.kwargs["timeout"], 10.0)
        self.assertEqual(create.call_args.kwargs["timeout"], 8.0)

    def test_explain_node_reads_exact_large_ids_and_rejects_unknown(self):
        data = self.root / "data"
        out = self.root / "out"
        data.mkdir()
        out.mkdir()
        nodes = [node_payload()["node"], node_payload(gid=OTHER_GID, role="transit", is_seed=False)["node"]]
        pd.DataFrame(nodes).to_csv(out / "nodes_roles.csv", index=False)
        pd.DataFrame([
            {"src": int(GID), "dst": int(OTHER_GID), "sum_kzt": 90000.0, "n_tx": 5, "depth": 1},
            {"src": int(OTHER_GID), "dst": int(GID), "sum_kzt": 123456.78, "n_tx": 12, "depth": 1},
        ]).to_parquet(data / "edges.parquet", index=False)
        with self.api(key=False) as (_, create):
            first = llm_layer.explain_node(GID, data_dir=data, out_dir=out)
            same = llm_layer.explain_node(int(GID), data_dir=data, out_dir=out)
            with self.assertRaises((ValueError, KeyError)):
                llm_layer.explain_node("100000000343175102", data_dir=data, out_dir=out)
        self.assertEqual(first, same)
        self.assertIn(GID, first)
        self.assertIn(OTHER_GID, first)
        create.assert_not_called()


    def test_overall_budget_stops_new_requests_after_expiry(self):
        with self.api() as (_, create), patch.object(llm_layer, "monotonic", side_effect=[0.0, 0.0, 121.0]):
            texts = self.layer().hypotheses([cluster_payload(i) for i in range(26)])
        self.assertEqual(len(texts), 26)
        self.assertEqual(create.call_count, 1)
        self.assertLessEqual(create.call_args.kwargs["timeout"], 30.0)
        cache = json.loads((self.root / "out/llm_cache.json").read_text(encoding="utf-8"))
        self.assertEqual(sum(entry["source"] == "llm" for entry in cache["entries"].values()), 25)
        self.assertEqual(sum(entry["source"] == "fallback" for entry in cache["entries"].values()), 1)

    def test_invalid_cached_selection_is_revalidated(self):
        with self.api():
            self.layer().hypotheses([cluster_payload()])
        path = self.root / "out/llm_cache.json"
        cache = json.loads(path.read_text(encoding="utf-8"))
        next(iter(cache["entries"].values()))["selection"]["purpose"] = "invented_999999"
        path.write_text(json.dumps(cache), encoding="utf-8")
        with self.api() as (_, create):
            text = self.layer().hypotheses([cluster_payload()])[0]
        self.assertEqual(create.call_count, 1)
        self.assertNotIn("999999", text)

    def test_nonfinite_metrics_are_json_null_and_noise_reuses_cache(self):
        original = cluster_payload()
        original["top_nodes"][0]["hubs"] = float("nan")
        original["top_nodes"][0]["authorities"] = float("inf")
        changed = deepcopy(original)
        changed["top_nodes"][0]["priority_score"] += 1e-13
        with self.api() as (_, create):
            self.layer().hypotheses([original])
            self.layer().hypotheses([changed])
        self.assertEqual(create.call_count, 1)
        encoded = create.call_args.kwargs["input"]
        self.assertNotIn("NaN", encoded)
        self.assertNotIn("Infinity", encoded)
        supplied = json.loads(encoded)["item_0"]["data"]["top_nodes"][0]
        self.assertIsNone(supplied["hubs"])
        self.assertIsNone(supplied["authorities"])
        self.assertEqual(supplied["gid"], GID)
        self.assertIsInstance(supplied["gid"], str)

    def test_unwritable_cache_does_not_lose_result(self):
        with self.api() as (_, create), patch.object(llm_layer.os, "replace", side_effect=PermissionError("read only")):
            result = self.layer().hypotheses([cluster_payload()])
        self.assertEqual(create.call_count, 1)
        self.assert_hypothesis(result[0])

if __name__ == "__main__":
    unittest.main()
