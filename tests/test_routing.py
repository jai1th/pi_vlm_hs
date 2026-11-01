"""Editable item -> bag mapping: the file, the decision node and the dashboard tab.

    python3 -m unittest discover -s tests -t . -v
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import routing
from actions import STREAM_ANGLE, act_on_result, route
from routing import CLEAR, DEFAULT_ROUTING, GREEN, LABELS
from tests.test_veto import FakeSensor, VetoTestCase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RoutingFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self._tmp.name, "routing.json")

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, raw):
        with open(self.path, "w") as f:
            f.write(raw if isinstance(raw, str) else json.dumps(raw))

    def test_no_file_is_the_built_in_mapping(self):
        self.assertEqual(routing.load(self.path), (DEFAULT_ROUTING, None))

    def test_round_trip(self):
        edited = DEFAULT_ROUTING.with_edit("tissue", bag=GREEN).with_edit("tissue", unchecked=True)
        saved = routing.save(edited, self.path)
        loaded, problem = routing.load(self.path)
        self.assertIsNone(problem)
        self.assertEqual(loaded.streams["tissue"], GREEN)
        self.assertEqual(loaded.unchecked, {"tissue"})
        self.assertEqual(loaded.digest, edited.digest)
        self.assertIsNotNone(saved.updated)
        self.assertFalse(os.path.exists(self.path + ".tmp"))

    def test_missing_items_take_their_defaults(self):
        self.write({"schema_version": 1, "items": {"straw": {"bag": "green"}}})
        loaded, problem = routing.load(self.path)
        self.assertIsNone(problem)
        self.assertEqual(loaded.streams["straw"], GREEN)
        self.assertEqual(loaded.streams["togo_box"], GREEN)
        self.assertEqual(loaded.streams["wrapper"], CLEAR)

    def test_invalid_files_fall_back_with_a_reason(self):
        for raw, why in (
                ("{not json", "unreadable"),
                ({"schema_version": 2, "items": {}}, "schema_version"),
                ({"schema_version": 1, "items": {"banana": {"bag": "green"}}}, "unknown item"),
                ({"schema_version": 1, "items": {"straw": {"bag": "blue"}}}, "'bag'"),
                ({"schema_version": 1, "items": {"straw": {"bag": "green",
                                                           "skip_contamination_check": "yes"}}},
                 "true or false")):
            with self.subTest(why=why):
                self.write(raw)
                loaded, problem = routing.load(self.path)
                self.assertIs(loaded, DEFAULT_ROUTING)
                self.assertIn(why, problem)

    def test_a_clear_item_never_skips_the_check(self):
        edited = DEFAULT_ROUTING.with_edit("straw", unchecked=True)
        self.assertEqual(edited.unchecked, frozenset())
        bypassed = DEFAULT_ROUTING.with_edit("paper", unchecked=True)
        self.assertEqual(bypassed.with_edit("paper", bag=CLEAR).unchecked, frozenset())

    def test_digest_tracks_what_routes_where(self):
        self.assertEqual(DEFAULT_ROUTING.digest, routing.Routing(dict(routing.DEFAULT_STREAMS)).digest)
        self.assertNotEqual(DEFAULT_ROUTING.digest, DEFAULT_ROUTING.with_edit("straw", bag=GREEN).digest)
        self.assertNotEqual(DEFAULT_ROUTING.digest,
                            DEFAULT_ROUTING.with_edit("paper", unchecked=True).digest)

    def test_outcomes(self):
        r = DEFAULT_ROUTING.with_edit("tissue", bag=GREEN).with_edit("paper", unchecked=True)
        self.assertIn("carton spectral check", routing.outcome(r, "togo_box"))
        self.assertIn("not built", routing.outcome(r, "plastic_bottle"))
        self.assertIn("no spectral model", routing.outcome(r, "tissue"))
        self.assertIn("label alone", routing.outcome(r, "paper"))
        self.assertEqual(routing.outcome(r, "straw"), "CLEAR")

    def test_changes_are_listed(self):
        r = DEFAULT_ROUTING.with_edit("tissue", bag=GREEN).with_edit("paper", unchecked=True)
        self.assertEqual(routing.changes(DEFAULT_ROUTING, r),
                         ["paper: contamination check skipped", "tissue: CLEAR → GREEN"])
        self.assertEqual(routing.changes(r, r), [])


class RoutedDecisionTests(VetoTestCase):
    def decide_with(self, labels, mapping, sensor=None):
        sensor = sensor or FakeSensor(level=5388)
        node = self.node(sensor)
        result = {"request_id": "cap-test", "objects": [
            {"label": l, "bbox_canvas": [0, 0, 1, 1], "bbox_source_px": [0, 0, 1, 1]} for l in labels]}
        log = os.path.join(self.tmp, "decisions.jsonl")
        with contextlib.redirect_stdout(io.StringIO()):
            stream, reason = act_on_result(result, node, None, log_path=log, routing=mapping)
        with open(log) as f:
            record = json.loads(f.readlines()[-1])
        return stream, reason, sensor, record

    def test_moving_an_item_to_clear(self):
        stream, reason, sensor, _ = self.decide_with(["togo_box"],
                                                     DEFAULT_ROUTING.with_edit("togo_box", bag=CLEAR))
        self.assertEqual((stream, reason), (CLEAR, "class_landfill"))
        self.assertEqual(sensor.calls, [])

    def test_green_without_a_model_still_needs_the_check(self):
        stream, reason, sensor, _ = self.decide_with(["tissue"],
                                                     DEFAULT_ROUTING.with_edit("tissue", bag=GREEN))
        self.assertEqual((stream, reason), (CLEAR, "substrate_uncharacterized"))

    def test_unchecked_item_goes_green_on_the_label(self):
        mapping = DEFAULT_ROUTING.with_edit("tissue", bag=GREEN).with_edit("tissue", unchecked=True)
        stream, reason, sensor, record = self.decide_with(["tissue"], mapping)
        self.assertEqual((stream, reason), (GREEN, "class_recyclable_unchecked"))
        self.assertEqual(sensor.calls, [])
        self.assertEqual(record["angle"], STREAM_ANGLE[GREEN])
        self.assertEqual(record["routing_digest"], mapping.digest)

    def test_unchecked_bottle_skips_the_missing_fill_check(self):
        mapping = DEFAULT_ROUTING.with_edit("plastic_bottle", unchecked=True)
        self.assertEqual(route([{"label": "plastic_bottle"}], None, mapping),
                         (GREEN, "class_recyclable_unchecked"))

    def test_bypass_needs_every_label_in_the_frame(self):
        # paper unchecked, togo_box checked: the frame keeps the gate (and the
        # labels disagree on substrate, so it is uncharacterized -> CLEAR).
        mapping = DEFAULT_ROUTING.with_edit("paper", unchecked=True)
        stream, reason, _, _ = self.decide_with(["togo_box", "paper"], mapping)
        self.assertEqual((stream, reason), (CLEAR, "substrate_uncharacterized"))

    def test_clear_label_still_wins_over_a_bypass(self):
        mapping = DEFAULT_ROUTING.with_edit("paper", unchecked=True)
        self.assertEqual(route([{"label": "paper"}, {"label": "straw"}], None, mapping),
                         (CLEAR, "class_landfill"))

    def test_default_mapping_logs_its_digest(self):
        _, _, _, record = self.decide_with(["straw"], DEFAULT_ROUTING)
        self.assertEqual(record["routing_digest"], DEFAULT_ROUTING.digest)


class StartupTests(unittest.TestCase):
    def test_open_routing_prints_and_warns(self):
        import main
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "routing.json")
            routing.save(DEFAULT_ROUTING.with_edit("paper", unchecked=True), path)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                loaded = main.open_routing(path)
            self.assertEqual(loaded.unchecked, {"paper"})
            self.assertIn("paper", out.getvalue())
            self.assertIn("no contamination check for: paper", err.getvalue())

            with open(path, "w") as f:
                f.write("{broken")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                self.assertIs(main.open_routing(path), DEFAULT_ROUTING)
            self.assertIn("built-in mapping instead", err.getvalue())


try:
    from streamlit.testing.v1 import AppTest
except ImportError:
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit not installed")
class RoutingTabTests(unittest.TestCase):
    def render(self, run_dir):
        # Patched for the whole test: every rerun (a click) re-reads the run directory.
        env = mock.patch.dict(os.environ, {"WASTESEP_RUN_DIR": run_dir})
        env.start()
        self.addCleanup(env.stop)
        app = AppTest.from_file(os.path.join(ROOT, "dashboard.py"), default_timeout=60)
        app.run()
        return app

    def test_reset_and_save_writes_the_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "routing.json")
            routing.save(DEFAULT_ROUTING.with_edit("paper", unchecked=True), path)
            app = self.render(tmp)
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any("No contamination check for: paper" in w.value for w in app.warning))
            self.assertTrue(app.button(key="routing_save").disabled)   # nothing to save yet

            app.button(key="routing_reset").click().run()
            self.assertTrue(any("paper: contamination check required again" in m.value
                                for m in app.markdown))
            app.button(key="routing_save").click().run()
            self.assertEqual(list(app.exception), [])
            self.assertEqual(routing.load(path)[0].digest, DEFAULT_ROUTING.digest)
            self.assertTrue(any("Restart main.py" in s.value for s in app.success))

    def test_broken_file_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "routing.json"), "w") as f:
                f.write("{broken")
            app = self.render(tmp)
        self.assertEqual(list(app.exception), [])
        self.assertTrue(any("falls back to the built-in mapping" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
