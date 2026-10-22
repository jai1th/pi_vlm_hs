"""Contamination veto: decision node + spectral gate against a mocked AS7265x.

    python3 -m unittest discover -s tests -t . -v
"""

import ast
import contextlib
import datetime as dt
import io
import itertools
import json
import os
import tempfile
import unittest
from unittest import mock

import contamination_config as cfg
from actions import CLEAR, GREEN, ROUTING, STREAM_ANGLE, act_on_result, route
from servo import Servo, deg_to_us
from spectral_node import (SpectralError, SpectralNode, calibrate_and_save, expected_settings,
                           load_reference)
from storage import UTC_FORMAT

ALL_CHANNELS = "ABCDEFGHIJKLRSTUVW"
EXCLUDED = "KLUVW"
CLEAN_REF = 5388.5


class FakeSensor:
    """Returns the same level on the 13 retained channels and a wild value on
    K L U V W, so including them would wreck every level. Raw and calibrated
    counts are equal; with the bulb off it reads `dark` when one is given.

    readback_drop: fraction the lamp output falls when the registers are read
    back while lit (1.0 = the readback kills the bulb).
    sag_per_reading: fractional output loss per lit conversion (LED warming).
    """

    def __init__(self, level=5388.0, replicate_levels=None, settings=None, fail=False,
                 dark=None, readback_drop=0.0, sag_per_reading=0.0):
        self.level = level
        self.dark = dark
        self.readback_drop = readback_drop
        self.sag_per_reading = sag_per_reading
        self.replicate_levels = list(replicate_levels or [])
        self.settings = settings
        self.fail = fail
        self.white_on = False
        self.calls = []

    def enable_white(self):
        self.calls.append("enable_white")
        self.white_on = True

    def disable_bulbs(self):
        self.calls.append("disable_bulbs")
        self.white_on = False

    def read_settings(self):
        self.calls.append("read_settings")
        settings = self.settings or expected_settings(bulb_on=self.white_on)
        if self.white_on:
            self.level *= 1 - self.readback_drop   # registers still read "on"
        return settings

    def take_reading(self):
        self.calls.append("take_reading")
        if self.fail:
            raise TimeoutError("AS7265x data not ready")
        if not self.white_on and self.dark is not None:
            level = self.dark
        elif self.replicate_levels:
            level = self.replicate_levels.pop(0)
        else:
            level = self.level
            self.level *= 1 - self.sag_per_reading
        calibrated = {ch: (50000.0 if ch in EXCLUDED else level) for ch in ALL_CHANNELS}
        return dict(calibrated), calibrated, {}


class FakeServo:
    def __init__(self):
        self.angles = []

    def swing(self, angle):
        self.angles.append(angle)


def reference(hours_old=0.0, **overrides):
    taken = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours_old)
    ref = {
        "substrate": "carton",
        "clean_ref_level": CLEAN_REF,
        "timestamp": taken.strftime(UTC_FORMAT),
        "gain": cfg.GAIN,
        "integration_cycles": cfg.INTEGRATION_CYCLES,
        "bulb_current": cfg.WHITE_BULB_CURRENT,
    }
    ref.update(overrides)
    return ref


class VetoTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def node(self, sensor, ref="default", clock=None):
        return SpectralNode(
            sensor, reference() if ref == "default" else ref,
            log_path=os.path.join(self.tmp, "spectral.jsonl"),
            sleep=lambda s: None, clock=clock or itertools.count().__next__)

    def decide(self, labels, sensor=None, node=None):
        """Run one frame through act_on_result; return (stream, reason, servo angles)."""
        sensor = sensor or FakeSensor()
        node = node or self.node(sensor)
        servo = FakeServo()
        result = {"request_id": "cap-test", "objects": [
            {"label": label, "bbox_canvas": [0, 0, 10, 10], "bbox_source_px": [0, 0, 10, 10]}
            for label in labels]}
        with contextlib.redirect_stdout(io.StringIO()):
            stream, reason = act_on_result(result, node, servo,
                                           log_path=os.path.join(self.tmp, "decisions.jsonl"))
        return stream, reason, servo.angles


class SpectralGateTests(VetoTestCase):
    def test_clean_carton_routes_green(self):
        sensor = FakeSensor(level=5388)
        self.assertEqual(self.decide(["togo_box"], sensor),
                         (GREEN, "class_recyclable_clean", [STREAM_ANGLE[GREEN]]))
        self.assertEqual(sensor.calls.count("take_reading"), cfg.REPLICATES)

    def test_darkened_carton_is_vetoed(self):
        # d = +0.187: a wetting film darkens the surface.
        self.assertEqual(self.decide(["togo_box"], FakeSensor(level=4380))[:2],
                         (CLEAR, "contamination_veto"))

    def test_pooled_spill_brighter_than_clean_is_vetoed(self):
        # d = -0.246: a pooled spill is specular and reads BRIGHTER than clean.
        # A one-sided veto passes every other test in this file and fails this one.
        reading = self.node(FakeSensor(level=6712)).read_contamination("carton")
        self.assertLess(reading.d, -cfg.CONTAM_MODEL["carton"][1])
        self.assertIs(reading.state, True)
        self.assertEqual(self.decide(["togo_box"], FakeSensor(level=6712))[:2],
                         (CLEAR, "contamination_veto"))

    def test_undetermined_band_routes_clear(self):
        # d = +0.053, between undetermined (0.04) and veto (0.08).
        reading = self.node(FakeSensor(level=5100)).read_contamination("carton")
        self.assertIsNone(reading.state)
        self.assertEqual(self.decide(["togo_box"], FakeSensor(level=5100))[:2],
                         (CLEAR, "contamination_undetermined"))

    def test_dead_bulb_is_illumination_failure_not_veto(self):
        # A near-dark read would otherwise be a confident contamination_veto (d ~ +0.9).
        reading = self.node(FakeSensor(level=500)).read_contamination("carton")
        self.assertEqual(reading.reason, "illumination_failure")
        self.assertIsNone(reading.state)
        self.assertEqual(self.decide(["togo_box"], FakeSensor(level=500))[:2],
                         (CLEAR, "illumination_failure"))

    def test_noisy_dark_counts_are_illumination_failure_not_unstable(self):
        reading = self.node(FakeSensor(replicate_levels=[9, 12, 10])).read_contamination("carton")
        self.assertEqual(reading.reason, "illumination_failure")

    def test_sensor_failure_retries_once_then_sensor_timeout(self):
        sensor = FakeSensor(fail=True)
        self.assertEqual(self.decide(["togo_box"], sensor)[:2], (CLEAR, "sensor_timeout"))
        self.assertEqual(sensor.calls.count("take_reading"), 1 + cfg.SENSOR_RETRIES)
        self.assertFalse(sensor.white_on)

    def test_read_budget_exceeded_is_sensor_timeout(self):
        sensor = FakeSensor()
        slow_clock = itertools.count(0, 5).__next__   # every clock read is 5 s later
        node = self.node(sensor, clock=slow_clock)
        self.assertEqual(self.decide(["togo_box"], node=node)[:2], (CLEAR, "sensor_timeout"))

    def test_retry_does_not_start_after_the_deadline(self):
        # One deadline for read + retry: the first attempt exhausts the budget,
        # so the retry never turns the bulb on.
        sensor = FakeSensor()
        node = self.node(sensor, clock=itertools.count(0, 5).__next__)
        node.read_contamination("carton")
        self.assertEqual(sensor.calls.count("enable_white"), 1)

    def test_unstable_replicates_route_clear_without_retry(self):
        sensor = FakeSensor(replicate_levels=[5388, 5500, 5388])
        self.assertEqual(self.decide(["togo_box"], sensor)[:2], (CLEAR, "replicate_unstable"))
        self.assertEqual(sensor.calls.count("enable_white"), 1)

    def test_instrument_readback_mismatch(self):
        wrong = dict(expected_settings(bulb_on=True), integration_cycles=50)
        sensor = FakeSensor(settings=wrong)
        self.assertEqual(self.decide(["togo_box"], sensor)[:2],
                         (CLEAR, "instrument_config_mismatch"))
        self.assertNotIn("take_reading", sensor.calls)

    def test_no_session_reference_does_not_touch_sensor(self):
        sensor = FakeSensor()
        node = self.node(sensor, ref=None)
        self.assertEqual(self.decide(["togo_box"], node=node)[:2], (CLEAR, "no_session_reference"))
        self.assertEqual(sensor.calls, [])

    def test_no_spectral_matches_missing_reference(self):
        node = SpectralNode(log_path=os.path.join(self.tmp, "spectral.jsonl"))
        self.assertEqual(self.decide(["togo_box"], node=node)[:2], (CLEAR, "no_session_reference"))

    def test_raw_counts_always_logged(self):
        self.decide(["togo_box"], FakeSensor(fail=True))
        with open(os.path.join(self.tmp, "spectral.jsonl")) as f:
            records = [json.loads(line) for line in f]
        failures = [r for r in records if r.get("event") == "sensor_failure"]
        self.assertEqual(len(failures), 1 + cfg.SENSOR_RETRIES)
        self.assertTrue(all("raw" in r for r in records))

        self.decide(["togo_box"], FakeSensor(level=5388))
        with open(os.path.join(self.tmp, "spectral.jsonl")) as f:
            last = json.loads(f.readlines()[-1])
        self.assertEqual(len(last["raw"]), cfg.REPLICATES)
        self.assertEqual(sorted(last["raw"][0]), sorted(ALL_CHANNELS))


class ClassRoutingTests(VetoTestCase):
    def test_every_clear_class_skips_the_sensor(self):
        for label in [l for l, s in ROUTING.items() if s == CLEAR]:
            with self.subTest(label=label):
                sensor = FakeSensor()
                self.assertEqual(self.decide([label], sensor),
                                 (CLEAR, "class_landfill", [STREAM_ANGLE[CLEAR]]))
                self.assertEqual(sensor.calls, [])

    def test_togo_cup_skips_the_sensor(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["togo_cup"], sensor)[:2], (CLEAR, "class_landfill"))
        self.assertEqual(sensor.calls, [])

    def test_label_conflict_resolves_to_clear_without_read(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["togo_box", "cup_lid"], sensor)[:2], (CLEAR, "class_landfill"))
        self.assertEqual(sensor.calls, [])

    def test_duplicate_boxes_for_one_object_read_once(self):
        sensor = FakeSensor(level=5388)
        self.assertEqual(self.decide(["togo_box", "togo_box"], sensor)[:2],
                         (GREEN, "class_recyclable_clean"))
        self.assertEqual(sensor.calls.count("enable_white"), 1)

    def test_no_detection_does_not_actuate(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide([], sensor), (CLEAR, "no_detection", []))
        self.assertEqual(sensor.calls, [])
        with open(os.path.join(self.tmp, "decisions.jsonl")) as f:
            self.assertEqual(json.loads(f.readline())["n_detections"], 0)

    def test_capture_item_type_is_unknown_label(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["paper_hot_cup"], sensor)[:2], (CLEAR, "unknown_label"))
        self.assertEqual(sensor.calls, [])

    def test_paper_has_no_spectral_model(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["paper"], sensor)[:2], (CLEAR, "substrate_uncharacterized"))
        self.assertEqual(sensor.calls, [])

    def test_plastic_bottle_gate_not_implemented(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["plastic_bottle"], sensor)[:2], (CLEAR, "gate_not_implemented"))
        self.assertEqual(sensor.calls, [])

    def test_green_labels_with_different_substrates_are_uncharacterized(self):
        sensor = FakeSensor()
        self.assertEqual(self.decide(["togo_box", "paper"], sensor)[:2],
                         (CLEAR, "substrate_uncharacterized"))
        self.assertEqual(sensor.calls, [])

    def test_route_refuses_to_guess_without_a_reading(self):
        with self.assertRaises(ValueError):
            route([{"label": "togo_box"}], None)


class CheckSpectralTests(VetoTestCase):
    def check(self, sensor, answer=""):
        node = self.node(sensor)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = node.check_bulb_readback(prompt=lambda _: answer)
        self.output = out.getvalue()
        return ok

    def test_healthy_readback_passes(self):
        self.assertTrue(self.check(FakeSensor(dark=4.5)))

    def test_readback_that_kills_the_bulb_fails(self):
        self.assertFalse(self.check(FakeSensor(readback_drop=1.0)))

    def test_thermal_sag_alone_passes(self):
        # ~0.9 % sag between legs: would fail a REPLICATE_CV_MAX (0.5 %) mark,
        # but the readback leg sags no more than the control leg.
        self.assertTrue(self.check(FakeSensor(sag_per_reading=0.003)))
        self.assertIn("thermal_drift", self.output)

    def test_readback_drop_beyond_thermal_fails(self):
        # 3 % drop at the readback against no thermal drift: under the 5 %
        # outright mark, but more than thermal_drift + 2 %.
        self.assertFalse(self.check(FakeSensor(readback_drop=0.03)))

    def test_numbers_printed_on_pass_and_fail(self):
        for sensor in (FakeSensor(), FakeSensor(readback_drop=1.0)):
            self.check(sensor)
            self.assertIn("thermal_drift", self.output)
            self.assertIn("readback_delta", self.output)

    def test_cal_raw_table(self):
        self.check(FakeSensor(dark=4.5))
        for row in ("dark", "white", "sample"):
            self.assertRegex(self.output, r"\n  {}\s+raw=".format(row))
        self.check(FakeSensor(dark=4.5), answer="s")
        self.assertNotRegex(self.output, r"\n  sample\s+raw=")

    def test_wrong_registers_fail(self):
        self.assertFalse(self.check(FakeSensor(settings=dict(expected_settings(True), gain=2))))


class CalibrationTests(VetoTestCase):
    def calibrate(self, sensor):
        path = os.path.join(self.tmp, "session_reference.json")
        node = self.node(sensor, ref=None)
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return calibrate_and_save(node, "carton", path, prompt=lambda _: ""), path
            except SpectralError as exc:
                return exc, path

    def test_lit_white_tile_calibrates(self):
        # dark raw 4.5, white raw 23331 -> ratio 5184
        record, path = self.calibrate(FakeSensor(level=23331, dark=4.5))
        self.assertIsInstance(record, dict)
        with open(path) as f:
            saved = json.load(f)
        self.assertAlmostEqual(saved["white_raw_level"] / saved["dark_raw_level"], 5184.67, places=1)
        for key in ("dark_raw_level", "white_raw_level", "white_flatness"):
            self.assertIn(key, saved)

    def test_single_placement_calibrates(self):
        with mock.patch.object(cfg, "CAL_SEATINGS", 1):
            record, _ = self.calibrate(FakeSensor(level=5600, dark=5.7))
        self.assertEqual(record["seating_levels"], [5600])
        self.assertIsNone(record["seating_cv"])

    def test_dead_bulb_aborts_calibration(self):
        # dark raw 10, white raw 12 -> ratio 1.2
        error, path = self.calibrate(FakeSensor(level=12, dark=10))
        self.assertIsInstance(error, SpectralError)
        self.assertIn("1.2x", str(error))
        self.assertFalse(os.path.exists(path))


class FakePi:
    connected = True

    def __init__(self):
        self.pulses = []

    def set_servo_pulsewidth(self, pin, us):
        self.pulses.append(us)


@mock.patch("servo.time.sleep", lambda s: None)
class DiverterTests(unittest.TestCase):
    def servo(self):
        pi = FakePi()
        servo = Servo(pi=pi)
        servo.open()
        return servo, pi

    def test_directions(self):
        self.assertEqual(STREAM_ANGLE, {CLEAR: -90, GREEN: +90})

    def test_rests_at_zero_on_open(self):
        _, pi = self.servo()
        self.assertEqual(pi.pulses, [deg_to_us(0)])

    def test_both_streams_swing_and_return_home(self):
        for stream in (CLEAR, GREEN):
            with self.subTest(stream=stream):
                servo, pi = self.servo()
                servo.swing(STREAM_ANGLE[stream])
                self.assertIn(deg_to_us(STREAM_ANGLE[stream]), pi.pulses)
                self.assertEqual(pi.pulses[-1], deg_to_us(0))


class SessionReferenceTests(VetoTestCase):
    def write(self, ref):
        path = os.path.join(self.tmp, "session_reference.json")
        with open(path, "w") as f:
            json.dump(ref, f)
        return path

    def live(self):
        return expected_settings(bulb_on=False)

    def test_fresh_reference_loads(self):
        ref, why = load_reference(self.write(reference(hours_old=1)), self.live())
        self.assertIsNone(why)
        self.assertEqual(ref["clean_ref_level"], CLEAN_REF)

    def test_missing_reference(self):
        ref, why = load_reference(os.path.join(self.tmp, "absent.json"), self.live())
        self.assertIsNone(ref)
        self.assertIn("--calibrate", why)

    def test_stale_reference(self):
        ref, _ = load_reference(self.write(reference(hours_old=cfg.REFERENCE_MAX_AGE_H + 1)), self.live())
        self.assertIsNone(ref)

    def test_reference_recorded_at_other_settings(self):
        ref, _ = load_reference(self.write(reference(integration_cycles=50)), self.live())
        self.assertIsNone(ref)

    def test_live_readback_differs_from_reference(self):
        live = dict(self.live(), gain=2)
        self.assertIsNone(load_reference(self.write(reference()), live)[0])
        self.assertIsNone(load_reference(self.write(reference()), None)[0])


class BoundaryTests(unittest.TestCase):
    def test_spectral_node_never_imports_routing(self):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "spectral_node.py")
        with open(path) as f:
            tree = ast.parse(f.read())
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                imported.update(a.name for a in n.names)
            elif isinstance(n, ast.ImportFrom):
                imported.add(n.module)
        self.assertNotIn("actions", imported)

    def test_spectral_reasons_are_never_streams(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "spectral.jsonl")
            for sensor in (FakeSensor(5388), FakeSensor(4380), FakeSensor(fail=True)):
                node = SpectralNode(sensor, reference(), log_path=log, sleep=lambda s: None)
                self.assertNotIn(node.read_contamination("carton").reason, (GREEN, CLEAR))


if __name__ == "__main__":
    unittest.main()
