"""AS7265x contamination gate (veto_integration_spec.md §1, §3).

The gate answers one question for one substrate: contaminated, clean, or
undetermined. It never routes: nothing here imports actions.py, and no
SpectralReading carries a stream name.

Hardware goes through spectral_collector/referencer.py's RealSensor, the driver
path already proven on this fixture (one-shot trigger + DATA_RDY poll, not
take_measurements(), which can hand back stale registers). This module adds the
settings readback and the read / calibration sequences on top of it.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import statistics
import sys
import time
from dataclasses import dataclass

import contamination_config as cfg
from storage import UTC_FORMAT, append_jsonl, utc_timestamp


@dataclass(frozen=True)
class SpectralReading:
    state: bool | None        # True=contaminated, False=clean, None=undetermined
    d: float | None           # (clean_ref - live) / clean_ref
    level: float | None       # mean cal over the 13 retained channels
    reason: str               # always populated
    raw: dict | None          # all 18 raw channel counts (one list entry per replicate)


class SpectralError(Exception):
    pass


def expected_settings(bulb_on):
    """What read_settings() must return: locked gain/cycles, white bulb only."""
    return {
        "gain": cfg.GAIN,
        "integration_cycles": cfg.INTEGRATION_CYCLES,
        "white_current": cfg.WHITE_BULB_CURRENT,
        "white_on": bulb_on,
        "ir_on": False,
        "uv_on": False,
    }


# --- Hardware ----------------------------------------------------------------------

# AS7265x virtual registers (datasheet; the same ones the SparkFun setters write).
_REG_CONFIG = 0x04            # bits 5:4 gain
_REG_INTEGRATION_TIME = 0x05  # integration cycles
_REG_LED_CONFIG = 0x07        # per device: bit 3 bulb on, bits 5:4 bulb current


class As7265xSensor:
    """referencer.RealSensor plus the settings readback the veto needs.

    TODO(bench): read_settings() reads virtual registers directly. It passes the
    mocked tests by construction; verify it on the Pi before the 28th.
    """

    def __init__(self, collector_dir=cfg.SPECTRAL_COLLECTOR_DIR):
        self._collector_dir = collector_dir
        self._ref = None
        self._real = None

    def open(self):
        """Connect, disable the indicator, lock gain/cycles/bulb current, bulbs off."""
        try:
            if self._collector_dir not in sys.path:
                sys.path.append(self._collector_dir)
            import referencer
            self._ref = referencer
            self._real = referencer.RealSensor({
                "i2c_address": cfg.I2C_ADDRESS,
                "gain": cfg.GAIN,
                "integration_cycles": cfg.INTEGRATION_CYCLES,
                "bulb_current_white": cfg.WHITE_BULB_CURRENT,
                "bulb_current_ir": cfg.WHITE_BULB_CURRENT,
                "bulb_current_uv": cfg.WHITE_BULB_CURRENT,
                "disable_indicator": True,
                "data_ready_timeout_s": cfg.CONVERSION_TIMEOUT_S,
            }, logging.getLogger("spectral_node"))
            self._real.initialize()
            self._real.configure()
            self._real.disable_all_bulbs()
        except Exception as exc:
            raise SpectralError("could not start AS7265x: {}".format(exc)) from exc

    def enable_white(self):
        self._real.enable_bulbs({"white": True})

    def disable_bulbs(self):
        self._real.disable_all_bulbs()

    def take_reading(self):
        """One conversion -> (raw, calibrated, ic_temperature_c), keyed by channel letter."""
        _, raw, calibrated, temperatures = self._real.take_reading()
        return raw, calibrated, temperatures

    def read_settings(self):
        """Read gain, integration cycles and bulb state back from the device.

        Returns None when the installed library has no virtual_read_register();
        the caller treats that as a mismatch, since unverifiable is not verified.
        """
        s = self._real._sensor  # RealSensor has no readback of its own
        if not hasattr(s, "virtual_read_register"):
            return None
        config = s.virtual_read_register(_REG_CONFIG)
        cycles = s.virtual_read_register(_REG_INTEGRATION_TIME)
        led = {}
        for name, device in self._ref.BULB_DEVICE.items():
            s.select_device(device)
            led[name] = s.virtual_read_register(_REG_LED_CONFIG)
        s.select_device(self._ref.MASTER_DEVICE)
        return {
            "gain": (config >> 4) & 0b11,
            "integration_cycles": cycles,
            "white_current": (led["white"] >> 4) & 0b11,
            "white_on": bool(led["white"] & 0b1000),
            "ir_on": bool(led["ir"] & 0b1000),
            "uv_on": bool(led["uv"] & 0b1000),
        }

    def close(self):
        if self._real is None or self._real._sensor is None:   # never connected
            self._real = None
            return
        try:
            self._real.disable_all_bulbs()
        except Exception as exc:
            print("warning: could not turn the AS7265x bulbs off: {}".format(exc), file=sys.stderr)
        finally:
            self._real.try_disable_indicator()
            self._real = None


# --- Derived quantities ------------------------------------------------------------------

def level_of(calibrated):
    """Mean over the retained channels of the per-channel replicate means."""
    return statistics.mean(
        statistics.mean(rep[ch] for rep in calibrated) for ch in cfg.LEVEL_CHANNELS)


def max_cv(calibrated):
    """Largest per-channel coefficient of variation across replicates."""
    worst = 0.0
    for ch in cfg.LEVEL_CHANNELS:
        values = [rep[ch] for rep in calibrated]
        mean = statistics.mean(values)
        if mean <= 0:
            return math.inf
        # Sample stdev (ddof=1): every threshold in the config was computed that way.
        worst = max(worst, statistics.stdev(values) / mean)
    return worst


def evaluate(raw, calibrated, clean_ref, veto, undetermined):
    """Two-sided tri-state on d. A thin wetting film darkens the surface
    (d > 0); a pooled spill is specular and reads brighter than clean
    (measured d = -0.246). A one-sided test would pass the spill as clean.

    The illumination floor comes first, before the CV check too: near-dark
    counts are noisy enough to fail CV, which would misreport a dead bulb as
    an object that moved."""
    raw_by_channel = {ch: [rep[ch] for rep in raw] for ch in raw[0]}
    level = level_of(calibrated)
    d = (clean_ref - level) / clean_ref
    if level < cfg.ILLUMINATION_FLOOR * clean_ref:
        return SpectralReading(None, d, level, "illumination_failure", raw_by_channel)
    if max_cv(calibrated) > cfg.REPLICATE_CV_MAX:
        return SpectralReading(None, d, level, "replicate_unstable", raw_by_channel)
    if abs(d) >= veto:
        return SpectralReading(True, d, level, "contaminated", raw_by_channel)
    if abs(d) >= undetermined:
        return SpectralReading(None, d, level, "contamination_undetermined", raw_by_channel)
    return SpectralReading(False, d, level, "clean", raw_by_channel)


def _no_reading(reason):
    return SpectralReading(None, None, None, reason, None)


def _new_acquisition():
    return {"settings": None, "raw": [], "calibrated": [], "ic_temperature_c": []}


# --- Node ------------------------------------------------------------------------------------

class SpectralNode:
    """Owns the AS7265x and the session reference for one run.

    sensor=None (--no-spectral) behaves exactly like a missing session
    reference: every substrate with a model reads as no_session_reference.
    """

    def __init__(self, sensor=None, reference=None, log_path=cfg.SPECTRAL_LOG_PATH,
                 sleep=time.sleep, clock=time.monotonic):
        self.sensor = sensor
        self.reference = reference if sensor is not None else None
        self.log_path = log_path
        self._sleep = sleep
        self._clock = clock

    def read_contamination(self, substrate, *, budget_s=cfg.READ_BUDGET_S, request_id=None):
        """Contamination tri-state for `substrate`. Never raises.

        One deadline covers the read and its retry: a retry starts only while
        budget remains, so the worst case is about budget_s plus one conversion.
        """
        deadline = self._clock() + budget_s
        model = cfg.CONTAM_MODEL.get(substrate)
        if model is None:
            return self._finish(request_id, substrate, None, _no_reading("substrate_uncharacterized"))
        if self.reference is None or self.reference["substrate"] != substrate:
            return self._finish(request_id, substrate, None, _no_reading("no_session_reference"))
        # The config clean_ref_level is superseded by the session measurement.
        _, veto, undetermined = model
        clean_ref = self.reference["clean_ref_level"]

        for attempt in range(1 + cfg.SENSOR_RETRIES):
            if self._clock() >= deadline:
                break
            acq = _new_acquisition()
            try:
                self._acquire(acq, deadline, bulb_on=True, replicates=cfg.REPLICATES)
            except Exception as exc:
                self._log(request_id, substrate, acq, event="sensor_failure",
                          attempt=attempt, error=repr(exc))
                continue
            if acq["settings"] != expected_settings(bulb_on=True):
                return self._finish(request_id, substrate, acq,
                                    _no_reading("instrument_config_mismatch"), attempt=attempt)
            reading = evaluate(acq["raw"], acq["calibrated"], clean_ref, veto, undetermined)
            return self._finish(request_id, substrate, acq, reading, attempt=attempt)
        return self._finish(request_id, substrate, None, _no_reading("sensor_timeout"))

    def _acquire(self, acq, deadline, bulb_on, replicates):
        """Readback, then replicates, under one bulb window. The readback runs
        before any measurement, never between replicates, because of the
        bulb-switching defect. Fills `acq` as it goes so a failure part way
        still leaves the raw counts for the log. Stops early, without
        replicates, when the readback does not match."""
        try:
            if bulb_on:
                self.sensor.enable_white()
            else:
                self.sensor.disable_bulbs()
            self._sleep(cfg.BULB_SETTLE_S)
            acq["settings"] = self.sensor.read_settings()
            if acq["settings"] != expected_settings(bulb_on):
                return
            self._replicates(acq, replicates, deadline)
            if self._clock() > deadline:
                raise TimeoutError("read budget exceeded")
        finally:
            if bulb_on:
                self.sensor.disable_bulbs()

    def _replicates(self, acq, n, deadline=None):
        """Take n conversions into `acq`; returns their (raw, calibrated) counts."""
        start = len(acq["calibrated"])
        for _ in range(n):
            if deadline is not None and self._clock() >= deadline:
                raise TimeoutError("read budget exceeded")
            raw, calibrated, temperatures = self.sensor.take_reading()
            acq["raw"].append(raw)
            acq["calibrated"].append(calibrated)
            acq["ic_temperature_c"].append(temperatures)
        return acq["raw"][start:], acq["calibrated"][start:]

    def check_bulb_readback(self, prompt=input):
        """Bench test for the register readback (main.py --check-spectral).

        The readback switches devices while the white bulb is lit, and
        referencer.py records a bulb-state defect in device-switching
        sequences. So the measured quantity is the bulb surviving it, judged
        against a thermal control leg with no readback:

            bulb on, settle; L1; L1' (control); readback; L2; bulb off; readback

        The register values are compared with the bulb on and off, and a
        cal/raw table (dark, white tile, optional clean sample) is printed as a
        diagnostic. Returns True when the bulb and register checks pass.
        """
        prompt("Seat the PTFE tile, then press Enter. ")
        acq = _new_acquisition()
        try:
            self.sensor.enable_white()
            self._sleep(cfg.BULB_SETTLE_S)
            white = self._replicates(acq, cfg.REPLICATES)
            l1 = level_of(white[1])
            l1_control = level_of(self._replicates(acq, cfg.REPLICATES)[1])
            lit = self.sensor.read_settings()          # the suspect operation
            l2 = level_of(self._replicates(acq, cfg.REPLICATES)[1])
        finally:
            self.sensor.disable_bulbs()
        self._sleep(cfg.BULB_SETTLE_S)
        dark_settings = self.sensor.read_settings()
        dark = self._replicates(acq, cfg.REPLICATES)

        thermal_drift = (l1 - l1_control) / l1 if l1 > 0 else math.inf
        readback_delta = (l1_control - l2) / l1_control if l1_control > 0 else math.inf
        bulb_ok = (abs(readback_delta) <= cfg.READBACK_DELTA_MAX
                   and readback_delta <= thermal_drift + cfg.READBACK_EXCESS_OVER_THERMAL)
        print("bulb survival on PTFE ({} replicates each):".format(cfg.REPLICATES))
        print("  L1   before readback   {:10.1f}".format(l1))
        print("  L1'  control, no readback {:7.1f}".format(l1_control))
        print("  L2   after readback    {:10.1f}".format(l2))
        print("  thermal_drift  (L1-L1')/L1   {:+8.4f}".format(thermal_drift))
        print("  readback_delta (L1'-L2)/L1'  {:+8.4f}  {} (|x| <= {}, and <= thermal_drift + {})".format(
            readback_delta, "PASS" if bulb_ok else "FAIL",
            cfg.READBACK_DELTA_MAX, cfg.READBACK_EXCESS_OVER_THERMAL))

        registers_ok = True
        for label, bulb_on, read in (("bulb on", True, lit), ("bulb off", False, dark_settings)):
            expected = expected_settings(bulb_on)
            ok = read == expected
            registers_ok = registers_ok and ok
            print("register readback, {}: {}".format(label, "PASS" if ok else "FAIL"))
            if read is None:
                print("  virtual_read_register() missing from the installed library")
                continue
            for key, want in expected.items():
                print("  {:<20} expected {!s:<6} read {!s:<6}{}".format(
                    key, want, read.get(key), "" if read.get(key) == want else "  <--"))

        # Diagnostic only, no pass/fail. Reference (dark, white) and sample
        # readings should agree in cal/raw; if they don't, the calibrated-value
        # path puts references and samples on different scales.
        rows = [("dark", dark), ("white", white)]
        answer = prompt("Optional: seat a clean carton for the cal/raw sample row and press "
                        "Enter, or type s to skip. ")
        if answer.strip().lower() != "s":
            try:
                self.sensor.enable_white()
                self._sleep(cfg.BULB_SETTLE_S)
                rows.append(("sample", self._replicates(acq, cfg.REPLICATES)))
            finally:
                self.sensor.disable_bulbs()
        ratios = {}
        print("cal/raw on the mean of the {} retained channels:".format(len(cfg.LEVEL_CHANNELS)))
        for name, (raw, calibrated) in rows:
            raw_level, cal_level = level_of(raw), level_of(calibrated)
            ratios[name] = cal_level / raw_level if raw_level > 0 else None
            print("  {:<7} raw={:<10.1f} cal={:<10.2f} cal/raw={}".format(
                name, raw_level, cal_level,
                "n/a" if ratios[name] is None else "{:.4f}".format(ratios[name])))

        self._log(None, None, acq, event="check_spectral", l1=l1, l1_control=l1_control, l2=l2,
                  thermal_drift=thermal_drift, readback_delta=readback_delta,
                  settings_lit=lit, settings_dark=dark_settings, cal_raw_ratios=ratios)
        return bulb_ok and registers_ok

    def calibrate_clean_reference(self, substrate, prompt=input):
        """Interactive session calibration (spec §3). Sets and returns the
        session reference record; the caller saves it."""
        if substrate not in cfg.CONTAM_MODEL:
            raise SpectralError("no contamination model for substrate {!r}".format(substrate))
        if self.sensor is None:
            raise SpectralError("calibration needs the AS7265x")

        prompt("1/3 dark: close the shroud with the platform empty, then press Enter. ")
        dark = self._calibration_block(substrate, "dark", False, cfg.CAL_DARK_REPLICATES)

        prompt("2/3 white: seat the PTFE tile, then press Enter. ")
        white = self._calibration_block(substrate, "white", True, cfg.CAL_WHITE_REPLICATES)

        # A dead bulb here would poison clean_ref_level, and ILLUMINATION_FLOOR is
        # a fraction of it, so nothing downstream could catch it. Raw counts: see
        # WHITE_DARK_RATIO_MIN.
        dark_raw, white_raw = level_of(dark["raw"]), level_of(white["raw"])
        if dark_raw > 0:
            ratio = white_raw / dark_raw
        else:
            ratio = math.inf if white_raw > 0 else 0.0
        print("white/dark raw: {:.1f} / {:.1f} = {:.1f}".format(white_raw, dark_raw, ratio))
        if ratio < cfg.WHITE_DARK_RATIO_MIN:
            raise SpectralError(
                "white tile raw level {:.1f} is only {:.1f}x dark raw level {:.1f} (need >= {}x): "
                "is the bulb lit? Calibration aborted, no reference written".format(
                    white_raw, ratio, dark_raw, cfg.WHITE_DARK_RATIO_MIN))

        # Clean PTFE is near-flat; a rising max/min means the tile is soiled.
        white_means = [statistics.mean(rep[ch] for rep in white["calibrated"])
                       for ch in cfg.LEVEL_CHANNELS]
        flatness = max(white_means) / min(white_means) if min(white_means) > 0 else math.inf
        print("white flatness (max/min over {} channels): {:.3f}".format(
            len(cfg.LEVEL_CHANNELS), flatness))

        levels = []
        for i in range(1, cfg.CAL_SEATINGS + 1):
            for _ in range(cfg.CAL_MAX_RESEATS):
                prompt("3/3 seat clean {} object {}/{}, then press Enter. ".format(
                    substrate, i, cfg.CAL_SEATINGS))
                seating = self._calibration_block(
                    substrate, "seating_{}".format(i), True, cfg.CAL_SEATING_REPLICATES)["calibrated"]
                if max_cv(seating) <= cfg.REPLICATE_CV_MAX:
                    break
                print("replicates unstable (max CV {:.4f} > {}): reseat and try again".format(
                    max_cv(seating), cfg.REPLICATE_CV_MAX))
            else:
                raise SpectralError("seating {} unstable after {} tries".format(i, cfg.CAL_MAX_RESEATS))
            levels.append(level_of(seating))

        config_level = cfg.CONTAM_MODEL[substrate][0]
        record = {
            "substrate": substrate,
            "clean_ref_level": statistics.mean(levels),
            "config_clean_ref_level": config_level,
            "seating_levels": levels,
            "seating_cv": statistics.stdev(levels) / statistics.mean(levels),
            "dark_level": level_of(dark["calibrated"]),
            "dark_raw_level": dark_raw,
            "white_raw_level": white_raw,
            "white_flatness": flatness,
            "timestamp": utc_timestamp(),
            "gain": cfg.GAIN,
            "integration_cycles": cfg.INTEGRATION_CYCLES,
            "bulb_current": cfg.WHITE_BULB_CURRENT,
        }
        append_jsonl(self.log_path, dict(record, event="calibration_result"))
        print("clean_ref_level: measured {:.1f}, config {:.1f} (box-to-box CV {:.4f})".format(
            record["clean_ref_level"], config_level, record["seating_cv"]))
        self.reference = record
        return record

    def _calibration_block(self, substrate, phase, bulb_on, replicates):
        acq = _new_acquisition()
        try:
            self._acquire(acq, self._clock() + cfg.READ_BUDGET_S, bulb_on, replicates)
        except Exception as exc:
            self._log(None, substrate, acq, event="calibration_" + phase, error=repr(exc))
            raise SpectralError("{} block failed: {}".format(phase, exc)) from exc
        self._log(None, substrate, acq, event="calibration_" + phase)
        if acq["settings"] != expected_settings(bulb_on):
            raise SpectralError("instrument_config_mismatch during {} block: read back {}".format(
                phase, acq["settings"]))
        return acq

    def _finish(self, request_id, substrate, acq, reading, attempt=None):
        self._log(request_id, substrate, acq or _new_acquisition(), event="read",
                  attempt=attempt, state=reading.state, d=reading.d, level=reading.level,
                  reason=reading.reason)
        return reading

    def _log(self, request_id, substrate, acq, **fields):
        """Raw counts go to the log on every path, success or not."""
        record = {
            "timestamp": utc_timestamp(),
            "request_id": request_id,
            "substrate": substrate,
            "reference_timestamp": (self.reference or {}).get("timestamp"),
            "clean_ref_level": (self.reference or {}).get("clean_ref_level"),
            "max_cv": max_cv(acq["calibrated"]) if len(acq["calibrated"]) > 1 else None,
        }
        record.update(acq)
        record.update(fields)
        append_jsonl(self.log_path, record)


# --- Session reference file ---------------------------------------------------------------------

def save_reference(record, path=cfg.SESSION_REFERENCE_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
        f.write("\n")


def calibrate_and_save(node, substrate, path, prompt=input):
    """Run the calibration and write `path` only if it completes."""
    record = node.calibrate_clean_reference(substrate, prompt=prompt)
    save_reference(record, path)
    return record


def load_reference(path, live_settings, now=None):
    """Load the --calibrate output. Returns (reference, None) when usable, else
    (None, why): missing, malformed, older than REFERENCE_MAX_AGE_H, or taken
    at instrument settings other than the locked config or the live readback."""
    try:
        with open(path, encoding="utf-8") as f:
            ref = json.load(f)
    except FileNotFoundError:
        return None, "no session reference at {} (run main.py --calibrate)".format(path)
    except (OSError, ValueError) as exc:
        return None, "unreadable session reference {}: {}".format(path, exc)

    try:
        taken = dt.datetime.strptime(ref["timestamp"], UTC_FORMAT).replace(tzinfo=dt.timezone.utc)
        recorded = {"gain": ref["gain"], "integration_cycles": ref["integration_cycles"],
                    "white_current": ref["bulb_current"]}
        substrate, level = ref["substrate"], float(ref["clean_ref_level"])
    except (KeyError, TypeError, ValueError) as exc:
        return None, "malformed session reference {}: {!r}".format(path, exc)

    if substrate not in cfg.CONTAM_MODEL or not level > 0:
        return None, "session reference has no usable model: {!r} level={}".format(substrate, level)
    now = now or dt.datetime.now(dt.timezone.utc)
    age_h = (now - taken).total_seconds() / 3600
    if not 0 <= age_h <= cfg.REFERENCE_MAX_AGE_H:
        return None, "session reference is {:.1f} h old (max {} h)".format(age_h, cfg.REFERENCE_MAX_AGE_H)
    locked = {k: v for k, v in expected_settings(bulb_on=False).items() if k in recorded}
    if recorded != locked:
        return None, "session reference recorded at {}, config locks {}".format(recorded, locked)
    live = None if live_settings is None else {k: live_settings.get(k) for k in locked}
    if live != locked:
        return None, "live instrument readback {} does not match {}".format(live, locked)
    return ref, None
