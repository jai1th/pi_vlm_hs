"""Every constant the contamination veto uses (veto_integration_spec.md §2).

spectral_node.py (the gate) and actions.py (the decision node) read from here;
no threshold, channel list or instrument setting appears as a literal in either.

Open items, deliberately not solved (spec §9):
TODO: togo_box -> carton assumes carton stock. Foam and PET to-go boxes exist in
      the taxonomy, have different clean baselines, and would be misrouted by the
      carton reference.
TODO: freshly applied contamination may read as clean. Oiled boxes measured
      immediately after application gave d = -0.007 and -0.002, rising to +0.2
      over the following minutes as the oil wicked into the fibre. Detectability
      is time-dependent and this is not handled.
TODO: no white referencing in deployment, so drift between the reference and
      inference is unbounded (the startup calibration bounds it to one session).
TODO: thresholds derive from n=5 boxes, one session, one contaminant (oil), one
      carton stock.
"""

import os

# --- Label -> substrate -------------------------------------------------------

# Item label (from the VLM) -> substrate. A label absent here has no spectral model.
# togo_box spans carton / EPS foam / PET and only carton is characterized, so this map
# is deliberately narrow. Do not widen it without capture data for the new substrate.
SUBSTRATE_OF = {
    "togo_box":       "carton",   # PROVISIONAL, carton stock assumed
    "paper":          None,       # fibrous, unmeasured
    "plastic_bottle": None,       # wrong sensor: gate is fill (depth), not surface
}

# GREEN classes whose gate is something the spectral node cannot measure. They
# route CLEAR as gate_not_implemented and never get a spectral read.
GATE_NOT_IMPLEMENTED = {
    "plastic_bottle": "fill level (depth): a bottle with liquid in it is the failure mode",
}

# substrate -> (clean_ref_level, veto, undetermined)
#
# Threshold provenance (session 20261014T0443Z): clean carton box-to-box
# CV = 0.0253 (n=5); widest clean single-read deviation = 0.0355; weakest oiled
# box = 0.102. 0.08 sits in that gap, gives 0/25 false positives and 4/25 false
# negatives on the capture set.
#
# clean_ref_level is a raw count from that one session with no white or dark
# reference, so it is never used at run time: --calibrate measures the session
# value and it is only logged next to the measured one.
CONTAM_MODEL = {
    "carton": (5388.5, 0.08, 0.04),
}

# A dead or dropped bulb reads near-dark and would otherwise be reported as a
# confident contamination_veto. Darkest real reading observed was 0.63 x clean_ref
# (oil 005, seating 0); dark captures sit near 0.002. A 0.30 floor has ~2x margin
# below any real contamination and ~150x above dark.
ILLUMINATION_FLOOR = 0.30

# Max per-channel CV across the replicates of one read. Above this the object
# moved mid-read; the on-device saturated / suspect_stale flags do not catch it.
REPLICATE_CV_MAX = 0.005

# --- Instrument (locked; asserted by register readback on every read) ----------

GAIN = 3                  # 0=1x, 1=3.7x, 2=16x, 3=64x
INTEGRATION_CYCLES = 255  # ~2.8 ms each. Part of what the reference means: do not
                          # lower it to recover latency.
WHITE_BULB_CURRENT = 1    # 0=12.5, 1=25, 2=50, 3=100 mA. IR and UV bulbs stay off.

# Level feature: mean calibrated count over these 13 channels. K, L, U, V, W
# (760-940 nm) are excluded as out-of-band visible-flux leakage.
LEVEL_CHANNELS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "R", "S", "T")

# --- Read sequence ---------------------------------------------------------------

REPLICATES = 3
BULB_SETTLE_S = 0.20         # same settle the channel-usability run used
CONVERSION_TIMEOUT_S = 3.0   # per conversion (DATA_RDY poll)
READ_BUDGET_S = 12.0         # whole replicate sequence including bulb settling
SENSOR_RETRIES = 1           # driver exceptions / timeouts only, never replicate_unstable

# --- Startup calibration (main.py --calibrate) -----------------------------------

CAL_DARK_REPLICATES = 3
CAL_WHITE_REPLICATES = 3
CAL_SEATINGS = 5
CAL_SEATING_REPLICATES = 3
CAL_MAX_RESEATS = 3          # unstable seatings re-prompted before calibration fails

# Refuse to calibrate with a dead bulb. RAW counts, not calibrated: the only white/dark
# records available are synthetic and their cal/raw ratio (0.005) disagrees with real
# captures (0.78) by ~150x. Dark raw sits at ~4.5 counts and real sample raw runs
# 290-15700, so a live white tile clears dark by more than 1000x. 100x is a wide guard
# that still catches a dead bulb, where the ratio would be ~1.
WHITE_DARK_RATIO_MIN = 100

# --check-spectral pass marks. L1 -> L2 spans ~10 s of lit readings, over which the LED
# junction warms and its output sags, so the readback is judged against a no-readback
# thermal control leg rather than against replicate noise (REPLICATE_CV_MAX).
READBACK_DELTA_MAX = 0.05          # fail outright above this level change across the readback
READBACK_EXCESS_OVER_THERMAL = 0.02  # fail if readback_delta > thermal_drift + this
REFERENCE_MAX_AGE_H = 12
SESSION_REFERENCE_PATH = "session_reference.json"

# --- Logs (JSON lines, appended) ---------------------------------------------------

SPECTRAL_LOG_PATH = "logs/spectral.jsonl"    # every read, raw counts always included
DECISION_LOG_PATH = "logs/decisions.jsonl"   # every routed frame

# --- Hardware driver ------------------------------------------------------------------

# The AS7265x driver proven on this fixture (referencer.RealSensor) lives here.
SPECTRAL_COLLECTOR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spectral_collector"))
I2C_ADDRESS = 0x49
