# Task: integrate the spectral contamination veto into the existing capture → VLM → routing pipeline

## Before writing any code

Read the existing pipeline first and conform to what is there. Do not restructure it.

1. Find and read the Pi-side capture script that sends images to the FastAPI VLM server and
   receives detections back.
2. Find and read the module that holds the `ROUTING` dictionary and the `route()` function.
3. Report back, before editing: the exact shape of a VLM detection object (field names for
   label, bbox, confidence, and the per-object `contaminated` flag if present), the current
   signature of `route()`, and where in the Pi script the routing decision is made.

If any of the following is ambiguous, stop and ask rather than guessing:
- whether `route()` is called per detection or per frame
- whether the Pi or the server owns the routing decision
- how the servo/diverter is commanded

## Architectural boundary — do not violate

    VLM (proposer)    -> object label + bbox
    spectral (gate)   -> contamination tri-state
    decision node     -> owns ALL routing logic

The VLM proposes. The spectral node gates. Neither routes. The decision node is a pure,
deterministic, config-driven function with a conservative fallback on every failure mode.

Nothing in the spectral module may import from or call the routing module, and the spectral
module must never return a stream name.

## 1. New module: `spectral_node.py` (runs on the Pi)

Owns the AS7265x. Exposes one function:

```python
def read_contamination(substrate: str, *, timeout_s: float = 5.0) -> SpectralReading
```

Returns a dataclass, never raises to the caller:

```python
@dataclass(frozen=True)
class SpectralReading:
    state: bool | None        # True=contaminated, False=clean, None=undetermined
    d: float | None           # (clean_ref - live) / clean_ref
    level: float | None       # mean cal over the 13 retained channels
    reason: str               # always populated, see reason codes below
    raw: dict | None          # all 18 raw channel counts, for logging
```

Requirements:

- **Instrument settings are locked and asserted, not set-and-hope.** gain 64x,
  `integration_cycles=255`, white bulb only at 25 mA. Read the settings back from the device
  after configuring and return `reason="instrument_config_mismatch"`, `state=None` if they
  do not match. A reading taken at different settings is not comparable to the stored
  reference and must not be used.
- **13 retained channels only** for the level feature: A B C D E F G H I J R S T.
  Exclude K, L, U, V, W (760–940 nm) as out-of-band visible-flux leakage.
  Level = arithmetic mean of the calibrated values over those 13.
- **Take 3 replicates per read.** Compute max per-channel CV across them. If it exceeds
  `REPLICATE_CV_MAX` (0.005), return `state=None`, `reason="replicate_unstable"`. This
  catches the object leaving the field of view mid-read — the on-device `saturated` and
  `suspect_stale` flags do NOT catch it (they were False for every row of a dataset that
  contained a 1000x intra-seating excursion).
- **Always write raw counts to the log.** Compute derived quantities (d, state) here but
  log raw alongside. Never discard raw.
- `substrate` not present in `CONTAM_MODEL` -> `state=None`,
  `reason="substrate_uncharacterized"`. Do not fall back to another substrate's model.

### Tri-state rule — two-sided

```python
d = (clean_ref - live) / clean_ref

abs(d) >= VETO          -> True   (contaminated)
UNDETERMINED <= abs(d)  -> None   (undetermined)
otherwise               -> False  (clean)
```

**The two-sidedness is not optional.** A thin wetting film darkens the surface (positive d);
a thick pooled spill forms a specular liquid surface and reads *brighter* than clean
(measured d = -0.246). A one-sided test passes a pooled spill as cleaner-than-clean, which
is a false negative routing contamination into the recyclable stream.

## 2. Config: `contamination_config.py`

Keep every constant in one module. No magic numbers in the logic.

```python
# Item label (from the VLM) -> substrate. A label absent here has no spectral model.
# togo_box spans carton / EPS foam / PET and only carton is characterized, so this map
# is deliberately narrow. Do not widen it without capture data for the new substrate.
SUBSTRATE_OF = {
    "togo_box": "carton",   # PROVISIONAL: assumes carton stock, see open items
    "paper":    None,       # fibrous, expected to behave like carton, UNMEASURED
}

# substrate -> (clean_ref_level, veto, undetermined)
CONTAM_MODEL = {
    "carton": (5388.5, 0.08, 0.04),
}

REPLICATE_CV_MAX = 0.005
```

Threshold provenance, as a comment in the file: clean carton box-to-box CV = 0.0253 (n=5);
widest clean single-read deviation = 0.0355; weakest oiled box = 0.102. 0.08 sits in that
gap, gives 0/25 false positives and 4/25 false negatives on the capture set. Session
20261014T0443Z.

## 3. Startup calibration (required)

`clean_ref_level` in the config is a raw count from one session at these instrument
settings. **No white or dark reference was captured in that session**, so the value cannot
be carried across sessions as an absolute.

Implement `calibrate_clean_reference(substrate)` to be run at service startup:

1. Dark reference, LEDs off, 3 replicates.
2. White reference on the PTFE tile, 3 replicates. Log the max/min across the 13 channels
   as a flatness figure — clean PTFE is near-flat, so a rising value means the tile is
   soiled.
3. A clean object of that substrate, 5 seatings x 3 replicates.
4. Overwrite the in-memory `clean_ref_level` for the session. Log both the config value and
   the measured one.

If calibration has not been run, `read_contamination` returns `state=None`,
`reason="no_session_reference"`. Do not silently use the config constant.

## 4. Decision node changes

Order of operations per detection — the skip comes before the read:

```python
stream = ROUTING.get(label)
if stream is None:          return CLEAR, "unknown_label"
if stream is CLEAR:         return CLEAR, "class_landfill"      # no spectral read at all

substrate = SUBSTRATE_OF.get(label)
reading = spectral.read_contamination(substrate)

if reading.state is None:   return CLEAR, f"contamination_undetermined:{reading.reason}"
if reading.state:           return CLEAR, "contamination_veto"
return GREEN, "class_recyclable_clean"
```

Seven of nine classes route CLEAR regardless of contamination state, so the skip avoids a
~1.6 s read plus LED settling on most objects. Keep `route()` pure — pass the
`SpectralReading` in as an argument, do not call the sensor from inside it.

## 5. Reason codes

Every failure mode gets a distinct code. Do not collapse them.

| code | meaning |
|---|---|
| `class_landfill` | routed CLEAR by class, no spectral read taken |
| `class_recyclable_clean` | GREEN, spectral gate passed |
| `contamination_veto` | spectral gate fired |
| `unknown_label` | label not in ROUTING |
| `substrate_uncharacterized` | no model for this substrate |
| `no_session_reference` | startup calibration not run |
| `instrument_config_mismatch` | gain / cycles / bulb readback wrong |
| `replicate_unstable` | replicate CV above threshold |
| `sensor_timeout` | no response within timeout |
| `multiple_detections` | see below |

## 6. Registration guard

The AS7265x is a single-point sensor with no spatial resolution. Nothing guarantees its spot
lands on the object the VLM detected, and camera-to-spectral registration is unsolved.

For the POC, guard rather than assume: if the VLM returns more than one detection, return
`CLEAR, "multiple_detections"` and do not take a spectral read. A single object on the
platform is the only case the spectral channel can currently claim.

Leave a TODO naming registration as the blocker.

## 7. Do not

- Do not average or otherwise fuse the VLM `contaminated` flag with the spectral state. They
  estimate different variables. The spectral state wins for substrates that have a model;
  the VLM flag is used only where no model exists.
- Do not compute reflectance on the Pi. Raw and calibrated counts only; derived quantities
  belong in analysis scripts.
- Do not add a spectral read for a class that routes CLEAR.
- Do not retry a failed spectral read more than once. A second failure is `None`, which
  routes conservatively, which is correct.
- Do not infer substrate from bbox size, color, or anything else. It comes from the label
  map or it is `None`.

## 8. Tests

Unit tests with a mocked sensor, no hardware:

- clean carton level 5388 -> `state=False` -> GREEN
- level 4380 (d=+0.187) -> `state=True` -> CLEAR, `contamination_veto`
- level 6712 (d=-0.246, pooled spill) -> `state=True` -> CLEAR  **(two-sided case — this
  test fails on a one-sided implementation, which is the point)**
- level 5100 (d=+0.053) -> `state=None` -> CLEAR, undetermined
- `paper_hot_cup` label -> no model -> `None` -> CLEAR, never GREEN
- `straw` / `wrapper` / any CLEAR-routing class -> sensor mock asserts it was never called
- sensor raises -> CLEAR, `sensor_timeout`
- 2 detections -> CLEAR, `multiple_detections`, sensor never called

## 9. Open items — leave as TODO comments, do not solve

- `togo_box -> carton` assumes carton stock. Foam and PET to-go boxes exist in the taxonomy,
  have different clean baselines, and would be misrouted by the carton reference.
- Freshly applied contamination may read as clean. Oiled boxes measured immediately after
  application gave d = -0.007 and -0.002, rising to +0.2 over the following minutes as the
  oil wicked into the fibre. Detectability is time-dependent and this is not handled.
- No white referencing in deployment means drift between the reference and inference is
  unbounded.
- Thresholds derive from n=5 boxes, one session, one contaminant (oil), one carton stock.
