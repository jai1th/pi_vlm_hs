"""Class -> servo action. Edit ACTIONS to change what each class does.

Direction follows servo_test.py (Left key = negative angle, Right = positive):
  GREEN bin = left  = -90 deg
  CLEAR bin = right = +90 deg
After the move the servo holds for 1 s and returns to 0 (see servo.py).

The model outputs 9 classes. togo_box and togo_cup each span both bins in the
routing split, so they follow the majority of their TACO source classes:
  togo_box = Meal carton, Pizza box, Disposable food container (GREEN)
             + Foam food container (CLEAR)                      -> GREEN
  togo_cup = Paper cup, Foam cup (CLEAR)
             + Disposable plastic cup (GREEN)                   -> CLEAR
"""

import time

GREEN = -90   # left
CLEAR = +90   # right

ACTIONS = {
    "togo_box": GREEN,
    "plastic_bottle": GREEN,
    "paper": GREEN,

    "togo_cup": CLEAR,
    "cup_lid": CLEAR,
    "straw": CLEAR,
    "tissue": CLEAR,
    "wrapper": CLEAR,
    "glass_bottle": CLEAR,
}


def act_on_result(result, servo=None, t0=None):
    """Act on the first detected object; servo=None only prints (dry run).

    The servo can only do one thing per frame, so extra objects are listed
    but ignored. Returns the angle used, or None if nothing was done.

    t0 is a time.monotonic() value taken at the Enter press; the elapsed time
    to the decision is printed before the servo moves, so actuation is not
    included.
    """
    took = "" if t0 is None else " [decision in {:.2f} s]".format(time.monotonic() - t0)
    objects = result.get("objects", [])
    if not objects:
        print("[{}] no objects detected: no action{}".format(result.get("request_id", "?"), took))
        return None

    label = objects[0]["label"]
    others = [o["label"] for o in objects[1:]]
    angle = ACTIONS.get(label)
    note = " (ignoring: {})".format(", ".join(others)) if others else ""
    if angle is None:
        print("{}: no action assigned{}{}".format(label, note, took))
        return None

    print("{} -> {:+d} deg{}, hold 1 s, home{}".format(label, angle, took, note))
    if servo is not None:
        servo.swing(angle)
    return angle
