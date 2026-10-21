"""HS-805BB servo on the Pi, driven through pigpiod (same setup as servo_test.py).

Wiring: signal -> GPIO23 (physical pin 16), servo supply 6 V, common ground.
Needs `sudo systemctl enable --now pigpiod`. pigpio is imported lazily so the
rest of the code (and --dry-run) works on machines without it.
"""

import math
import time

SERVO_PIN = 23
CENTER_US = 1500                # pulse for 0 degrees (home)
US_PER_DEG = 400 / 45           # Hitec: ~400 us per 45 deg
MIN_DEG, MAX_DEG = -90, 90

DEFAULT_SPEED_DPS = 90.0        # "medium": a full 90 deg sweep takes ~1 s
HOLD_S = 1.0                    # time spent at the target before returning to rest
STEP_S = 0.02                   # one servo frame (50 Hz)


class ServoError(Exception):
    pass


def deg_to_us(deg):
    return int(CENTER_US + deg * US_PER_DEG)


class Servo:
    def __init__(self, pin=SERVO_PIN, speed_dps=DEFAULT_SPEED_DPS, hold_s=HOLD_S,
                 rest_deg=0.0, host="::1", port=8888, pi=None):
        if speed_dps <= 0:
            raise ValueError("servo speed must be > 0 deg/s")
        self.pin = pin
        self.speed_dps = speed_dps
        self.hold_s = hold_s
        self.rest_deg = max(MIN_DEG, min(MAX_DEG, rest_deg))
        self.host = host
        self.port = port
        self._pi = pi
        self._angle = 0.0

    def open(self):
        """Connect to pigpiod and park the servo at its rest angle."""
        if self._pi is None:
            try:
                import pigpio
            except ImportError as exc:
                raise ServoError("pigpio is not installed: {}".format(exc)) from exc
            self._pi = pigpio.pi(self.host, self.port)
        if not self._pi.connected:
            raise ServoError("pigpiod not running: sudo systemctl start pigpiod")
        self._set(self.rest_deg)

    def _set(self, angle):
        self._angle = angle
        self._pi.set_servo_pulsewidth(self.pin, deg_to_us(angle))

    def _ramp(self, target):
        """Move smoothly to `target` degrees at speed_dps."""
        start = self._angle
        delta = target - start
        if delta == 0:
            return
        duration = abs(delta) / self.speed_dps
        steps = max(1, math.ceil(duration / STEP_S))
        for i in range(1, steps + 1):
            self._set(start + delta * i / steps)
            time.sleep(duration / steps)

    def swing(self, target_deg):
        """Go to target_deg, stay hold_s seconds, return to rest. Blocks until
        back at rest; a target equal to the rest angle is a no-op."""
        target = max(MIN_DEG, min(MAX_DEG, target_deg))
        if target == self.rest_deg:
            return
        self._ramp(target)
        time.sleep(self.hold_s)
        self._ramp(self.rest_deg)

    def close(self):
        if self._pi is not None:
            try:
                self._pi.set_servo_pulsewidth(self.pin, 0)  # stop pulses
                self._pi.stop()
            finally:
                self._pi = None
