(base) niche4b@niche4b-desktop:~/wastesep/servo_control_tesst$ ls
test.py
(base) niche4b@niche4b-desktop:~/wastesep/servo_control_tesst$ cat test.py
#!/usr/bin/env python3
"""
HS-805BB keyboard control on Raspberry Pi 4.
Left/Right arrow keys (or A/D) turn the servo. C = center, Q = quit.
Hold a key to keep turning (uses keyboard auto-repeat).

Wiring:
  Servo red    -> PSU +6V
  Servo black  -> PSU GND
  Servo yellow -> Pi GPIO23 (physical pin 16)
  Pi GND       -> PSU GND (common ground)

Setup:
  sudo apt install pigpio python3-pigpio
  sudo systemctl enable --now pigpiod
Run in a terminal (works over SSH):
  python3 servo_demo.py
"""

import curses
import pigpio

SERVO_PIN = 23

CENTER_US = 1500
US_PER_DEG = 400 / 45   # Hitec: ~400 us per 45 deg
MIN_DEG, MAX_DEG = -90, 90
STEP_DEG = 10           # degrees per key press


def deg_to_us(deg):
    return int(CENTER_US + deg * US_PER_DEG)


def main(stdscr):
    pi = pigpio.pi('::1', 8888)
    if not pi.connected:
        raise SystemExit("pigpiod not running: sudo systemctl start pigpiod")

    curses.curs_set(0)
    stdscr.keypad(True)
    angle = 0

    try:
        while True:
            pi.set_servo_pulsewidth(SERVO_PIN, deg_to_us(angle))
            stdscr.erase()
            stdscr.addstr(0, 0, "Left/A: turn left   Right/D: turn right   C: center   Q: quit")
            stdscr.addstr(2, 0, f"Angle: {angle:+4d} deg   Pulse: {deg_to_us(angle)} us")
            stdscr.refresh()

            key = stdscr.getch()
            if key in (curses.KEY_LEFT, ord('a'), ord('A')):
                angle = max(MIN_DEG, angle - STEP_DEG)
            elif key in (curses.KEY_RIGHT, ord('d'), ord('D')):
                angle = min(MAX_DEG, angle + STEP_DEG)
            elif key in (ord('c'), ord('C')):
                angle = 0
            elif key in (ord('q'), ord('Q')):
                break
    except KeyboardInterrupt:
        pass
    finally:
        pi.set_servo_pulsewidth(SERVO_PIN, 0)  # stop pulses (servo goes limp)
        pi.stop()


if __name__ == "__main__":
    curses.wrapper(main)
