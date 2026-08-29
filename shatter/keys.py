"""GLFW key codes, redeclared so headless code paths need no GLFW.

The tuning mode and the app's hotkeys both dispatch on key codes, and both
have to be constructible in a standalone context where no window -- and
therefore no GLFW -- exists. These values are stable parts of the GLFW ABI.
"""

from __future__ import annotations

RELEASE, PRESS, REPEAT = 0, 1, 2

ESCAPE = 256
ENTER = 257
TAB = 258
SPACE = 32
RIGHT, LEFT, DOWN, UP = 262, 263, 264, 265

A, B, C, D, E, F, G, H, I, J, K, L, M = 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77
N, O, P, Q, R, S, T, U, V, W, X, Y, Z = 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90

DIGIT_0 = 48
LEFT_BRACKET, RIGHT_BRACKET = 91, 93
MINUS, EQUAL = 45, 61

MOD_SHIFT = 0x0001
