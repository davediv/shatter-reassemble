"""Every tunable in one place.

Two kinds of numbers live here:

*Structural* constants (resolution, substep rate, solver iteration counts)
are module-level and frozen -- changing them changes the shape of the
program, not its feel.

*Tunables* are the gesture thresholds. Snap detection in particular is
finicky and has to be dialled in against a real hand in real lighting, so
they live in a mutable dataclass that the tuning mode edits live and
persists to ``tuning.json``.

Distance invariance
-------------------
Nothing here is expressed in pixels except where noted. Gesture thresholds
are in units of *hand span* -- the pixel distance from the wrist (landmark
0) to the middle-finger MCP (landmark 9), calibrated over the first 60
frames a hand is visible. Velocities are hand spans per second. This is
what makes the app behave identically whether the user is leaning into the
lens or standing back from it. See tracking.HandSpanCalibrator.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
RECORDING_DIR = ROOT / "recordings"
TUNING_FILE = ROOT / "tuning.json"

HAND_MODEL = MODEL_DIR / "hand_landmarker.task"
SEGMENT_MODEL = MODEL_DIR / "selfie_segmenter.tflite"

# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 60
CAMERA_FPS_FALLBACK = 30
CAMERA_INDEX = 0

# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------

CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080

# --------------------------------------------------------------------------
# Tracking
# --------------------------------------------------------------------------

NUM_HANDS = 2

# MediaPipe is fed a downscaled copy of the frame. Landmarks come back in
# normalised coordinates, so an aspect-preserving downscale costs nothing
# in mapping accuracy -- and the palm detector resamples to 192x192
# internally regardless, so a hand occupies the same fraction of the
# detector's input either way. Measured on an M1 Air with capture running:
# 1280x720 -> 9.6ms/frame, 640x360 -> 9.1ms, 512x288 -> 6.6ms. 640 keeps
# more detail for the landmark model, which is what snap detection reads.
TRACKING_INPUT_WIDTH = 640

# Python's default 5ms GIL switch interval is poison here: MediaPipe's
# binding dispatches every detect_for_video through a thread-pool round
# trip, so a waiting thread can eat a full switch interval per call. With
# capture running, dropping 5ms -> 1ms took detection from 12.2ms to 9.3ms
# and worst case from 26.0ms to 21.4ms.
GIL_SWITCH_INTERVAL = 0.001

MIN_HAND_DETECTION_CONFIDENCE = 0.5
MIN_HAND_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

NUM_LANDMARKS = 21

# One Euro filter. Deliberately light: this app lives off velocity spikes,
# and an aggressive filter eats exactly the transients that snap and clap
# detection key on. The unsmoothed channel stays available in parallel --
# smoothing is for what you *see*, never for what you *detect*.
ONE_EURO_MIN_CUTOFF = 1.0
ONE_EURO_BETA = 0.007
ONE_EURO_D_CUTOFF = 1.0

# Hand span calibration: median of this many frames, then held.
HAND_SPAN_CALIBRATION_FRAMES = 60
# Below this many pixels the hand is too far away to trust.
HAND_SPAN_MIN_PIXELS = 18.0
HAND_SPAN_DEFAULT_PIXELS = 90.0

# Landmark indices (MediaPipe hand topology).
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_MCP = 13
RING_TIP = 16
PINKY_MCP = 17
PINKY_TIP = 20

# Finger chains, used both for the debug skeleton and to build the capsule
# colliders that let an open palm stir the pile.
FINGER_CHAINS = (
    (0, 1, 2, 3, 4),        # thumb
    (0, 5, 6, 7, 8),        # index
    (5, 9, 10, 11, 12),     # middle
    (9, 13, 14, 15, 16),    # ring
    (13, 17, 18, 19, 20),   # pinky
    (0, 17),                # palm base closing edge
)

# Silhouette segmentation runs on its own thread at its own rate: it is a
# faint background outline, not something that needs to track at 60Hz, and
# stacking it onto the landmark thread blew the tracking budget (measured
# 8.9ms at full res, 2.8ms at 256x144).
SEGMENT_INPUT_WIDTH = 256
SEGMENT_MAX_HZ = 20.0

# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------

PHYSICS_HZ = 120.0
PHYSICS_DT = 1.0 / PHYSICS_HZ
# Never spiral: if we fall this far behind, drop the backlog rather than
# running an unbounded catch-up loop.
MAX_SUBSTEPS_PER_FRAME = 5

SOLVER_ITERATIONS = 8
SOLVER_ITERATIONS_DEGRADED = 5
POSITION_ITERATIONS = 3

GRAVITY = 2600.0            # px/s^2 in canvas space
LINEAR_DAMPING = 0.06
ANGULAR_DAMPING = 0.10
RESTITUTION = 0.16
FRICTION = 0.42
# Baumgarte-style positional bias, applied as pseudo-velocity so it does
# not inject real kinetic energy into the pile.
BAUMGARTE = 0.2
PENETRATION_SLOP = 0.35     # px

# Sleeping. This is the single thing that makes 800 bodies affordable: a
# settled pile costs almost nothing because settled bodies leave the solver.
SLEEP_LINEAR_VELOCITY = 22.0    # px/s -- 0.37px per frame at 60fps
SLEEP_ANGULAR_VELOCITY = 0.45   # rad/s
SLEEP_TIME = 0.45               # s below both thresholds before sleeping
# A body only keeps its neighbours awake once it is moving this many times
# faster than the sleep threshold. With no gap, anything sitting just over
# the line holds the whole pile awake through the contact graph.
SLEEP_DISTURB_SCALE = 2.5

# Fake perspective. Bodies are planar; depth scales and shades them, and
# it also decides who collides with whom.
DEPTH_NEAR = 0.86
DEPTH_FAR = 1.14
PERSPECTIVE_STRENGTH = 0.055

# The shards tessellate the canvas exactly, so in a strictly 2D world they
# have nowhere to fall: a perfect tiling cannot compact. Sorting them into
# depth layers and only colliding within a layer is what lets the pile
# actually pile -- each layer holds a third of the total area, so it
# collapses into roughly the bottom third of the frame and empties the top
# into the void the silhouette lives in. It also cuts the broadphase pair
# count by about the same factor, which is the difference between making
# the physics budget and missing it.
DEPTH_LAYERS = 3

# --------------------------------------------------------------------------
# Fracture
# --------------------------------------------------------------------------

SHARD_COUNT_TIERS = (800, 550, 350)
SHARD_COUNT_MIN = 300
# Fraction of seeds drawn from a Gaussian around the snap point; the rest
# are blue-noise-ish across the frame. Clustering is what makes the break
# radiate from the hand instead of looking like uniform crazy paving.
FRACTURE_CLUSTER_FRACTION = 0.55
FRACTURE_CLUSTER_SIGMA = 0.17     # fraction of canvas diagonal
FRACTURE_RELAX_ITERATIONS = 1     # Lloyd relaxation passes on the far field
# Clipping the diagram to the canvas occasionally leaves a sliver a couple
# of pixels across. It is invisible, but it still costs a physics body and
# a solver slot, so anything under this is dropped.
MIN_SHARD_AREA = 16.0             # px^2
# How far the hand may drift between arming and firing before a
# speculatively-built fracture is thrown away and rebuilt.
FRACTURE_PREDICT_TOLERANCE = 160.0   # px

# --------------------------------------------------------------------------
# Shard appearance
# --------------------------------------------------------------------------

BEVEL_WIDTH = 3.2           # px, inset of the lit rim
SHARD_THICKNESS = 5.0       # px, parallax extrusion of the side wall
REFRACTION_STRENGTH = 11.0  # px of UV displacement at the bevel
SHADOW_OFFSET = (7.0, 11.0)
SHADOW_ALPHA = 0.42

# --------------------------------------------------------------------------
# Reassembly
# --------------------------------------------------------------------------

REASSEMBLE_DURATION = 0.62      # per-shard flight time
REASSEMBLE_MAX_DELAY = 0.42     # extra delay for the outermost shard
REASSEMBLE_OVERSHOOT = 1.30     # back-out easing overshoot
REASSEMBLE_CROSSFADE = 0.150    # frozen -> live, over the final 150ms
REASSEMBLE_FLASH_DURATION = 0.28

# --------------------------------------------------------------------------
# Degradation ladder
# --------------------------------------------------------------------------

# Rolling average frame time above this and we step down a rung.
LADDER_STEP_DOWN_MS = 15.0
# ... and back up only well below it, so we never oscillate on the boundary.
LADDER_STEP_UP_MS = 12.0
LADDER_WINDOW_FRAMES = 90
LADDER_DWELL_SECONDS = 1.5      # minimum time on a rung before moving again


@dataclass(frozen=True)
class QualityLevel:
    """One rung of the degradation ladder."""

    name: str
    shard_count: int
    refraction: bool
    bevel: bool
    solver_iterations: int
    shadows: bool


# Rung 0 is everything on. Each subsequent rung gives up exactly one thing,
# cheapest-looking sacrifice first. Shard count leads because it is the only
# knob that reduces *all three* budgets at once (physics, vertex, fill).
QUALITY_LADDER: tuple[QualityLevel, ...] = (
    QualityLevel("full",        800, True,  True,  SOLVER_ITERATIONS,          True),
    QualityLevel("shards-550",  550, True,  True,  SOLVER_ITERATIONS,          True),
    QualityLevel("shards-350",  350, True,  True,  SOLVER_ITERATIONS,          True),
    QualityLevel("flat",        350, False, True,  SOLVER_ITERATIONS,          True),
    QualityLevel("no-bevel",    350, False, False, SOLVER_ITERATIONS,          True),
    QualityLevel("solver-5",    350, False, False, SOLVER_ITERATIONS_DEGRADED, True),
    QualityLevel("no-shadow",   350, False, False, SOLVER_ITERATIONS_DEGRADED, False),
)

# --------------------------------------------------------------------------
# Recording
# --------------------------------------------------------------------------

RECORD_FPS = 60
RECORD_BITRATE = "12M"
RECORD_CODEC = "libvpx-vp9"
# 16:9 is the canvas as-is; 9:16 is a centre crop of it.
RECORD_LANDSCAPE = (1920, 1080)
RECORD_PORTRAIT = (1080, 1920)


@dataclass
class Tunables:
    """Gesture thresholds -- the numbers the tuning mode exists to dial in.

    All distances are in hand spans, all velocities in hand spans/second,
    so these hold at any distance from the camera.
    """

    # -- snap -------------------------------------------------------------
    # Thumb tip (4) to middle tip (12) closer than this arms the snap.
    snap_pinch_distance: float = 0.25
    # Middle tip must then exceed this speed to fire.
    snap_velocity_threshold: float = 13.0
    # ... and must also have actually travelled this far, wrist-relative,
    # within the window. Landmark jitter is a zero-mean random walk: it
    # can spike the instantaneous velocity but it accumulates almost no
    # net displacement, while a real snap moves the fingertip most of a
    # hand span. Measured against synthetic 3px landmark noise, velocity
    # alone left only a 1.3x margin over the noise floor; adding travel
    # takes the margin to roughly 6x.
    snap_min_travel: float = 0.30
    # Upper sanity bound. A fingertip during a real snap peaks somewhere
    # around 15-35 spans/s depending on the tracking rate; anything past
    # this is a tracking discontinuity -- a re-association, or MediaPipe
    # jumping after a brief loss -- not a hand. Without it, a landmark
    # teleport reads as the most emphatic snap the user has ever performed.
    snap_max_velocity: float = 50.0
    # ... within this long of the pinch, or the arm expires.
    snap_window: float = 0.120
    # Re-arm guard so one snap cannot fire twice.
    snap_lockout: float = 0.500
    # The release has to actually separate the fingers, which rejects a slow
    # squeeze-and-wave that would otherwise trip the velocity gate.
    snap_release_distance: float = 0.34

    # -- clap -------------------------------------------------------------
    # Palm centroids (landmark 9 of each hand) closer than this...
    clap_distance: float = 0.50
    # ... while closing at least this fast.
    clap_velocity_threshold: float = 1.90
    clap_lockout: float = 0.400

    # -- open palm (stir) -------------------------------------------------
    # Mean fingertip-to-wrist distance above this counts the hand as open.
    open_palm_extension: float = 1.45
    # Radius of the capsule colliders along the finger chains, in hand spans.
    stir_capsule_radius: float = 0.16
    # How hard a moving hand shoves a shard it sweeps through.
    stir_impulse_scale: float = 1.0

    def save(self, path: Path | None = None) -> Path:
        path = path or TUNING_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2, sort_keys=True))
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "Tunables":
        path = path or TUNING_FILE
        base = cls()
        if not path.exists():
            return base
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return base
        known = {f for f in asdict(base)}
        return replace(base, **{k: float(v) for k, v in data.items() if k in known})


@dataclass
class RuntimeOptions:
    """Command-line shaped switches that pick which parts of the app run."""

    width: int = CANVAS_WIDTH
    height: int = CANVAS_HEIGHT
    camera_index: int = CAMERA_INDEX
    source: str = "camera"          # camera | synthetic | <path to video>
    tuning_mode: bool = False
    show_debug: bool = True
    fullscreen: bool = False
    vsync: bool = True
    shard_count: int = SHARD_COUNT_TIERS[0]
    ladder_enabled: bool = True
    gpu_delegate: bool = True
    segmentation: bool = True
    tunables: Tunables = field(default_factory=Tunables.load)
