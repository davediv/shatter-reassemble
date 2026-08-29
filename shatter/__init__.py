"""Shatter & Reassemble -- a finger snap breaks the camera feed into glass;
a clap puts it back, frame-perfect.

Package layout:
    config      tunables, quality tiers, degradation ladder thresholds
    camera      threaded capture (60fps target, 30fps fallback)
    filters     One Euro filter, vectorised over the whole landmark set
    tracking    MediaPipe hand landmarks -> mirrored, smoothed, span-normalised
    gestures    snap / clap / open-palm detection off the raw velocity channel
    fracture    one-shot Voronoi decomposition into renderable shard geometry
    physics     2.5D sequential-impulse solver over flat Float32 arrays
    reassemble  staggered easing back to the rest pose
    profiler    frame budget instrumentation + the degradation ladder
    recorder    VP9 capture of the canvas in 16:9 and 9:16
    render/     GL context, shard batch, background, debug overlays
"""

__version__ = "1.0.0"
