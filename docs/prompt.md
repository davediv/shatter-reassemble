Build "Shatter & Reassemble" — a standalone browser app. Empty directory, from scratch.

WHAT IT IS
A finger snap shatters the entire camera feed into glass shards that fall and pile up. A
clap sucks every shard back into place, frame-perfect. The reassembly is why people watch
to the end, so it must be flawless.

TRACKING FOUNDATION — build and verify before any effect code
1. getUserMedia({ video: { width: 1280, height: 720, frameRate: 60 } }), fall back to 30.
2. Upload the video to a GL texture each frame via texImage2D from the <video> element.
3. HandLandmarker: GPU delegate, runningMode "VIDEO", numHands 2, confidences 0.5/0.5,
   detectForVideo(video, performance.now()).
4. MIRROR: scaleX(-1) on display, mirror landmark x as (1 - x) once in tracking.ts. Verify
   with a skeleton debug overlay before anything else.
5. SMOOTH: One Euro filter per landmark per axis (mincutoff 1.0, beta 0.007, dcutoff 1.0).
   Note: do NOT over-smooth — this app depends on velocity spikes for snap and clap
   detection. Keep a parallel unsmoothed velocity channel for gesture detection only.
6. DEPTH: never use landmark.z. handSpan = pixel distance from landmark 0 to 9, calibrated
   over 60 frames, normalizing every threshold so behavior is distance-invariant.

LANDMARK INDICES: 0 wrist; 1-4 thumb (4 = tip); 5-8 index (5 = MCP, 8 = tip); 9-12 middle
(9 = MCP); 13-16 ring; 17-20 pinky (17 = MCP).

GESTURES NEEDED
- snap: thumb tip (4) and middle tip (12) come within 0.25 handSpan, followed by a middle-
  tip velocity spike above threshold within 120ms. This is finicky — ship a dedicated
  tuning mode showing the two signals as live traces, and tune it before building visuals.
- clap: two hands, palm centroids (landmark 9 each) converging, distance below 0.5 handSpan,
  approach velocity above threshold. Fires once then locks out 400ms.
- openPalm moving through the pile stirs it (capsule colliders on the finger chains).

PERFORMANCE — top priority, above visual ambition
Target: locked 60fps at 1080p on an M1 MacBook Air with 800 active shards; 60fps at 720p on
2019 Intel integrated. Budget (16.6ms): tracking ≤6ms, physics ≤4ms, shard render ≤4ms.
- Physics as a purpose-built 2.5D solver: planar rigid bodies with a fake perspective tilt.
  Sweep-and-prune broadphase on a sorted axis, sequential impulses, 8 iterations, with
  sleeping enabled. Sleeping is what makes 800 bodies affordable once the pile settles.
- Store body state in flat pre-allocated Float32Arrays (x, y, rot, vx, vy, w) in
  structure-of-arrays layout, not an array of objects. Zero allocation per frame.
- Fixed 120Hz substeps with an accumulator; render interpolates. Never let physics run on a
  variable dt or the reassembly timing will drift.
- Shards drawn in ONE instanced draw call with per-instance transform + UV rect. No
  per-shard draw calls at any count.
- Generate the Voronoi fracture ONCE at snap time on the CPU (it is a one-off ~3ms cost,
  acceptable inside the snap's flash frame), then never touch it again.
- Instrument with EXT_disjoint_timer_query_webgl2.
- DEGRADATION LADDER — above a 15ms rolling average, step down: (1) shard count 800→550→350,
  (2) refraction sampling off (flat texture only), (3) bevel geometry off, (4) solver
  iterations 8→5, (5) shadow pass off. Never drop below 60fps.

THE EFFECT
- Fracture: Voronoi decomposition, 300-800 cells, seeds clustered near the snap position so
  the break radiates from the hand.
- Each shard carries its slice of the FROZEN last frame's UVs. This is the trick — the pile
  must be a recognizably shattered image of the room.
- Behind the shards: pure black, with the live person silhouette faintly outlined. The user
  keeps moving in the void while their world lies broken.
- Shards need thickness, beveled edges that catch light, and subtle refraction of what is
  behind them. Flat triangles look like a PNG explosion.
- REASSEMBLY: store each shard's rest transform. Animate with per-shard staggered easing
  (delay proportional to distance from frame center), slight overshoot, and a flash on the
  final shard landing. Cross-fade the frozen frame back to live video over the last 150ms.
  Sub-pixel accuracy on the final pose is mandatory — any misalignment kills the whole app.

DONE WHEN
- Reassembly lands with zero visible seams at 1080p, verified frame-by-frame.
- Stirring shows correct shard-vs-hand collision against the capsule chain.
- 800 shards falling simultaneously holds 60fps.

RECORDING
MediaRecorder on canvas.captureStream(60), VP9 12Mbps, 16:9 and 9:16 center-crop. Hotkey R
toggles and downloads; H hides debug UI. Under 1ms/frame.

README documents measured budget, shard count achieved, and ladder thresholds.
