# Bad Frame Capture — implementation plan

**Status:** Agreed
**Last updated:** 2026-07-31
**Source:** [prd.md](prd.md)

Records the feasibility work done before implementation, the decisions taken on
the PRD's open questions, and the module plan that follows from them.

## Feasibility findings

Measured on a 4-core CPU-only container, 640×480 / 25 fps synthesized clip.
The published model could not be fetched (no Hugging Face access from the agent
environment), so detection timings used `YOLO("yolo11n.yaml")` — real inference
path and real timings, meaningless boxes.

| Question | Result |
| --- | --- |
| FR-10, on-demand detection for an arbitrary frame | **145–271 ms** per frame *while* a batched coverage pass saturated all cores. The 1 s NFR holds with margin. |
| Cost of a second `YOLO` instance | ~100 MB RSS. |
| Cold start | First inference in the process costs ~5.7 s (torch warmup). Normally paid by the coverage pass; problem reporting mode must not be the first to pay it. |
| Frame seek + decode | 6–50 ms. |
| Atomic PNG write of a marked frame | ~8 ms; verified byte-identical to the decoded frame on read-back. |
| `index.jsonl` replay dedupe | Survives a restart; retraction removes files and appends a `retract` row. |
| `.zip` archive to Desktop → Documents → home | Works; `QStandardPaths` resolves without a `QApplication`, so the CLI needs no GUI. |
| `video_key` crc32 | ~3 GiB/s — recomputation is affordable, reuse is still preferred. |

### The momentary mask is not reconstructible

`CoverageHistory[i]` is the cumulative OR, and the per-frame presence mask in
[`presence_frames`](../../../rat_tracer/paint.py#L51-L96) comes from a MOG2
background subtractor whose state is sequential. Reproducing it for a seek
target would mean replaying hundreds of preceding frames — seconds, against a
1 s bound. Problem reporting mode can therefore show only what the detector
output for that frame, which settles the PRD's "boxes or momentary mask"
question in favour of boxes.

## Decisions

| PRD open question / gap | Decision |
| --- | --- |
| Scope of this round | All three phases. |
| Overlay in problem reporting mode | **Box outlines.** Shows an offset or oversized box, which a fill would hide. |
| Detector model | **Separate `YOLO` instance** for on-demand inference. ~100 MB is worth not waiting on the coverage pass's in-flight batch. |
| Keyboard shortcuts (FR-2) | **F2** marks, **Left/Right** step one frame. Layout-independent, so `B` / `,` / `.` are dropped — those move under a Russian layout. |
| Archive format | **`.zip`.** |
| `model_id` in the sidecar | HF ref by default; under `RAT_TRACER_MODEL`, the local path plus a content hash (`file:/…/last.pt#crc32=…`), since `last.pt` is overwritten every training run. |
| Filename stem collision (FR-15) | Keep the PRD's accepted overwrite, but **log a warning** at mark time so the loss is visible. |
| ±N neighbouring frames | No, per the PRD's own proposal. Revisit after Phase 1. |
| Unmarking a stored frame | **Allowed from the mark control**, logged as an ordinary `retract` row. The PRD rules pruning out of scope, but its reason is that marked frames cannot be *found* again — which does not apply to the frame already on screen, whose control already says it is stored. Consequence to accept: `retract` rows now mix deliberate withdrawals with five-second misclick corrections, so the retraction-rate metric measures both. |

## Code gaps to address on the way

1. `VideoMasker._cap` is set only in `_set_video`, never in `__init__`, so
   [`time_text`](../../../rat_tracer/ui.py#L199) raises `AttributeError` before a
   video is opened. Reproduced live. FR-7 rewrites this readout anyway.
2. No frame-rate source in `MaskRenderCore`; `time_text` reads
   `CAP_PROP_POS_MSEC` off the shared, stateful `VideoCapture`. FR-7's index and
   timestamp should both derive from `frame_index` and `CAP_PROP_FPS` inside the
   core, where they are testable.
3. `cv2.VideoCapture` is not thread-safe and is already used from the render
   path. The detector gets its own capture, or is handed the decoded frame.
4. `apply_red_mask` mutates the decoded frame in place — FR-12 needs the raw
   copy taken before any overlay.
5. `video_key` is computed inside `CoverageComputer` and not exposed; capture
   must reuse it rather than recompute.
6. No `__version__`; `app_version` comes from `importlib.metadata`. The PRD's
   example `0.5.1` is fictional — the package is at `0.0.1`.
7. `QStandardPaths.AppDataLocation` depends on `applicationName`, which the app
   never sets (it currently derives from `argv[0]`). Set it in one shared place,
   and do **not** set `organizationName` — Linux would nest `rat_tracer/rat_tracer`.
8. No model is available in the agent environment or CI, so the detector must be
   injectable, in the humble-object style already used by `MaskRenderCore`.

## Module plan

- `rat_tracer/bad_frames.py` — Qt-free storage: root resolution
  (`RAT_TRACER_BAD_FRAMES` → `AppDataLocation`), atomic image and sidecar
  writes, `index.jsonl` append and replay, `(video_key, frame_index)` dedupe.
- `rat_tracer/frame_detector.py` — on-demand single-frame inference behind an
  injectable interface, with its own model instance and a prewarm hook.
- `rat_tracer/mask_render_core.py` — problem-reporting-mode state, box overlay
  decision, frame index and timestamp derivation. Stays Qt-free.
- `rat_tracer/ui.py` — Qt wiring only: detector thread, save worker, new
  properties and slots for the mode, the mark control and the toast.
- `rat_tracer/Main.qml` — mode toggle, stateful mark control, frame-step
  buttons, index/timestamp readout, undo toast, `Shortcut` bindings.
- `rat_tracer/collect.py` — `rat_tracer-collect` console entry point.
- `rat_tracer/translations.py` — `en` and `ru` for every new string.

## Verified end to end

All three phases are implemented. Beyond the unit and integration suites
(`make test`, 83 offline tests), the whole flow was driven once through the
real stack — real video, real ultralytics, real Qt, real storage — with
locally-built weights standing in for the published model, which cannot be
downloaded in the agent environment:

- Warm detection after a seek: **0.11 s**, comfortably inside the 1 s bound.
- Marking stores pixels byte-identical to the decoded frame, does not move the
  position, and the control turns to "marked".
- Undo deletes both files and leaves `mark`, `retract` in the index.
- Resuming playback leaves the mode; stepping moves exactly one frame.
- `rat_tracer-collect` archives the tree without touching the source.
- QML: both toast paths render with the frame index, controls are disabled
  with no video open, and the slider no longer takes keyboard focus (a focused
  Slider would consume Left/Right and move the position by a slider step at the
  same time as the frame-step shortcut moved it by one frame).

## Still open

- **Reference machine for the 1 s bound.** The PRD leaves it TBD. Measurements
  are from a 4-core container; a cold model load on a CPU-only field laptop is
  the case most likely to break the bound.
- **The first frame after entering the mode costs the model load.** Measured
  3.6 s end to end, against 0.11 s for every frame after it. The detector
  prewarms on its worker thread, but that thread only starts when the mode is
  entered, so the first request queues behind the load. Starting the worker
  when the video opens would hide it entirely, at the cost of loading a second
  model for researchers who never enter the mode. Worth deciding once the
  reference machine is known, since it is exactly the "cold load on a slow
  laptop" case the PRD flags.
- **Near-duplicate frames are not deduplicated on ingest.** The PRD's headline
  risk is annotation cost, and neighbouring frames of the same failure are
  highly correlated. `duplicates.py` already exists in the repo; whether ingest
  should use it, or the app should cap marks per video, is still the open
  decision the PRD records.
- **The detection cache is unbounded** for the lifetime of an open video (one
  small entry per frame visited in the mode). Not a practical concern at
  session scale, but it is not evicted.
- **Undo covers the most recent mark only**, matching the PRD's five-second
  window. Marking a second frame makes the first one unreachable, as designed.
- The pre-existing mypy baseline (10 errors, `Results | Tensor` unions from
  ultralytics) is untouched.
