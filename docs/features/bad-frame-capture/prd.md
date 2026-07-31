# Bad Frame Capture

**Status:** Draft
**Author:** Vasili Gulevich
**Last updated:** 2026-07-30

## Summary

Let a researcher see what the detector decided about a single frame, mark that
frame as a failure with one click, and package every frame they have marked —
across all videos and experiments — into one archive file without typing a path.

## Problem

Rat Tracer paints a cumulative red presence mask over the video
([`apply_red_mask`](../../../rat_tracer/paint.py#L153)). While seeking
through a recording — the normal way of locating key moments in a traversal — a
researcher keeps running into evidence that the detector is wrong: a red splat in
an empty corridor (false positive), a rat that leaves no trail (false negative),
or a mask that is offset or far too large (bad bounding box).

Today that observation is lost. Getting the failure back into the model requires
the training workflow in [README.md](../../../README.md) — run `track.py`, sift
through `runs/detect/track*/track_loss`, delete the good images, run
`track_to_frames.py`, annotate, `split.py`, `train.py`. That is lab work for a
technician, not something a researcher can do mid-experiment in the field. So bad
frames are noticed, sighed at, and forgotten, and the model never improves on the
cases that actually break it.

## Goals

- The researcher can judge a single frame's detection on its own, with the
  accumulated history hidden.
- Once in problem reporting mode, marking a bad frame costs one action and no
  typing. Reaching that mode is a deliberate switch, not part of the per-frame
  cost.
- One position, one slider. The same control that finds key moments in the
  experiment finds detection defects; the researcher never has to think about
  where "the other" position is.
- Marked frames survive restarts and accumulate across sessions, videos and
  experiments in a single location the application manages.
- The researcher can produce a single archive of everything marked so far with
  minimal effort — critically, **without having to know or type the storage path**.
- A misclick is visibly confirmed and reversible, so it does not silently poison
  the training set.
- What the technician receives drops into the existing annotation → `split.py` →
  `train.py` pipeline without renaming or re-keying.

## Non-Goals

- **No in-app annotation or box correction.** The researcher says *"this is
  wrong"*; deciding *what* is wrong and drawing the correct box stays lab work.
- **No failure-type classification by the researcher** (false positive vs.
  negative vs. bad box). The sidecar records what the model produced, so the
  technician can infer it.
- **No bug-report text, severity, or comment field.**
- **No defined delivery channel.** Mail, cloud, USB stick — out of scope by
  decision. The feature ends at "one archive file exists"; how it travels is the
  researcher's business.
- **No second position or defect-review timeline.** Searching for a detection
  defect is the same navigation as searching for a key moment, on the same slider.
  No "return to where you were" behaviour, no marker list that seeks independently.
- **No automatic upload from the field.** Field machines may be offline.
- **No changes to training, `train.py`, or the published model.**
- **No automatic mining of failures** — that is `find_worst_annotations.py`.

## User Stories

**Story:** As a field researcher, I want to mark the frame on screen as a
detection failure in one action, so that it can be used for retraining later
without me stopping to file a report.

**Acceptance criteria:**

- [ ] Given a detection for the current frame is displayed, when the researcher
      clicks "Mark bad frame" or presses its shortcut once, then that frame is
      captured and no dialog, prompt or file chooser appears.
- [ ] Given the video is playing, when the researcher looks at the mark control,
      then it is disabled — playback shows no detections, so there is nothing to
      have judged.
- [ ] Given a frame is marked, when the save completes, then the position does not
      jump and the recorded coverage is unchanged — marking reads the current
      frame, it does not navigate.
- [ ] Given a frame that is already marked is displayed, when the researcher looks
      at the mark control, then it shows as marked, so a duplicate is never
      attempted in the first place.
- [ ] Given any sequence of marking and undoing one frame, when it finishes, then
      at most one stored frame exists for it.
- [ ] Given the current frame's detection has not appeared yet, when the researcher
      looks at the mark control, then it is disabled, and it becomes enabled as
      soon as the detection is shown.
- [ ] Given the detection has been shown and found nothing, when the researcher
      looks at the mark control, then it is **enabled** — a missed detection is the
      most important defect to report, not an absent result.
- [ ] Given problem reporting mode is not active, when the researcher looks at the
      mark control, then it is disabled or not shown at all — no detection is on
      screen to have judged, so entering the mode is a precondition of marking.
- [ ] Given no video is open, when the researcher looks at the control, then it is
      disabled.

**Story:** As a field researcher, I want to see what the detector decided about
this one frame, with the accumulated track hidden, so that I can tell whether the
detection is actually wrong before marking it.

**Acceptance criteria:**

- [ ] Given a video is open, when the researcher switches to problem reporting
      mode, then the cumulative history is hidden and only the current frame's own
      detection is drawn over it.
- [ ] Given problem reporting mode is active, when the researcher navigates to any
      frame, then that frame's own detection is shown — including frames the
      cumulative pass has not reached yet.
- [ ] Given the video is playing, when any frame is displayed, then no per-frame
      detection is drawn. Detections are shown only for a frame the researcher has
      stopped on.
- [ ] Given problem reporting mode is active, when a frame has no detection, then
      the frame is shown clean — which, per the mark control being enabled, means
      the detector found nothing rather than that an answer is pending.
- [ ] Given problem reporting mode is active, when the researcher leaves it, then
      the cumulative mask returns unchanged, with nothing lost from the history.
- [ ] Given a video is open, when the researcher activates previous-frame or
      next-frame, then the position moves by exactly one frame and playback pauses.
- [ ] Given a frame is displayed in either mode, when the researcher inspects it,
      then its frame index and timestamp are visible.
- [ ] Given the researcher has sought backwards to inspect a defect, when they
      resume playing, then playback continues from where the slider now is — the
      application does not restore an earlier position.

*(Two reasons this mode is required rather than nice-to-have. The mask is
cumulative and never erased — per the README that is exactly what makes key
moments findable, but it means a red region on screen is the union of every
detection so far, so a single frame's detection cannot be judged from it at all.
And because the track is painted after the fact, a failure is usually noticed
seconds after the frame that caused it, hence the frame stepping.)*

**Story:** As a field researcher, I want to be told what I just saved and be able
to undo it, so that a misclick does not end up in the training data.

**Acceptance criteria:**

- [ ] Given a frame is marked, when the save succeeds, then a non-blocking
      confirmation appears showing the frame index.
- [ ] Given the confirmation is visible, when the researcher activates Undo within
      5 seconds, then every file stored for that frame is deleted and a retraction
      row is appended to the index.
- [ ] Given the confirmation is visible, when 5 seconds pass without input, then it
      auto-dismisses without having blocked seeking — and, once Phase 2 adds frame
      stepping, without blocking that either.
- [ ] Given the save fails (disk full, path not writable), when the researcher is
      notified, then the message states in their UI language that the frame was
      **not** saved, and the application keeps running.

**Story:** As a field researcher, I want to package everything I have marked into
one file, so that I can send it to the technician without hunting for a storage
directory.

**Acceptance criteria:**

- [ ] Given frames have been marked across several videos and sessions, when the
      researcher runs the archive command with no arguments, then a single archive
      containing all of them is written to a path that is printed on completion.
- [ ] Given the researcher runs the archive command, when it executes, then they do
      not have to supply, type, or know the storage location.
- [ ] Given the archive is produced, when it is written, then it lands somewhere the
      researcher can find in a file manager without navigating to a hidden or
      OS-internal directory.
- [ ] Given the archive command runs, when it completes, then nothing on the
      researcher's machine has been deleted or moved.
- [ ] Given the archive command is run twice, when the second run completes, then
      it succeeds and both archives exist.

**Story:** As the technician responsible for model quality, I want the archive to
drop straight into the training pipeline, so that collecting a round of frames
costs me one command and no renaming.

**Acceptance criteria:**

- [ ] Given an archive from a researcher, when it is extracted with one command,
      then image files are named `<video-stem>_<frame:06d>.png`, matching
      [`extract_frames`](../../../rat_tracer/video_to_images.py#L17).
- [ ] Given the extracted tree, when labels are placed in `labels/` beside the
      existing `images/`, then `split.py` accepts the directory as a ground-truth
      dir with no restructuring, per its existing lookup in
      [split.py](../../../rat_tracer/split.py#L13-L18).
- [ ] Given archives from two collection rounds, when the technician ingests the
      second, then frames already ingested are identifiable by
      `video_key + frame_index` and are not annotated twice.
- [ ] Given an ingested frame, when the technician opens its sidecar, then it states
      what the model detected on that frame (empty list for a missed detection).
- [ ] Given an ingested archive, when the technician filters by `marked_at`, then
      they can determine which frames are new since the last training round without
      asking the researcher.

## Requirements

### Functional Requirements

**UI** — [Main.qml](../../../rat_tracer/Main.qml)

1. A "Mark bad frame" control in the existing button row, alongside Open / Play /
   timestamp. It is **stateful, not a fire-and-forget button** — it reflects
   whether the frame on screen is already marked (a checkbox shown pre-checked, or
   equivalent), so the researcher can see the answer before acting rather than
   discovering it by pressing. Outside problem reporting mode the control is
   **disabled or hidden**; it is never live-looking but unresponsive.
2. *(Optional)* A keyboard shortcut for the same action (proposed `B`), active
   whenever the window has focus. Not required for the feature to work: marking is
   already behind a mode switch, so the shortcut saves one click on a workflow
   that is not one-click anyway. It also introduces the application's first
   keyboard handling — [Main.qml](../../../rat_tracer/Main.qml) has none today —
   and carries an unresolved layout question, so it is worth building only if it
   costs little.
3. Previous-frame / next-frame controls (proposed `,` and `.`, plus two buttons).
   Stepping pauses playback.
4. A **problem reporting mode** toggle. Entering it pauses playback (see FR-5).
   While active, the cumulative mask is not drawn; the current frame is rendered
   with its own detection only. The mode is
   a display state only — it does not alter the recorded coverage, and it is not
   remembered between sessions.
5. Per-frame detections are never drawn during playback — only for a frame the
   researcher has stopped on. Playback exists to show how far the background pass
   has got, which is a technical detail of this application rather than anything
   about the experiment; overlaying detection results on it would conflate a
   progress indicator with a judgement about model output.
6. The mark control is enabled only while a detection result for the current frame
   is on screen. That is the case only inside problem reporting mode, on a stopped
   frame — so outside the mode, and during playback, the control is disabled or
   hidden per FR-1. Every stored mark is therefore something the researcher looked
   at, so no metadata field asserting that is needed; the enabling condition
   guarantees it.
7. The frame index and timestamp of the displayed frame are shown. Today only an
   HH:MM:SS readout exists ([Main.qml](../../../rat_tracer/Main.qml#L86-L95)); a
   frame index is needed so a researcher and a technician can refer to the same
   frame unambiguously.
8. A non-blocking confirmation showing the frame index, offering Undo for 5 seconds
   and then auto-dismissing. It must never block seeking (or, from Phase 2, frame
   stepping) while visible.
9. Every new string routed through
   [translations.py](../../../rat_tracer/translations.py) with `en` and `ru`
   entries — the field user may not read English.

**Capture** — [ui.py](../../../rat_tracer/ui.py) / new module

10. Problem reporting mode shows the detection for **any** frame the researcher
    navigates to, including frames the cumulative pass has not reached yet.
11. The frame index comes from `MaskRenderCore.position_to_frame_index`
    ([mask_render_core.py:160](../../../rat_tracer/mask_render_core.py#L160)), not
    from a re-derived slider value.
12. The saved image is the **raw decoded frame, without any overlay** — masked or
    box-annotated pixels are unusable as training data.
13. Duplicate detection is keyed on `(video_key, frame_index)`, reusing the crc32
    fingerprint from `video_key`
    ([progress_cache.py:16](../../../rat_tracer/progress_cache.py#L16)), so the same
    physical video marked from different paths or machines deduplicates.

**Storage**

14. Root directory is a **user-level, persistent** location per OS convention
    (`QStandardPaths.AppDataLocation` / XDG data dir), overridable via
    `RAT_TRACER_BAD_FRAMES`.
    Explicitly **not** `tempfile.gettempdir()` — that is where `progress_cache`
    lives, correct there (regenerable cache), fatal here (the only copy of the
    data, wiped on reboot).
15. Layout:

    ```
    bad_frames/
      images/   <video-stem>_<frame:06d>.png   # raw frame — annotate these
      meta/     <video-stem>_<frame:06d>.json  # sidecar, see below
      index.jsonl                              # append-only log of all marks
    ```

    No rendered preview is stored: the technician has the raw frame and the boxes
    in the sidecar, so any overlay they want can be regenerated exactly.

    Names carry no `video_key`, so they stay byte-identical to what
    `extract_frames` produces and stay parseable by `track_to_frames.py`. The cost
    is that two videos sharing a stem collide on the same frame index and one
    image overwrites the other. The stored data remains correctly keyed
    (`video_key + frame_index`); only the filename is ambiguous.

16. Sidecar contents per frame:

    ```json
    {
      "video_path": "/media/exp42/2026-07-30_run3.mp4",
      "video_stem": "2026-07-30_run3",
      "video_key": "1a2b3c4d",
      "frame_index": 4821,
      "timestamp_ms": 160700,
      "marked_at": "2026-07-30T14:22:05Z",
      "model_id": "basilevs83/rat-tracer:rat_tracer.pt",
      "app_version": "0.5.1",
      "detection": {"boxes": [[0.51, 0.33, 0.08, 0.11]], "conf": [0.91]}
    }
    ```

    `boxes` uses normalized `[cx, cy, w, h]` — the same convention as the YOLO
    label files the technician will produce, so no conversion is needed. A frame
    where the detector found nothing has `"boxes": []`, which is the record of a
    false negative and must not be confused with a missing `detection` key.

17. Every mark and every retraction appends a row to `index.jsonl` carrying an
    event type (`mark` / `retract`), `marked_at`, `video_key` and `frame_index`.
    The log is append-only: a retraction deletes the stored files but records a
    `retract` row rather than removing the original, so the rate of retraction
    stays measurable. Duplicate marks cannot arise — FR-1 makes the existing mark
    visible — so mark rows are never inflated. This is the instrumentation backing
    the success metrics below that are derived from the log.
18. Writes are atomic (temp file + `replace`), matching the existing
    `save_progress` pattern, so an interrupted save leaves no half-written PNG.

**Archival**

19. A `rat_tracer-collect` console entry point that, with no arguments, packages
    the whole storage root into one archive and prints the resulting path.
20. The archive lands in a location the researcher can reach from a file manager —
    Desktop or Documents — named
    `rat_tracer_bad_frames_<hostname>_<YYYYMMDD-HHMMSS>.<ext>` (extension per the
    open question on archive format), never in the storage root itself. The
    timestamp keeps successive archives distinct.
21. Archival is non-destructive; it never deletes or moves the source frames.

### Non-Functional Requirements

- Saving a marked frame must not interrupt defect hunting: the researcher can
  navigate to the next frame of interest immediately after marking, without
  waiting for the write to finish.
- A frame's detection appears within 1 s of arriving at that frame. This is a
  hard bound, which makes "the answer is not ready yet" a brief and uncommon state
  rather than a normal one. The mark control being disabled is the only signal it
  needs — no dedicated visual treatment, since the wait is not a state the
  researcher should have to reason about.
  **Reference machine: TBD** — the bound is not testable until the field hardware
  is pinned down, and a cold model load on a CPU-only laptop is the case most
  likely to break it.
- The UI never freezes while waiting: navigating onward must remain possible while
  an answer for the previous frame is still outstanding.
- A save failure must be logged and surfaced, and must never crash the UI or lose
  the researcher's position in the video.
- The whole flow works fully offline. No network call is on the marking path.
- The archive command runs on the researcher's OS with no dependency beyond what
  the installed application already provides — no rsync, no ssh, no cloud client.
- No telemetry leaves the researcher's machine. `index.jsonl` is local data that
  happens to travel inside the archive.

## Success Metrics

Two of the three are measured from `index.jsonl` in received archives; the third
is observed in the technician's own pipeline. No separate dashboard, no external
reporting.

- **Marked frames spread across ≥ 2 videos per collection round**, indicating use
  across experiments rather than a single enthusiastic session.
  *Backed by: FR-17 (`video_key` in every index row).*
- **100% of marked frames reach a training round without manual renaming or
  re-keying.** The only metric not read from `index.jsonl` — it is observed in the
  technician's pipeline. *Backed by the naming and `split.py` compatibility
  criteria in the technician story.*
- **Retraction rate < 10% of marks.** A high rate means the control is too easy to
  hit by accident or the confirmation is unclear. Note this measures only *caught*
  mistakes — a wrong mark not undone within 5 seconds is unlikely to be found
  again, so the true error rate is at least this and cannot be observed.
  *Backed by: FR-17 (`retract` rows are recorded, not erased from the log).*

## Risks

- **A wrong mark is effectively permanent.** Marked frames cannot be found again —
  there is no marked-frame navigation, and locating one exact frame among thousands
  by slider is not realistic. So the 5-second Undo is the only correction, and
  anything missed travels to the technician and gets annotated as though it were a
  real defect. The exposure is bounded by how often a researcher marks in error,
  which the retraction-rate metric can only under-report.
  *Open mitigation question:* if this bites, the cheapest fix is
  jump-to-next/previous-marked-frame on the existing slider, which needs no new
  position concept.
- **Annotation cost becomes the new bottleneck** — making collection effortless
  raises volume, and every collected frame still needs manual YOLO labelling by the
  technician before it is worth anything. A researcher who marks 300 frames in a
  week has moved the problem, not solved it.
  *Open mitigation question:* should the technician-side ingest deduplicate
  near-identical frames before annotation — neighbouring frames of the same failure
  are highly correlated and `duplicates.py` already exists in the repo — or should
  the app cap marks per video? Needs a decision before volume becomes real.

## Open Questions

- **How far should playback and problem reporting separate?** Decided: no
  detections during playback, mark control disabled while playing. Under
  consideration: making them fully separate modes, on the grounds that they answer
  unrelated questions — playback reports how far the background pass has got (a
  technical detail), problem reporting asks whether a given frame's detection is
  right (a model-quality question). Neither concerns the experiment itself.
  Both of the constraints such a split would need already hold unconditionally:
  the position is shared (Non-Goals) and entering problem reporting mode pauses
  (FR-4). So the open part is purely how the UI presents the two — one screen with
  a toggle, or two distinct modes.
- **Boxes or momentary mask in problem reporting mode?** Boxes show what the model
  actually output; the momentary mask is the visual language the researcher has
  been reading all along. Worth judging on real footage rather than on paper.
- Should marking also capture ±N neighbouring frames? More training context per
  mark, but it multiplies the annotation cost that is already the headline risk.
  Proposal: no, revisit after Phase 1.
- Is `B` a sensible shortcut under a Russian keyboard layout, or should it be a
  layout-independent key?
- Reviewing and pruning marked frames in-app is **out of scope**: the 5-second
  Undo is the only practical correction. Reopening this would mean adding a way to
  navigate to marked frames — jump-to-next-mark, or ticks on the slider — neither
  of which the Non-Goals actually forbid, since both move the one shared slider
  rather than introducing a second position. Revisit only if the wrong-mark rate
  turns out to matter in practice.
- Archive format: `.zip` (opens natively on every OS the researcher might use) vs.
  `.tar.gz` (friendlier to the technician's shell). Proposal: `.zip`, since the
  researcher is the one who has to handle it.

## Timeline / Milestones

- **Phase 1 — seeing and marking.** Problem reporting mode (FR-4..6, **FR-10** —
  showing a detection for any frame is the hardest item in the feature and the
  whole mode rests on it), stateful mark control (FR-1), frame index readout
  (FR-7, FR-11), raw frame + sidecar save and storage layout (FR-12, FR-14, FR-15),
  atomic writes (FR-18), `index.jsonl` (FR-17), duplicate detection (FR-13), toast
  with Undo (FR-8), `en`/`ru` strings (FR-9). The keyboard shortcut (FR-2) is
  optional and can slip.
  *The mode ships with the marking, not after it: a researcher cannot judge a
  frame from the cumulative overlay, so marking without it is guesswork.
  Duplicate detection ships with it too — re-marking a frame is possible from the
  first day, and a duplicate costs the technician a second annotation of the same
  image, which the Risks section calls the headline cost.*
- **Phase 2 — finding the right frame.** Frame-step controls (FR-3).
  *Until this ships, the only way to reach a specific frame is the normalized
  slider, which at typical frame rates cannot reliably land on one — so Phase 1
  supports judging a frame, not hunting for one.*
- **Phase 3 — getting it out.** `rat_tracer-collect` archival command (FR-19..21),
  README section covering the researcher's one click and one command.
  *No technician-side tooling: the sidecar and index already let them dedupe by
  `video_key + frame_index` and filter by `marked_at` with ordinary shell work.*
