"""Decoding, position, and when a frame needs drawing again.

This holds no Qt dependency (no QObject, QThread, QTimer, QVideoFrame) so it
can be driven and unit-tested directly, without an event loop, real threads,
or monkeypatching Qt's scheduling primitives.

Its remit stops at the frame: which one is displayed, when it must be redrawn,
and the index and timestamp that name it. *What* is drawn over it belongs to
the active mode (:mod:`rat_tracer.review_modes`), which this asks rather than
decides -- so adding a way of looking at the video does not touch this file,
and this file has no opinion about which way is in use.

The state behind all of that is private. Whether a render is due is decided
here, from flags that only :meth:`MaskRenderCore.render_now` may clear, and an
outside assignment to any of them silently costs a repaint; what callers get
instead are the questions they actually ask -- :attr:`~MaskRenderCore.position`,
:attr:`~MaskRenderCore.raw_frame`, :attr:`~MaskRenderCore.showing_judged_frame`.
"""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

from rat_tracer.coverage import CoverageHistory
from rat_tracer.review_modes import ReviewMode

logger = getLogger(__name__)


class FrameCapture(Protocol):
    """Structural type for the subset of cv2.VideoCapture used here."""

    def frame_count(self) -> int: ...

    def fps(self) -> float: ...

    def read(self, frame_idx: int) -> ndarray | None: ...


@dataclass(frozen=True)
class RenderOutcome:
    """Result of :meth:`MaskRenderCore.render_now`.

    ``should_emit`` distinguishes "nothing changed, leave the current frame
    on screen" (False) from "show something now" (True). When ``should_emit``
    is True, ``image`` is either the frame to display or ``None`` for an
    empty/placeholder frame -- three states that a plain ``ndarray | None``
    can't express on its own.
    """

    should_emit: bool
    image: ndarray | None = None


_NOTHING_TO_RENDER = RenderOutcome(should_emit=False)


class MaskRenderCore:
    """Decides what is shown, and what may be marked, as the pass progresses."""

    def __init__(self, mode: ReviewMode | None = None) -> None:
        # Outlives every video: the modes are built around this one object, so
        # closing a video clears it rather than replacing it.
        self._history = CoverageHistory()
        self._mode = mode
        self._clear()

    def _clear(self) -> None:
        """Put everything a video owns back to its opening value.

        One place, used by both ``__init__`` and :meth:`reset`, because the
        fields that matter most here are the ones easiest to forget: the
        pending-render flags decide whether the *next* video gets its first
        repaint at all, and a stale ``_pending_position`` makes a seek to the
        position the last video was left at look like no seek at all.
        """
        self._video_path: Path | None = None
        self._video_key: str | None = None
        self._cap: FrameCapture | None = None
        self._total_frames = 0
        self._fps = 0.0
        self._position = 0.0
        self._pending_position: float | None = None
        self._render_pending = False
        self._force_render = False
        self._playing = True
        # The frame as decoded, before the mode drew on it. Every overlay
        # mutates in place, and a marked frame must be stored raw -- annotated
        # pixels are unusable as training data (FR-12).
        self._raw_frame: ndarray | None = None
        self._rendered_frame_index: int | None = None
        # Whether the active mode drew everything it currently has. False means
        # something is still outstanding -- an unprocessed frame, a detection
        # not back yet -- and a repaint is due when it arrives.
        self._overlay_complete = False

    def reset(self) -> None:
        self._history.clear()
        self._clear()

    def open(self, cap: FrameCapture, video_path: Path) -> None:
        self._cap = cap
        self._video_path = video_path
        self._total_frames = int(cap.frame_count())
        self._fps = cap.fps()
        if self._fps <= 0:
            logger.warning("Video reports no frame rate; timestamps will read 00:00:00")

    def identify(self, video_key: str) -> None:
        """Name the open video by its content fingerprint.

        Separate from :meth:`open` because the fingerprint costs a full read of
        the file, so it is paid by the cumulative pass on its own thread rather
        than by whoever opened the video. Until it lands nothing can be marked,
        since a mark is stored under this key. Assigning one reference is all
        that crosses the thread boundary.
        """
        self._video_key = video_key

    # --- what is being looked at --------------------------------------------

    @property
    def history(self) -> CoverageHistory:
        """The cumulative track, which the pass appends to as it runs."""
        return self._history

    @property
    def mode(self) -> ReviewMode | None:
        return self._mode

    @property
    def video_open(self) -> bool:
        return self._cap is not None

    @property
    def video_path(self) -> Path | None:
        return self._video_path

    @property
    def video_key(self) -> str | None:
        return self._video_key

    @property
    def total_frames(self) -> int:
        return self._total_frames

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> float:
        """Where the frame on screen is, as a fraction of the video."""
        return self._position

    @property
    def raw_frame(self) -> ndarray | None:
        """The displayed frame as decoded, before the mode drew on it."""
        return self._raw_frame

    @property
    def rendered_frame_index(self) -> int | None:
        return self._rendered_frame_index

    @property
    def overlay_complete(self) -> bool:
        return self._overlay_complete

    @property
    def showing_judged_frame(self) -> bool:
        """Whether the active mode's full answer for the current frame is up.

        True only while stopped on a frame that has been drawn with nothing
        outstanding. Problem reporting mode marks on this, which is why every
        stored mark is something the researcher actually looked at and no
        metadata field has to assert it.
        """
        return (
            self._cap is not None
            and not self._playing
            and self._rendered_frame_index == self.current_frame_index
            and self._overlay_complete
        )

    # --- what changes it ----------------------------------------------------

    def adopt_mode(self, mode: ReviewMode) -> None:
        """Set the opening mode, with none of :meth:`set_mode`'s consequences.

        For construction only, before there is a video or a frame on screen:
        there is no previous mode to leave and nothing to force a repaint of.
        Going through ``set_mode`` there would leave a render already marked as
        pending, which swallows the first real one.
        """
        self._mode = mode

    def set_playing(self, value: bool) -> bool:
        """Returns True if the caller should schedule a render now."""
        self._playing = value
        return self.frame_ready()

    def set_mode(self, mode: ReviewMode) -> bool:
        """Show the video the way *mode* does from now on.

        Returns True if the caller should schedule a render now. Swapping the
        mode changes only what is drawn over the frame -- the recorded coverage
        is untouched, so going back brings the track return with nothing lost.
        """
        if mode is self._mode:
            return False
        if self._mode is not None:
            self._mode.left()
        self._mode = mode
        mode.entered()
        # The frame on screen was drawn for the other mode, and neither the
        # position nor the coverage has changed -- so ask for a repaint
        # explicitly rather than relying on the usual change detection.
        self._force_render = True
        return self._schedule_render()

    def _processed_position(self) -> float:
        return float(len(self._history) - 1) / self._total_frames

    @property
    def repaint_due(self) -> bool:
        """Whether the pass has produced something the screen is not showing.

        Reads state without touching it, so the pass's own thread may ask it
        directly and only wake the UI when the answer is yes. Applying the
        answer is :meth:`frame_ready`'s job, on the thread that owns this.
        """
        if self._total_frames == 0:
            return False
        processed = self._processed_position()
        if self._playing:
            return self._cap is not None and self._pending_position != processed
        return not self._overlay_complete and self._position < processed

    def frame_ready(self) -> bool:
        """Fold the pass's progress into the render decision.

        Returns True if the caller should schedule a render now. Playback is
        this too: playing means following the processed frontier, so a frame
        appended by the pass is what advances the position.
        """
        logger.debug(
            "frame_ready: %d/%d, playing: %s, overlay_complete: %s",
            len(self._history) - 1,
            self._total_frames,
            self._playing,
            self._overlay_complete,
        )
        if not self.repaint_due:
            return False
        if self._playing:
            return self.set_position(self._processed_position())
        return self._schedule_render()

    def force_repaint(self) -> None:
        """Redraw even though neither the position nor the coverage moved.

        Needed when what the *mode* can draw has changed under a still frame --
        a detection arriving for the frame on screen, say.
        """
        self._force_render = True

    def set_position(self, new_value: float) -> bool:
        """Returns True if the caller should schedule a render now."""
        if self._pending_position == new_value:
            return False
        self._pending_position = new_value
        logger.debug("set_position: requested %.3f", new_value)
        return self._schedule_render()

    def _schedule_render(self) -> bool:
        if self._render_pending:
            return False
        self._render_pending = True
        return True

    def render_now(self) -> RenderOutcome:
        """Produce the frame to display now, if anything has changed.

        Always clears the render-pending flag, even if nothing is rendered,
        so a later ``set_position``/``frame_ready`` call can schedule again.
        """
        try:
            if (
                not self._force_render
                and self._position == self._pending_position
                and not self._repaint_needed()
            ):
                logger.debug("render_now: nothing to render")
                return _NOTHING_TO_RENDER
            new_value = self._pending_position
            if new_value is None:
                # A forced repaint can be the very first render of a video, so
                # there is no requested position yet -- repaint where we are.
                new_value = self._position
                self._pending_position = new_value
            self._position = new_value
            return self._produce_frame(new_value)
        finally:
            self._render_pending = False
            self._force_render = False

    def _repaint_needed(self) -> bool:
        """True when the frame on screen is missing something it could show.

        The answer is the active mode's: only it knows what it is waiting for.
        """
        if self._mode is None:
            return False
        return self._mode.repaint_needed(
            self.position_to_frame_index(self._position), self._overlay_complete
        )

    def _produce_frame(self, position: float) -> RenderOutcome:
        capture = self._cap
        if not capture:
            logger.warning("_produce_frame: no video capture available for rendering")
            return RenderOutcome(should_emit=True, image=None)
        frame_idx = self.position_to_frame_index(position)
        r = capture.read(frame_idx)
        if r is None:
            logger.warning("_produce_frame: cannot read frame %d", frame_idx)
            return RenderOutcome(should_emit=True, image=None)
        img: ndarray = r
        self._rendered_frame_index = frame_idx
        # Kept before the mode draws anything: a marked frame must be stored
        # without annotation.
        self._raw_frame = img.copy()
        self._overlay_complete = self._mode.draw(img, frame_idx) if self._mode is not None else True
        return RenderOutcome(should_emit=True, image=img)

    # --- naming a frame -----------------------------------------------------

    def position_to_frame_index(self, position: float) -> int:
        frame_idx = int(position * self._total_frames)
        # A slider at its maximum maps one past the last frame; clamping keeps
        # every reported index one a researcher and a technician can both name.
        last = self._total_frames - 1
        return max(0, min(frame_idx, last)) if last >= 0 else 0

    def frame_index_to_position(self, frame_index: int) -> float:
        """Inverse of :meth:`position_to_frame_index`, for frame stepping.

        Aims at the middle of the frame's slice of the slider so that rounding
        cannot land the result back on a neighbouring frame.
        """
        if self._total_frames <= 0:
            return 0.0
        return (frame_index + 0.5) / self._total_frames

    @property
    def current_position(self) -> float:
        """Where the researcher is now, seek included.

        ``position`` only catches up when a render happens, so it answers
        "what is on screen" rather than "where are we". Frame stepping,
        detection requests and the index readout all need the latter: a
        detection must be requested for the frame just seeked to, not for the
        one still being displayed.
        """
        return self._pending_position if self._pending_position is not None else self._position

    @property
    def current_frame_index(self) -> int:
        return self.position_to_frame_index(self.current_position)

    def timestamp_ms(self, frame_index: int) -> int:
        """Position of *frame_index* in the recording, in milliseconds.

        Derived from the frame index rather than read back from the capture:
        the capture's position is a side effect of the last read, which the
        detector and the renderer both perform.
        """
        if self._fps <= 0:
            return 0
        return int(frame_index / self._fps * 1000)

    def step_frame(self, delta: int) -> float | None:
        """Position *delta* frames away, or None when it would leave the video.

        Returns the new position without applying it; the caller pauses
        playback and seeks, so that stepping is meaningful in the first place.
        """
        if self._cap is None or self._total_frames <= 0:
            return None
        target = self.current_frame_index + delta
        if target < 0 or target >= self._total_frames:
            return None
        return self.frame_index_to_position(target)
