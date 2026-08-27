"""The two ways of looking at a video, each owning its own behaviour.

The application answers two unrelated questions, and answering both at once is
what made the UI complicated:

* :class:`CoverageMode` -- *how far has the background pass got?* It draws the
  cumulative track, which is never erased, so a red region is the union of
  every detection so far.
* :class:`ProblemReportMode` -- *is this one frame's detection right?* It hides
  that history and draws only what the detector produced for the frame the
  researcher stopped on, and it is the only mode in which a frame can be
  marked as a failure.

A single frame cannot be judged from the cumulative view at all, which is why
these are modes rather than layers. Each is self-contained: what it draws, when
it needs redrawing, and what it makes possible. Choosing between them belongs
to :class:`~rat_tracer.video_review.VideoReview`; neither mode knows the
other exists, and neither imports Qt.
"""

from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

from rat_tracer.bad_frames import Detection, MarkRequest
from rat_tracer.coverage import CoverageHistory
from rat_tracer.paint import apply_red_mask, draw_detection_boxes

logger = getLogger(__name__)


class VideoView(Protocol):
    """What a mode may know about the video being looked at.

    Satisfied by :class:`~rat_tracer.mask_render_core.MaskRenderCore`, which
    owns the position and the decoding; a mode only reads from it.
    """

    @property
    def video_path(self) -> Path | None: ...

    @property
    def video_key(self) -> str | None: ...

    @property
    def current_frame_index(self) -> int: ...

    @property
    def rendered_frame_index(self) -> int | None: ...

    @property
    def raw_frame(self) -> ndarray | None: ...

    @property
    def playing(self) -> bool: ...

    def timestamp_ms(self, frame_index: int) -> int: ...


class ReviewMode(Protocol):
    """One way of looking at the video.

    A mode draws over the decoded frame and says whether it has drawn
    everything it has: an incomplete overlay is what makes the renderer come
    back once the missing part -- a processed frame, a detection -- arrives.
    """

    def draw(self, image: ndarray, frame_index: int) -> bool:
        """Draw over *image* in place; return True if nothing is outstanding."""
        ...

    def repaint_needed(self, frame_index: int, drawn: bool) -> bool:
        """Whether what is on screen is now missing something it could show."""
        ...

    def entered(self) -> None:
        """Called when this mode becomes the active one."""
        ...

    def left(self) -> None:
        """Called when another mode takes over."""
        ...


class CoverageMode:
    """Shows how far the background pass has got.

    The track is cumulative and never erased -- per the README that is exactly
    what makes key moments findable, since a region painted red is one the
    subject has visited at some point.
    """

    def __init__(self, history: CoverageHistory):
        self.history = history

    def draw(self, image: ndarray, frame_index: int) -> bool:
        if not self.history.contains(frame_index):
            logger.debug("CoverageMode: frame %d is not processed yet", frame_index)
            return False
        apply_red_mask(image, self.history[frame_index])
        return True

    def repaint_needed(self, frame_index: int, drawn: bool) -> bool:
        # Waiting on the cumulative pass: repaint once it reaches this frame.
        return not drawn and self.history.contains(frame_index)

    def entered(self) -> None:
        pass

    def left(self) -> None:
        pass


class DetectionSource(Protocol):
    """Computes a frame's detection somewhere the UI thread is not."""

    @property
    def model_id(self) -> str: ...

    def request(self, frame_index: int, image: ndarray) -> None: ...


class MarkStorage(Protocol):
    """Stores and removes marked frames somewhere the UI thread is not.

    ``is_marked`` is the exception: it is answered immediately, because the
    control shows whether the frame on screen is stored every time the
    researcher moves.
    """

    def is_marked(self, video_key: str, frame_index: int) -> bool: ...

    def store(self, request: MarkRequest) -> None: ...

    def remove(self, video_key: str, frame_index: int, video_stem: str) -> None: ...


class ProblemReportMode:
    """Shows what the detector decided about the frame on screen, and lets the
    researcher say it is wrong.

    The cumulative track is not drawn here at all: it is the union of every
    detection so far, so a red region says nothing about *this* frame.
    """

    def __init__(
        self,
        view: VideoView,
        detection: DetectionSource | None = None,
        storage: MarkStorage | None = None,
    ):
        self.view = view
        self.detection = detection
        self.storage = storage
        # Cached per frame index, so returning to a frame does not pay for
        # inference twice and "the detector found nothing" stays
        # distinguishable from "no answer yet".
        self._detections: dict[int, Detection] = {}
        # Frames already asked about, so seeking back and forth does not
        # re-request an answer that is already on its way.
        self._requested: set[int] = set()
        # Frames whose write or removal has been handed to storage but has not
        # finished. Storage is asynchronous, so ``frame_marked`` stays false
        # meanwhile, and without this the control would look available for a
        # second click.
        self._in_flight: set[int] = set()
        self._last_mark: tuple[str, int, str] | None = None

    # --- drawing ------------------------------------------------------------

    def draw(self, image: ndarray, frame_index: int) -> bool:
        detection = self._detections.get(frame_index)
        if detection is None:
            logger.debug("ProblemReportMode: no detection for frame %d yet", frame_index)
            return False
        draw_detection_boxes(image, detection.boxes)
        return True

    def repaint_needed(self, frame_index: int, drawn: bool) -> bool:
        # Waiting on the detector: repaint once its answer lands.
        return not drawn and frame_index in self._detections

    def entered(self) -> None:
        self.request_detection()

    def left(self) -> None:
        pass

    def forget_video(self) -> None:
        self._detections.clear()
        self._requested.clear()
        self._in_flight.clear()
        self._last_mark = None

    # --- detection ----------------------------------------------------------

    def request_detection(self) -> None:
        """Ask for the displayed frame's detection, at most once per frame."""
        source = self.detection
        if source is None or self.view.playing:
            return
        frame_index = self.view.current_frame_index
        if frame_index in self._detections or frame_index in self._requested:
            return
        if self.view.rendered_frame_index != frame_index or self.view.raw_frame is None:
            # Not on screen yet; the render that puts it there asks again.
            return
        logger.debug("request_detection: frame %d", frame_index)
        self._requested.add(frame_index)
        # A copy, because detection runs later while rendering keeps mutating
        # its own frame.
        source.request(frame_index, self.view.raw_frame.copy())

    def detection_ready(self, frame_index: int, detection: Detection) -> bool:
        """Record an answer; returns True if the displayed frame changed."""
        self._detections[frame_index] = detection
        # An answer for a frame the researcher has already left is kept for
        # their return, but nothing on screen changes.
        return frame_index == self.view.current_frame_index

    def detection_failed(self, frame_index: int) -> None:
        """Let the frame be asked about again.

        Otherwise one failure -- a model that will not load, say -- would leave
        the control disabled for that frame for the rest of the session.
        """
        self._requested.discard(frame_index)

    def detection_for(self, frame_index: int) -> Detection | None:
        return self._detections.get(frame_index)

    # --- marking ------------------------------------------------------------

    def can_mark(self, showing_judged_frame: bool) -> bool:
        """Whether the control acts on the frame on screen at all.

        Covers both directions -- storing a judged frame and withdrawing a
        stored one. Only a frame whose storage is mid-flight is off limits, so
        a second click cannot queue a second write or removal.
        """
        return (
            showing_judged_frame
            and self.storage is not None
            and self.view.video_key is not None
            and self.view.current_frame_index not in self._in_flight
        )

    @property
    def frame_marked(self) -> bool:
        """Whether the displayed frame is already stored."""
        if self.storage is None or self.view.video_key is None:
            return False
        return self.storage.is_marked(self.view.video_key, self.view.current_frame_index)

    def mark(self, showing_judged_frame: bool) -> bool:
        """Store the frame on screen. Returns False if nothing was stored.

        Reads the current frame; it never navigates, so the position and the
        recorded coverage are untouched.
        """
        request = self._build_mark_request(showing_judged_frame)
        if request is None:
            return False
        assert self.storage is not None
        self._last_mark = (request.video_key, request.frame_index, request.video_stem)
        self._in_flight.add(request.frame_index)
        self.storage.store(request)
        return True

    def unmark(self, showing_judged_frame: bool) -> bool:
        """Withdraw the frame on screen. Returns False if nothing was removed.

        The five-second Undo is the correction for a misclick, but it cannot
        help a researcher looking straight at a frame they marked earlier.
        Nothing has to be navigated to for that -- the frame is on screen and
        the control already says it is stored.
        """
        if not self.can_mark(showing_judged_frame) or not self.frame_marked:
            return False
        video_path = self.view.video_path
        video_key = self.view.video_key
        assert video_path is not None and video_key is not None
        frame_index = self.view.current_frame_index
        if self._last_mark is not None and self._last_mark[1] == frame_index:
            # Undo would now have nothing left to remove.
            self._last_mark = None
        self._remove(video_key, frame_index, video_path.stem)
        return True

    def undo(self) -> bool:
        """Withdraw the most recent mark. Returns False if there is none."""
        if self._last_mark is None or self.storage is None:
            return False
        video_key, frame_index, stem = self._last_mark
        self._last_mark = None
        self._remove(video_key, frame_index, stem)
        return True

    def _remove(self, video_key: str, frame_index: int, stem: str) -> None:
        assert self.storage is not None
        self._in_flight.add(frame_index)
        self.storage.remove(video_key, frame_index, stem)

    def storage_finished(self, frame_index: int) -> None:
        """Storage is done with *frame_index*, whichever way it went."""
        self._in_flight.discard(frame_index)

    def forget_last_mark(self) -> None:
        self._last_mark = None

    def _build_mark_request(self, showing_judged_frame: bool) -> MarkRequest | None:
        """Describe the frame on screen as a record to store, if it may be.

        Returns None -- rather than raising -- for every reason a mark cannot
        be made, because the caller's job is only to relay the answer to the
        control, not to distinguish them.
        """
        if not self.can_mark(showing_judged_frame):
            logger.debug("mark: nothing judged on screen")
            return None
        video_path = self.view.video_path
        video_key = self.view.video_key
        assert video_path is not None and video_key is not None
        frame_index = self.view.current_frame_index
        image = self.view.raw_frame
        detection = self._detections.get(frame_index)
        if image is None or detection is None:
            logger.warning("mark: no raw frame or detection for %d", frame_index)
            return None
        if self.frame_marked:
            logger.debug("mark: frame %d is already marked", frame_index)
            return None
        return MarkRequest(
            image=image.copy(),
            video_path=video_path,
            video_key=video_key,
            frame_index=frame_index,
            timestamp_ms=self.view.timestamp_ms(frame_index),
            detection=detection,
            model_id=self.detection.model_id if self.detection is not None else "unknown",
        )
