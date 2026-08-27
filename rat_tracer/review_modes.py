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

Problem reporting mode is the one with work to do -- inference for the frame on
screen, and writing the frames judged wrong. It runs that work on the executor
it is given and keeps the whole lifecycle to itself: what is worth asking, what
is already on its way, what may not be clicked twice while a write is in
flight. None of that is anybody else's to sequence, so none of it is on its
API; what comes out is :meth:`ProblemReportMode.request_detection`,
:meth:`ProblemReportMode.mark`, :meth:`ProblemReportMode.unmark` and
:meth:`ProblemReportMode.undo`, and a notification when something has landed.
"""

from logging import getLogger
from pathlib import Path
from typing import Protocol

from numpy import ndarray

from rat_tracer.background import BackgroundExecutor, InlineExecutor, Job
from rat_tracer.bad_frames import BadFrameStore, Detection, MarkRequest
from rat_tracer.coverage import CoverageHistory
from rat_tracer.frame_detector import FrameDetector
from rat_tracer.paint import apply_red_mask, draw_detection_boxes
from rat_tracer.review_listener import ReviewListener

logger = getLogger(__name__)

#: What a retraction needs to name: the video, the frame, and the file stem.
type Retraction = tuple[str, int, str]


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


class ProblemReportMode:
    """Shows what the detector decided about the frame on screen, and lets the
    researcher say it is wrong.

    The cumulative track is not drawn here at all: it is the union of every
    detection so far, so a red region says nothing about *this* frame.

    Everything slow it needs -- inference for the frame on screen, and writing
    the frames judged wrong -- goes to the executor it was given, and it keeps
    the whole lifecycle of both: what is worth asking about, what is already on
    its way, what a cancelled request means, and what may not be clicked again
    until a write has landed. Nobody outside sequences any of that; the listener
    is told when something has landed, and the answer is there to be looked at.
    """

    def __init__(
        self,
        view: VideoView,
        executor: BackgroundExecutor | None = None,
        listener: ReviewListener | None = None,
        detector: FrameDetector | None = None,
        store: BadFrameStore | None = None,
    ):
        self.view = view
        self._executor = executor if executor is not None else InlineExecutor()
        self._listener = listener if listener is not None else ReviewListener()
        self._detector = detector
        self._store = store
        # Cached per frame index, so returning to a frame does not pay for
        # inference twice and "the detector found nothing" stays
        # distinguishable from "no answer yet".
        self._detections: dict[int, Detection] = {}
        # Frames already asked about, so seeking back and forth does not
        # re-request an answer that is already on its way.
        self._requested: set[int] = set()
        #: The one request still out, so a newer frame can abandon it.
        self._outstanding: tuple[int, Job] | None = None
        # Frames whose write or removal has been handed to storage but has not
        # finished. Storage is asynchronous, so ``frame_marked`` stays false
        # meanwhile, and without this the control would look available for a
        # second click.
        self._in_flight: set[int] = set()
        self._last_mark: Retraction | None = None
        self._prewarmed = False

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
        self._prewarm()

    def left(self) -> None:
        pass

    def forget_video(self) -> None:
        self._abandon_outstanding()
        self._detections.clear()
        self._requested.clear()
        self._in_flight.clear()
        self._last_mark = None

    # --- detection ----------------------------------------------------------

    def request_detection(self) -> None:
        """Ask about the frame on screen, if that is worth doing.

        Does nothing when there is no detector, when playing, when the answer is
        already known or already on its way, or when the frame is not on screen
        yet -- in which case the render that puts it there asks again. Called
        after every render, so "worth doing" is decided here rather than by the
        caller.
        """
        detector = self._detector
        if detector is None or self.view.playing:
            return
        frame_index = self.view.current_frame_index
        if frame_index in self._detections or frame_index in self._requested:
            return
        if self.view.rendered_frame_index != frame_index or self.view.raw_frame is None:
            return
        logger.debug("request_detection: frame %d", frame_index)
        self._requested.add(frame_index)
        # A copy, because detection runs later while rendering keeps mutating
        # its own frame.
        image = self.view.raw_frame.copy()
        self._abandon_outstanding()
        self._outstanding = (
            frame_index,
            self._executor.submit(
                lambda: detector.detect(image),
                on_done=lambda detection: self._detection_arrived(frame_index, detection),
                on_error=lambda _error: self._detection_lost(frame_index),
            ),
        )

    def _prewarm(self) -> None:
        """Load the model before a frame is waiting on it, once per review.

        The first inference in a process costs seconds while later ones cost a
        fraction of one. The queue is serial, so paying it as the mode is
        entered puts it ahead of the request it would otherwise delay.
        """
        detector = self._detector
        if self._prewarmed or detector is None:
            return
        self._prewarmed = True
        self._executor.submit(detector.prewarm)

    def _abandon_outstanding(self) -> None:
        """Drop a request the researcher has already seeked past.

        Seeking produces requests faster than inference answers them, and they
        are looking at the newest frame -- so an outstanding request is
        cancelled rather than left queued in front of the one that matters. A
        cancelled frame goes back to being unasked, so returning to it asks
        again.
        """
        outstanding = self._outstanding
        self._outstanding = None
        if outstanding is None:
            return
        frame_index, job = outstanding
        if job.cancel():
            self._requested.discard(frame_index)

    def _detection_arrived(self, frame_index: int, detection: Detection) -> None:
        logger.debug("detection: frame %d, %d box(es)", frame_index, len(detection.boxes))
        # An answer for a frame the researcher has already left is kept for
        # their return; the render that follows redraws only if it is this one,
        # which ``repaint_needed`` is what decides.
        self._detections[frame_index] = detection
        self._listener.changed()

    def _detection_lost(self, frame_index: int) -> None:
        """Let the frame be asked about again.

        Without this, one failure -- a model that will not load, say -- would
        leave the control disabled for that frame for the rest of the review.
        """
        logger.warning("No detection for frame %d", frame_index)
        self._requested.discard(frame_index)
        self._listener.changed()

    # --- marking ------------------------------------------------------------

    def can_mark(self, showing_judged_frame: bool) -> bool:
        """Whether the control acts on the frame on screen at all.

        Covers both directions -- storing a judged frame and withdrawing a
        stored one. Only a frame whose storage is mid-flight is off limits, so
        a second click cannot queue a second write or removal.
        """
        return (
            showing_judged_frame
            and self._store is not None
            and self.view.video_key is not None
            and self.view.current_frame_index not in self._in_flight
        )

    @property
    def frame_marked(self) -> bool:
        """Whether the displayed frame is already stored."""
        if self._store is None or self.view.video_key is None:
            return False
        return self._store.is_marked(self.view.video_key, self.view.current_frame_index)

    def mark(self, showing_judged_frame: bool) -> None:
        """Store the frame on screen, if it may be.

        Does nothing if nothing may be stored. Reads the current frame; it never
        navigates, so the position and the recorded coverage are untouched. The
        frame counts as in flight from here until the write lands, which is what
        stops a second click queueing a second write.
        """
        request = self._build_mark_request(showing_judged_frame)
        if request is None:
            return
        store = self._store
        assert store is not None  # _build_mark_request answers None without one
        frame_index = request.frame_index
        self._last_mark = (request.video_key, frame_index, request.video_stem)
        self._in_flight.add(frame_index)
        self._executor.submit(
            lambda: store.mark(request),
            on_done=lambda _name: self._stored(frame_index),
            on_error=lambda _error: self._not_stored(frame_index),
        )

    def unmark(self, showing_judged_frame: bool) -> None:
        """Withdraw the frame on screen, if it is stored.

        The five-second Undo is the correction for a misclick, but it cannot
        help a researcher looking straight at a frame they marked earlier.
        Nothing has to be navigated to for that -- the frame is on screen and
        the control already says it is stored.
        """
        if not self.can_mark(showing_judged_frame) or not self.frame_marked:
            return
        video_path = self.view.video_path
        video_key = self.view.video_key
        assert video_path is not None and video_key is not None
        frame_index = self.view.current_frame_index
        if self._last_mark is not None and self._last_mark[1] == frame_index:
            # Undo would now have nothing left to remove.
            self._last_mark = None
        self._retract((video_key, frame_index, video_path.stem))

    def undo(self) -> None:
        """Delete everything stored for the most recent mark."""
        if self._last_mark is None or self._store is None:
            return
        last_mark, self._last_mark = self._last_mark, None
        self._retract(last_mark)

    def _retract(self, retraction: Retraction) -> None:
        store = self._store
        assert store is not None  # both callers check
        video_key, frame_index, stem = retraction
        self._in_flight.add(frame_index)
        self._executor.submit(
            lambda: store.retract(video_key, frame_index, stem),
            on_done=lambda _none: self._removed(frame_index),
            on_error=lambda _error: self._not_removed(frame_index),
        )

    def _stored(self, frame_index: int) -> None:
        logger.info("Marked frame %d", frame_index)
        self._in_flight.discard(frame_index)
        self._listener.changed()
        self._listener.mark_stored(frame_index)

    def _not_stored(self, frame_index: int) -> None:
        logger.error("Could not store frame %d", frame_index)
        self._in_flight.discard(frame_index)
        # Undo must not offer to remove a frame that was never written.
        self._last_mark = None
        self._listener.changed()
        self._listener.mark_failed(frame_index)

    def _removed(self, frame_index: int) -> None:
        # The files are gone only once storage has run, so the control is
        # refreshed then rather than when the removal was requested.
        logger.info("Retracted frame %d", frame_index)
        self._in_flight.discard(frame_index)
        self._listener.changed()

    def _not_removed(self, frame_index: int) -> None:
        # Nothing was removed, but the control must not stay disabled for a
        # write that will never finish -- the researcher can try again.
        logger.error("Could not retract frame %d", frame_index)
        self._in_flight.discard(frame_index)
        self._listener.changed()

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
            model_id=self._detector.model_id if self._detector is not None else "unknown",
        )
