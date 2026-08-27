"""What the review tells the UI.

Its own module because it is the outbound contract of the whole review, not of
one class in it: :class:`~rat_tracer.video_review.VideoReview` reports
navigation through it and
:class:`~rat_tracer.review_modes.ProblemReportMode` reports the answers it was
waiting for, and neither should have to know about the other to do so.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class ReviewListener:
    """What the review tells the UI, as plain callbacks.

    Deliberately not Qt signals: the review must stay drivable without an event
    loop. ``VideoMasker`` supplies callbacks that emit the real signals.

    Three, not one. ``changed`` is everything the UI can work out for itself by
    looking -- where the video is, what the controls should show, whether a
    repaint is due. A finished write cannot be worked out by looking: by the
    time anything repaints, a frame is simply stored or not, with no record of
    which write just landed or whether it succeeded. So the mark outcomes stay
    separate and carry the frame index the researcher is told about.
    """

    #: Something the UI displays has moved, and a render may be due with it.
    #: May arrive on the cumulative pass's thread.
    changed: Callable[[], None] = lambda: None
    #: A frame is safely on disk -- its index is shown in the confirmation.
    mark_stored: Callable[[int], None] = lambda _index: None
    #: A frame could *not* be stored; the researcher must be told.
    mark_failed: Callable[[int], None] = lambda _index: None
