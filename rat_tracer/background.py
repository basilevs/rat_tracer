"""Running the review's slow work somewhere the researcher is not waiting.

Two things must not happen on the thread that answers the keyboard: inference
for the frame being judged, and writing a marked frame to disk. Neither is Qt's
business, so this is the whole of what the model layer knows about threads --
hand over a callable, hear back about it afterwards.

Serial on purpose. A mark and its retraction must not interleave into a state
where the files and the index disagree, and a stale detection must not overwrite
a fresh one, so an implementation runs one job at a time in submission order.
Completions are reported on the thread that submitted the job, which is what
keeps the review a single-threaded object despite all of this.

The long cumulative pass is deliberately *not* run here: it lasts as long as the
video does, and a serial executor running it would starve every other job. It
gets a thread of its own, and :meth:`rat_tracer.video_review.VideoReview.process_video`
is the whole of its interface.

:class:`InlineExecutor` satisfies the contract by running each job immediately,
which is what lets a test drive an entire review with no threads at all.
"""

from collections.abc import Callable
from logging import getLogger
from typing import Protocol

logger = getLogger(__name__)


class Job(Protocol):
    """A submitted piece of work, in case the answer stops being wanted."""

    def cancel(self) -> bool:
        """Abandon the job; True only if it had not started and now never will.

        A job that is already running cannot be stopped, and one that has
        finished has nothing to abandon -- both answer False, so a caller that
        has to undo its own bookkeeping knows whether to.
        """
        ...


class BackgroundExecutor(Protocol):
    """Runs submitted callables one at a time, off the caller's thread."""

    def submit[T](
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> Job:
        """Run *work*; report its result or its failure back to the caller.

        Exactly one of *on_done* and *on_error* is called, on the submitting
        thread. An implementation with nowhere to defer the work may call it
        before ``submit`` returns, so a caller must not assume the returned
        :class:`Job` is recorded anywhere by the time a completion runs.
        """
        ...


class _Finished:
    """The job handle for work that is already over by the time it is returned."""

    def cancel(self) -> bool:
        return False


_FINISHED = _Finished()


class InlineExecutor:
    """Runs each job immediately, on the calling thread.

    The honest default for a review with nowhere to put the work: a test drives
    one with this and every completion has already happened by the time the call
    returns. Not for the application, where it would block the UI thread on
    inference.
    """

    def submit[T](
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> Job:
        try:
            result = work()
        except Exception as error:
            logger.exception("background job failed")
            if on_error is not None:
                on_error(error)
            return _FINISHED
        if on_done is not None:
            on_done(result)
        return _FINISHED
