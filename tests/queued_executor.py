"""A background executor that holds every job until a test runs it.

Shared by the model-level and the end-to-end tests: both want the review's
asynchrony to be real -- a request goes out, a completion arrives later -- while
choosing when the completion lands. Serial and in submission order, like the
real one, so a job that is cancelled is genuinely still waiting.
"""

from collections.abc import Callable

from rat_tracer.background import Job


class QueuedJob:
    def __init__(self, run: Callable[[], None]):
        self.run = run
        self.cancelled = False

    def cancel(self) -> bool:
        if self.cancelled:
            return False
        self.cancelled = True
        return True


class QueuedExecutor:
    """Satisfies :class:`~rat_tracer.background.BackgroundExecutor`, on demand."""

    def __init__(self):
        self.jobs: list[QueuedJob] = []

    def submit[T](
        self,
        work: Callable[[], T],
        on_done: Callable[[T], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> Job:
        def run() -> None:
            try:
                result = work()
            except Exception as error:
                if on_error is not None:
                    on_error(error)
                return
            if on_done is not None:
                on_done(result)

        job = QueuedJob(run)
        self.jobs.append(job)
        return job

    def pump(self) -> None:
        """Run everything queued, and everything queued while running it."""
        for _ in range(20):
            pending, self.jobs = self.jobs, []
            if not pending:
                return
            for job in pending:
                if not job.cancelled:
                    job.run()

    def stop(self) -> None:
        self.pump()
