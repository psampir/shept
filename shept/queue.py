"""GTK-free batch queue model.

The window owns one `Queue` and mirrors it into widgets; which file runs when,
what counts as finished and how a batch is exported all live here so they can
be tested without PyGObject. The actual ffmpeg/whisper-cli run stays in
`shept.app`, because it needs pipes and the GTK main loop.

Jobs are transcribed one at a time on purpose: decoding and whisper-cli are
already CPU/GPU-bound, so running two of them would only make both slower and
the progress bars meaningless.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from shept.transcribe import EXPORTERS, Segment


class JobState(Enum):
    """Where one file is in the queue's life cycle."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States a job never leaves on its own.
FINISHED_STATES = frozenset({JobState.DONE, JobState.FAILED, JobState.CANCELLED})


@dataclass(eq=False)
class Job:
    """One file in the queue, plus everything learned while transcribing it.

    `eq=False` keeps the default identity hash: jobs are mutable rows that the
    UI indexes by object (a job is "the same job" even after its state, error
    and segments changed), and a value-based `__eq__` would make them
    unhashable in the first place.
    """

    path: Path
    state: JobState = JobState.PENDING
    segments: list[Segment] = field(default_factory=list)
    error: str | None = None
    progress: float = 0.0

    @property
    def display_name(self) -> str:
        """The file name alone — rows show names, tooltips show full paths."""
        return self.path.name

    @property
    def is_finished(self) -> bool:
        """True once the job has reached a state it will not leave by itself."""
        return self.state in FINISHED_STATES


class Queue:
    """An ordered, duplicate-free list of `Job`s.

    Nothing here starts or stops anything: the caller asks for `next_pending()`
    and reports back with `mark_running()` / `mark_done()` / `mark_failed()`,
    which keeps the state machine testable without a main loop.
    """

    def __init__(self) -> None:
        self._jobs: list[Job] = []
        #: Paths the last `add()` call refused to queue, because they do not
        #: exist or are not regular files. The caller reports them to the user.
        self.last_skipped: list[Path] = []

    @property
    def jobs(self) -> tuple[Job, ...]:
        """A read-only snapshot of the queue, in insertion order."""
        return tuple(self._jobs)

    def add(self, paths: Iterable[Path]) -> list[Job]:
        """Append `paths` to the queue and return the jobs that were added.

        Duplicates are matched on the resolved path, so `a.mp3`, `./a.mp3` and
        a symlink to it are the same job; a duplicate already in the queue (or
        earlier in the same call) is ignored silently. Paths that are not
        existing regular files are skipped and listed in `last_skipped`.
        """
        self.last_skipped = []
        added: list[Job] = []
        known = {job.path for job in self._jobs}
        for raw in paths:
            try:
                resolved = Path(raw).expanduser().resolve()
            except OSError:
                self.last_skipped.append(Path(raw))
                continue
            if resolved in known:
                continue
            if not resolved.is_file():
                self.last_skipped.append(Path(raw))
                continue
            job = Job(path=resolved)
            self._jobs.append(job)
            known.add(resolved)
            added.append(job)
        return added

    def next_pending(self) -> Job | None:
        """The oldest job still waiting, or None when the batch is finished."""
        for job in self._jobs:
            if job.state is JobState.PENDING:
                return job
        return None

    # ----------------------------------------------------------- transitions

    def mark_running(self, job: Job) -> Job:
        """Start `job`, discarding anything a previous attempt produced."""
        job.state = JobState.RUNNING
        job.segments.clear()
        job.error = None
        job.progress = 0.0
        return job

    def mark_done(self, job: Job, segments: Iterable[Segment] | None = None) -> Job:
        """Finish `job`, optionally replacing its segments in one go."""
        job.state = JobState.DONE
        job.error = None
        job.progress = 1.0
        if segments is not None:
            job.segments[:] = list(segments)
        return job

    def mark_failed(self, job: Job, error: str) -> Job:
        """Finish `job` with a message; its partial segments are kept."""
        job.state = JobState.FAILED
        job.error = error
        return job

    def mark_cancelled(self, job: Job) -> Job:
        """Abandon `job` at the user's request, keeping what it produced."""
        job.state = JobState.CANCELLED
        return job

    # --------------------------------------------------------------- queries

    def counts(self) -> dict[JobState, int]:
        """Number of jobs per state, with every state present (0 if unused)."""
        counts = {state: 0 for state in JobState}
        for job in self._jobs:
            counts[job.state] += 1
        return counts

    @property
    def is_running(self) -> bool:
        """True while a job is being transcribed."""
        return any(job.state is JobState.RUNNING for job in self._jobs)

    @property
    def has_pending(self) -> bool:
        """True while at least one job is still waiting its turn."""
        return any(job.state is JobState.PENDING for job in self._jobs)

    # --------------------------------------------------------------- removal

    def remove(self, job: Job) -> bool:
        """Drop `job` from the queue; return True when it was there."""
        for index, candidate in enumerate(self._jobs):
            if candidate is job:
                del self._jobs[index]
                return True
        return False

    def clear_finished(self) -> list[Job]:
        """Remove every finished job and return the ones that were removed."""
        removed = [job for job in self._jobs if job.is_finished]
        self._jobs = [job for job in self._jobs if not job.is_finished]
        return removed


def export_all(jobs: Iterable[Job], fmt: str, dest_dir: Path) -> list[Path]:
    """Write every job that produced segments to `<dest_dir>/<stem>.<fmt>`.

    Jobs without segments (pending, or failed before the first segment) are
    skipped, so exporting a half-run queue never creates empty files. Two jobs
    with the same file name get ` (2)`, ` (3)`, … appended to the stem instead
    of overwriting each other. Returns the paths written, in queue order.
    """
    formatter = EXPORTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"unknown export format: {fmt}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    taken: set[Path] = set()
    for job in jobs:
        if not job.segments:
            continue
        target = _free_path(dest_dir / f"{job.path.stem}.{fmt}", taken)
        target.write_text(formatter(job.segments), encoding="utf-8")
        taken.add(target)
        written.append(target)
    return written


def _free_path(path: Path, taken: set[Path]) -> Path:
    """Return `path`, or `stem (2).ext`, `stem (3).ext`, … when it is used."""
    if path not in taken:
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if candidate not in taken:
            return candidate
        counter += 1
