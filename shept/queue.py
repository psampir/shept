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

from shept.transcribe import EXPORTERS, Segment, write_text_atomic


class JobState(Enum):
    """Where one file is in the queue's life cycle."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


#: States a job never leaves on its own.
FINISHED_STATES = frozenset({JobState.DONE, JobState.FAILED, JobState.CANCELLED})

#: Longest stem, in UTF-8 bytes, that an exported file may use. `NAME_MAX` is
#: 255 bytes almost everywhere; the headroom covers a ` (123)` collision suffix
#: and the extension, so one over-long source name cannot abort a batch export.
MAX_EXPORT_STEM_BYTES = 200


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
    #: Full diagnostics for the row tooltip when `error` is the short version.
    error_details: str | None = None
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

    def begin_next(self, *, cancelled: bool) -> Job | None:
        """Start the oldest pending job, or return None when the batch is over.

        A cancel that lands between two jobs must not consume the next one:
        marking it RUNNING and then abandoning the batch would leave it
        unreachable — no retry, and `next_pending()` skips it from then on.
        So the cancel flag is checked before anything is mutated.
        """
        if cancelled:
            return None
        job = self.next_pending()
        if job is None:
            return None
        return self.mark_running(job)

    # ----------------------------------------------------------- transitions

    def mark_running(self, job: Job) -> Job:
        """Start `job`, discarding anything a previous attempt produced."""
        job.state = JobState.RUNNING
        job.segments.clear()
        job.error = None
        job.error_details = None
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

    def mark_failed(
        self,
        job: Job,
        error: str,
        details: str | None = None,
    ) -> Job:
        """Finish `job` with a message; its partial segments are kept.

        `error` is the short reason a row's subtitle shows; `details` keeps the
        full child output, which belongs in the tooltip.
        """
        job.state = JobState.FAILED
        job.error = error
        job.error_details = details or None
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
    """Write every successfully finished job to `<dest_dir>/<stem>.<fmt>`.

    Only `DONE` jobs with segments are written: a failed or cancelled file may
    hold a partial transcript, and "Export all" must not pass half a recording
    off as the result. Jobs without segments (pending, or failed before the
    first segment) are skipped too, so a half-run queue never creates empty
    files. Files already in `dest_dir` are never overwritten — the new one gets
    ` (2)`, ` (3)`, … appended — and over-long stems are cut so `NAME_MAX`
    cannot abort the batch midway. Returns the paths written, in queue order.
    """
    formatter = EXPORTERS.get(fmt)
    if formatter is None:
        raise ValueError(f"unknown export format: {fmt}")

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    taken: set[Path] = set()
    for job in exportable_jobs(jobs):
        stem = export_stem(job.path.stem)
        target = _free_path(dest_dir / f"{stem}.{fmt}", taken)
        write_text_atomic(target, formatter(job.segments))
        taken.add(target)
        written.append(target)
    return written


def exportable_jobs(jobs: Iterable[Job]) -> list[Job]:
    """The jobs "Export all" writes: finished successfully, with text."""
    return [job for job in jobs if job.state is JobState.DONE and job.segments]


def export_stem(name: str) -> str:
    """Return `name` cut to the export byte budget, never mid-character."""
    encoded = name.encode("utf-8")
    if len(encoded) <= MAX_EXPORT_STEM_BYTES:
        return name
    return encoded[:MAX_EXPORT_STEM_BYTES].decode("utf-8", errors="ignore")


def _free_path(path: Path, taken: set[Path]) -> Path:
    """Return `path`, or `stem (2).ext`, `stem (3).ext`, … when it is used."""
    if not _is_used(path, taken):
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem} ({counter}){path.suffix}")
        if not _is_used(candidate, taken):
            return candidate
        counter += 1


def _is_used(path: Path, taken: set[Path]) -> bool:
    """True when this batch wrote `path` already, or it exists on disk."""
    if path in taken:
        return True
    try:
        return path.exists()
    except OSError:
        # An unreadable directory entry is not proof the path is free; the
        # write itself will report the real problem.
        return False
