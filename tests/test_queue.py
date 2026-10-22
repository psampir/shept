"""Tests for shept.queue — the GTK-free batch queue model."""

from __future__ import annotations

import json

import pytest

from shept.queue import (
    FINISHED_STATES,
    Job,
    JobState,
    Queue,
    export_all,
    exportable_jobs,
)
from shept.transcribe import Segment

SEGMENTS = [
    Segment(start_ms=0, end_ms=1_500, text="Dzień dobry"),
    Segment(start_ms=1_500, end_ms=3_000, text="Do widzenia"),
]


@pytest.fixture
def media(tmp_path):
    """Create real files, so `Queue.add()` sees them as existing inputs."""
    directory = tmp_path / "media"
    directory.mkdir()

    def make(name: str):
        path = directory / name
        path.write_bytes(b"RIFF")
        return path

    return make


def _queue_of(*paths) -> Queue:
    """A queue holding `paths`, with the jobs available via `queue.jobs`."""
    queue = Queue()
    queue.add(paths)
    return queue


# ---------------------------------------------------------------------- add


def test_add_creates_pending_jobs_in_order(media):
    first, second = media("a.mp3"), media("b.wav")

    added = _queue_of(first, second).jobs

    assert [job.display_name for job in added] == ["a.mp3", "b.wav"]
    assert all(job.state is JobState.PENDING for job in added)
    assert all(job.progress == 0.0 for job in added)
    assert added[0].path == first.resolve()


def test_add_dedupes_by_resolved_path(media, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = media("talk.mp3")
    queue = Queue()

    added = queue.add([path, path.relative_to(tmp_path), path.resolve(), str(path)])

    assert len(added) == 1
    assert len(queue.jobs) == 1
    assert queue.last_skipped == []


def test_add_ignores_paths_already_in_the_queue(media):
    path = media("talk.mp3")
    queue = _queue_of(path)

    assert queue.add([path]) == []
    assert len(queue.jobs) == 1


def test_add_skips_missing_files_and_reports_them(media, tmp_path):
    present = media("present.mp3")
    missing = tmp_path / "nope.mp3"
    queue = Queue()

    added = queue.add([present, missing, tmp_path / "media"])

    assert [job.display_name for job in added] == ["present.mp3"]
    assert queue.last_skipped == [missing, tmp_path / "media"]
    assert len(queue.jobs) == 1


def test_last_skipped_is_reset_by_each_add(media, tmp_path):
    queue = Queue()
    queue.add([tmp_path / "gone.mp3"])
    queue.add([media("here.mp3")])

    assert queue.last_skipped == []


# ---------------------------------------------------------------- next_pending


def test_next_pending_returns_jobs_in_insertion_order(media):
    queue = _queue_of(media("1.mp3"), media("2.mp3"), media("3.mp3"))

    order = []
    while (job := queue.next_pending()) is not None:
        order.append(job.display_name)
        queue.mark_done(job)

    assert order == ["1.mp3", "2.mp3", "3.mp3"]
    assert queue.next_pending() is None


def test_next_pending_skips_finished_jobs(media):
    queue = _queue_of(
        media("a.mp3"),
        media("b.mp3"),
        media("c.mp3"),
    )
    first, second, third = queue.jobs
    queue.mark_failed(first, "boom")
    queue.mark_cancelled(second)

    assert queue.next_pending() is third


def test_next_pending_on_empty_queue_is_none():
    assert Queue().next_pending() is None


# ------------------------------------------------------------------ begin_next


def test_begin_next_starts_the_oldest_pending_job(media):
    queue = _queue_of(media("a.mp3"), media("b.mp3"))
    first, second = queue.jobs

    job = queue.begin_next(cancelled=False)

    assert job is first
    assert first.state is JobState.RUNNING
    assert second.state is JobState.PENDING


def test_begin_next_after_a_cancel_does_not_consume_the_next_job(media):
    """A cancel between two jobs must leave the next one retryable."""
    queue = _queue_of(media("a.mp3"))
    pending = queue.jobs[0]

    assert queue.begin_next(cancelled=True) is None
    assert pending.state is JobState.PENDING
    assert queue.next_pending() is pending


def test_begin_next_on_empty_queue_is_none():
    assert Queue().begin_next(cancelled=False) is None


# ---------------------------------------------------------------- transitions


def test_mark_running_resets_previous_attempt(media):
    job = Job(path=media("a.mp3"))
    job.segments.extend(SEGMENTS)
    job.error = "old failure"
    job.progress = 0.9

    Queue().mark_running(job)

    assert job.state is JobState.RUNNING
    assert job.segments == []
    assert job.error is None
    assert job.progress == 0.0


def test_mark_done_sets_state_and_progress(media):
    job = Job(path=media("a.mp3"))

    Queue().mark_done(job, SEGMENTS)

    assert job.state is JobState.DONE
    assert job.segments == SEGMENTS
    assert job.progress == 1.0
    assert job.error is None
    assert job.is_finished


def test_mark_done_keeps_streamed_segments_when_none_passed(media):
    job = Job(path=media("a.mp3"))
    job.segments.extend(SEGMENTS)

    Queue().mark_done(job)

    assert job.segments == SEGMENTS


def test_mark_failed_records_message_and_keeps_partial_segments(media):
    job = Job(path=media("a.mp3"))
    job.segments.append(SEGMENTS[0])

    Queue().mark_failed(job, "whisper-cli exited with 3")

    assert job.state is JobState.FAILED
    assert job.error == "whisper-cli exited with 3"
    assert job.error_details is None
    assert job.segments == [SEGMENTS[0]]
    assert job.is_finished


def test_mark_failed_keeps_the_full_details_for_the_tooltip(media):
    job = Job(path=media("a.mp3"))

    Queue().mark_failed(job, "whisper-cli exited with 1", "error: bad model")

    assert job.error == "whisper-cli exited with 1"
    assert job.error_details == "error: bad model"


def test_mark_cancelled_leaves_remaining_jobs_pending(media):
    queue = _queue_of(media("a.mp3"), media("b.mp3"))
    first, second = queue.jobs
    queue.mark_running(first)

    queue.mark_cancelled(first)

    assert first.state is JobState.CANCELLED
    assert first.is_finished
    assert second.state is JobState.PENDING


def test_defaults_match_the_documented_contract(media):
    job = Job(path=media("a.mp3"))

    assert job.state is JobState.PENDING
    assert job.segments == []
    assert job.error is None
    assert job.progress == 0.0
    assert job.display_name == "a.mp3"


def test_finished_states_are_done_failed_cancelled():
    assert FINISHED_STATES == {
        JobState.DONE,
        JobState.FAILED,
        JobState.CANCELLED,
    }


# -------------------------------------------------------------------- queries


def test_counts_reports_every_state(media):
    queue = _queue_of(*(media(f"{n}.mp3") for n in "abcde"))
    a, b, c, d, _e = queue.jobs
    queue.mark_done(a, SEGMENTS)
    queue.mark_running(b)
    queue.mark_failed(c, "nope")
    queue.mark_cancelled(d)

    assert queue.counts() == {
        JobState.PENDING: 1,
        JobState.RUNNING: 1,
        JobState.DONE: 1,
        JobState.FAILED: 1,
        JobState.CANCELLED: 1,
    }


def test_counts_on_empty_queue_is_all_zero():
    assert Queue().counts() == {state: 0 for state in JobState}


def test_is_running_and_has_pending(media):
    queue = _queue_of(media("a.mp3"), media("b.mp3"))
    first, second = queue.jobs

    assert queue.is_running is False
    assert queue.has_pending is True

    queue.mark_running(first)
    assert queue.is_running is True
    assert queue.has_pending is True

    queue.mark_done(first)
    assert queue.is_running is False
    assert queue.has_pending is True

    queue.mark_done(second)
    assert queue.has_pending is False


def test_jobs_is_a_read_only_view(media):
    queue = _queue_of(media("a.mp3"))

    with pytest.raises(AttributeError):
        queue.jobs.append(Job(path=media("b.mp3")))  # type: ignore[attr-defined]


# -------------------------------------------------------------------- removal


def test_remove_drops_one_job(media):
    queue = _queue_of(media("a.mp3"), media("b.mp3"))
    first = queue.jobs[0]

    assert queue.remove(first) is True
    assert [job.display_name for job in queue.jobs] == ["b.mp3"]
    assert queue.remove(first) is False


def test_remove_keeps_order_of_the_rest(media):
    queue = _queue_of(*(media(f"{n}.mp3") for n in "abc"))
    queue.remove(queue.jobs[1])

    assert [job.display_name for job in queue.jobs] == ["a.mp3", "c.mp3"]


def test_clear_finished_removes_only_finished_jobs(media):
    queue = _queue_of(*(media(f"{n}.mp3") for n in "abcd"))
    a, b, c, _d = queue.jobs
    queue.mark_done(a, SEGMENTS)
    queue.mark_failed(b, "boom")
    queue.mark_running(c)

    removed = queue.clear_finished()

    assert [job.display_name for job in removed] == ["a.mp3", "b.mp3"]
    assert [job.display_name for job in queue.jobs] == ["c.mp3", "d.mp3"]


def test_clear_finished_with_nothing_finished_is_a_noop(media):
    queue = _queue_of(media("a.mp3"))

    assert queue.clear_finished() == []
    assert len(queue.jobs) == 1


# ----------------------------------------------------------------- export_all


def test_export_all_writes_each_completed_job(media, tmp_path):
    queue = _queue_of(media("first.mp3"), media("second.wav"))
    done, failed = queue.jobs
    queue.mark_done(done, SEGMENTS)
    queue.mark_failed(failed, "boom")

    written = export_all(queue.jobs, "txt", tmp_path / "out")

    assert written == [tmp_path / "out" / "first.txt"]
    assert written[0].read_text(encoding="utf-8") == "Dzień dobry\nDo widzenia\n"


def test_export_all_skips_jobs_without_segments(media, tmp_path):
    queue = _queue_of(media("pending.mp3"), media("cancelled.mp3"))
    queue.mark_cancelled(queue.jobs[1])

    assert export_all(queue.jobs, "srt", tmp_path) == []
    assert list(tmp_path.glob("*.srt")) == []


def test_export_all_skips_failed_jobs_even_with_partial_segments(media, tmp_path):
    """A partial transcript is not a result; "Export all" only writes DONE."""
    queue = _queue_of(media("done.mp3"), media("failed.mp3"), media("stopped.mp3"))
    done, failed, cancelled = queue.jobs
    queue.mark_done(done, SEGMENTS)
    failed.segments.extend(SEGMENTS)
    queue.mark_failed(failed, "boom")
    cancelled.segments.extend(SEGMENTS)
    queue.mark_cancelled(cancelled)

    written = export_all(queue.jobs, "txt", tmp_path)

    assert [path.name for path in written] == ["done.txt"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["done.txt", "media"]


def test_export_all_skips_the_running_job(media, tmp_path):
    queue = _queue_of(media("running.mp3"), media("done.mp3"))
    running, done = queue.jobs
    queue.mark_running(running)
    running.segments.extend(SEGMENTS)
    queue.mark_done(done, SEGMENTS)

    written = export_all(queue.jobs, "txt", tmp_path)

    assert [path.name for path in written] == ["done.txt"]


def test_exportable_jobs_is_done_with_text(media):
    queue = _queue_of(
        media("done.mp3"),
        media("empty.mp3"),
        media("failed.mp3"),
        media("running.mp3"),
    )
    done, empty, failed, running = queue.jobs
    queue.mark_done(done, SEGMENTS)
    queue.mark_done(empty)
    failed.segments.extend(SEGMENTS)
    queue.mark_failed(failed, "boom")
    queue.mark_running(running)
    running.segments.extend(SEGMENTS)

    assert exportable_jobs(queue.jobs) == [done]


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [
        ("txt", "Dzień dobry\nDo widzenia\n"),
        ("srt", "1\n00:00:00,000 --> 00:00:01,500\nDzień dobry\n\n"),
        ("vtt", "WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nDzień dobry\n\n"),
        ("lrc", "[00:00.00]Dzień dobry\n[00:01.50]Do widzenia\n"),
    ],
)
def test_export_all_matches_the_single_file_exporters(media, tmp_path, fmt, expected):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)

    written = export_all(queue.jobs, fmt, tmp_path)

    assert written == [tmp_path / f"talk.{fmt}"]
    assert written[0].read_text(encoding="utf-8").startswith(expected)


def test_export_all_json_is_the_versioned_schema(media, tmp_path):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)

    written = export_all(queue.jobs, "json", tmp_path)

    payload = json.loads(written[0].read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["segments"][0] == {
        "start_ms": 0,
        "end_ms": 1_500,
        "text": "Dzień dobry",
    }


def test_export_all_handles_name_collisions(tmp_path):
    paths = []
    for directory_name in ("one", "two", "three"):
        directory = tmp_path / directory_name
        directory.mkdir()
        path = directory / "talk.mp3"
        path.write_bytes(b"RIFF")
        paths.append(path)
    queue = _queue_of(*paths)
    for job in queue.jobs:
        queue.mark_done(job, SEGMENTS)

    written = export_all(queue.jobs, "txt", tmp_path / "out")

    assert [path.name for path in written] == [
        "talk.txt",
        "talk (2).txt",
        "talk (3).txt",
    ]
    assert all(
        path.read_text(encoding="utf-8") == "Dzień dobry\nDo widzenia\n"
        for path in written
    )


def test_export_all_creates_the_destination_directory(media, tmp_path):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)
    dest = tmp_path / "nested" / "out"

    written = export_all(queue.jobs, "txt", dest)

    assert dest.is_dir()
    assert written[0].is_file()


def test_export_all_returns_paths_in_queue_order(media, tmp_path):
    queue = _queue_of(*(media(f"{n}.mp3") for n in "abc"))
    for job in queue.jobs:
        queue.mark_done(job, SEGMENTS)

    written = export_all(queue.jobs, "vtt", tmp_path)

    assert [path.stem for path in written] == ["a", "b", "c"]


def test_export_all_rejects_unknown_formats(media, tmp_path):
    queue = _queue_of(media("a.mp3"))

    with pytest.raises(ValueError):
        export_all(queue.jobs, "docx", tmp_path)


def test_export_all_does_not_overwrite_a_file_in_the_destination(media, tmp_path):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)
    dest = tmp_path / "out"
    dest.mkdir()
    existing = dest / "talk.txt"
    existing.write_text("hand-written notes\n", encoding="utf-8")

    written = export_all(queue.jobs, "txt", dest)

    assert [path.name for path in written] == ["talk (2).txt"]
    assert existing.read_text(encoding="utf-8") == "hand-written notes\n"


def test_export_all_does_not_overwrite_in_any_format(media, tmp_path):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)
    (tmp_path / "talk.srt").write_text("keep", encoding="utf-8")

    written = export_all(queue.jobs, "srt", tmp_path)

    assert [path.name for path in written] == ["talk (2).srt"]
    assert (tmp_path / "talk.srt").read_text(encoding="utf-8") == "keep"


def test_export_all_truncates_an_over_long_stem(tmp_path):
    """NAME_MAX is 255 bytes; a 250-character stem must not abort a batch."""
    long_stem = "ł" * 250  # two bytes per character in UTF-8
    job = Job(path=tmp_path / f"{long_stem}.mp3")
    Queue().mark_done(job, SEGMENTS)

    written = export_all([job], "txt", tmp_path / "out")

    assert len(written) == 1
    name = written[0].name
    assert name.isprintable()
    assert len(name.encode("utf-8")) <= 210
    assert written[0].is_file()


def test_export_all_reports_a_write_failure(media, tmp_path, monkeypatch):
    queue = _queue_of(media("talk.mp3"))
    queue.mark_done(queue.jobs[0], SEGMENTS)

    def failing_write(target, _text):
        raise OSError(f"Read-only file system: {target}")

    monkeypatch.setattr("shept.queue.write_text_atomic", failing_write)

    with pytest.raises(OSError, match="Read-only file system"):
        export_all(queue.jobs, "txt", tmp_path / "out")
