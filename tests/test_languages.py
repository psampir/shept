"""Tests for shept.languages — the whisper table and the fuzzy matcher.

These must run without PyGObject: nothing here imports `gi`.
"""

from __future__ import annotations

import pytest

from shept.environment import default_language
from shept.languages import (
    AUTO,
    LANGUAGES,
    Language,
    fuzzy_score,
    language_by_code,
    language_name,
    search,
)

#: The upstream `g_lang` map in whisper.cpp v1.8.4 has ids 0..99.
UPSTREAM_COUNT = 100


# ------------------------------------------------------------------ the table


def test_the_table_matches_the_upstream_count():
    assert len(LANGUAGES) == UPSTREAM_COUNT


def test_no_duplicate_codes():
    codes = [language.code for language in LANGUAGES]

    assert len(codes) == len(set(codes))


def test_codes_are_lowercase_two_or_three_letters():
    for language in LANGUAGES:
        assert language.code.isalpha(), language.code
        assert language.code.islower(), language.code
        assert language.code.isascii(), language.code
        assert 2 <= len(language.code) <= 3, language.code


def test_auto_is_not_part_of_the_table():
    assert AUTO.code == "auto"
    assert AUTO not in LANGUAGES
    assert all(language.code != "auto" for language in LANGUAGES)


@pytest.mark.parametrize(
    ("code", "name"),
    [
        ("pl", "Polish"),
        ("en", "English"),
        ("ja", "Japanese"),
        ("zh", "Chinese"),
        ("yue", "Cantonese"),
        ("haw", "Hawaiian"),
    ],
)
def test_well_known_codes_are_present(code, name):
    assert language_by_code(code) == Language(code, name)


def test_language_by_code_ignores_case_and_padding():
    assert language_by_code(" PL ") == Language("pl", "Polish")
    assert language_by_code("auto") == AUTO


@pytest.mark.parametrize("code", ["xx", "", "english", "auto!"])
def test_language_by_code_returns_none_for_unknown_codes(code):
    assert language_by_code(code) is None


def test_language_name_falls_back_to_the_code():
    assert language_name("yue") == "Cantonese"
    assert language_name("auto") == "Auto-detect"
    assert language_name("xx") == "xx"


# --------------------------------------------------------------- fuzzy_score


def test_empty_query_matches_everything():
    assert fuzzy_score("", "Polish") == 1
    assert fuzzy_score("   ", "anything at all") == 1


def test_a_non_subsequence_does_not_match():
    assert fuzzy_score("zzz", "Polish") == 0
    assert fuzzy_score("abc", "Polish") == 0
    assert fuzzy_score("x", "") == 0


def test_matching_is_case_insensitive():
    assert fuzzy_score("POL", "Polish") > 0
    assert fuzzy_score("pol", "POLISH") > 0


def test_prefix_beats_word_start_beats_scattered():
    prefix = fuzzy_score("hai", "Haitian Creole")
    word_start = fuzzy_score("cre", "Haitian Creole")
    scattered = fuzzy_score("ata", "Haitian Creole")

    assert prefix > word_start > scattered > 0


def test_a_consecutive_run_beats_a_scattered_match():
    consecutive = fuzzy_score("iti", "Haitian Creole")
    scattered = fuzzy_score("ata", "Haitian Creole")

    assert consecutive > scattered > 0


def test_wide_gaps_cost_more_than_narrow_ones():
    assert fuzzy_score("ae", "a----e") > fuzzy_score("ae", "a----------e")


def test_shorter_texts_win_when_the_match_is_the_same():
    assert fuzzy_score("ca", "Catalan") > fuzzy_score("ca", "Catalan Spanish")


def test_a_real_match_never_scores_zero():
    # Contrived: every penalty applies, but it is still a subsequence.
    assert fuzzy_score("a", "b" * 40 + "a") >= 1


# ------------------------------------------------------------------- search


def _codes(results: list[Language]) -> list[str]:
    return [language.code for language in results]


def test_search_ranks_polish_first_for_pol():
    assert search("pol")[0] == Language("pl", "Polish")


def test_search_ranks_ukrainian_first_for_ukr():
    assert search("ukr")[0] == Language("uk", "Ukrainian")


@pytest.mark.parametrize("query", ["ja", "jpn", "japanese"])
def test_search_finds_japanese_by_name_or_code(query):
    assert search(query)[0].code == "ja"


def test_search_finds_hawaiian_by_its_three_letter_code():
    assert search("haw")[0].code == "haw"


def test_search_returns_nothing_for_a_non_match():
    assert search("zzz") == []


def test_search_accepts_an_empty_query_as_the_full_list():
    results = search("")

    assert results[0] == AUTO
    assert len(results) == UPSTREAM_COUNT + 1
    assert len({language.code for language in results}) == len(results)


def test_empty_query_puts_favourites_then_preferred_then_the_rest():
    results = search("", favourites=["uk", "ja"], preferred="pl")

    assert _codes(results[:4]) == ["auto", "uk", "ja", "pl"]
    rest = results[4:]
    assert [language.name for language in rest] == sorted(
        language.name for language in rest
    )
    assert "uk" not in _codes(rest)
    assert "ja" not in _codes(rest)
    assert "pl" not in _codes(rest)


def test_empty_query_ignores_duplicate_and_unknown_favourites():
    results = search("", favourites=["uk", "xx", "uk", "auto"], preferred="uk")

    assert _codes(results[:2]) == ["auto", "uk"]
    assert _codes(results).count("uk") == 1
    assert len(results) == UPSTREAM_COUNT + 1


def test_a_favourite_that_is_not_a_language_is_ignored():
    plain = search("pol")
    unknown = search("pol", favourites=["xx", "auto", ""])

    assert _codes(unknown) == _codes(plain)


def test_preferred_is_ignored_when_it_is_not_a_language():
    assert _codes(search("pol", preferred="xx")) == _codes(search("pol"))


def test_a_favourite_breaks_a_tie_in_favour_of_the_user():
    # "es" scores the same for Maltese and a few others, so without a bonus
    # the alphabetical tie-break puts them in name order.
    plain = _codes(search("es"))
    starred = _codes(search("es", favourites=["mt"]))

    assert plain.index("mt") > plain.index("zh")
    assert starred.index("mt") < starred.index("zh")
    assert len(plain) == len(starred)


def test_the_preferred_language_breaks_a_tie_too():
    plain = _codes(search("es"))
    preferred = _codes(search("es", preferred="mt"))

    assert plain.index("mt") > plain.index("fo")
    assert preferred.index("mt") < preferred.index("fo")


def test_query_results_are_ordered_by_score():
    results = search("an")

    scores = [
        max(fuzzy_score("an", language.name), fuzzy_score("an", language.code))
        for language in results
    ]
    assert scores == sorted(scores, reverse=True)
    assert all(score > 0 for score in scores)


def test_equal_scores_are_alphabetical():
    results = search("ia")

    ranked: dict[int, list[str]] = {}
    for language in results:
        score = max(fuzzy_score("ia", language.name), fuzzy_score("ia", language.code))
        ranked.setdefault(score, []).append(language.name)
    for names in ranked.values():
        assert names == sorted(names)


# ------------------------------------------------------- default_language


@pytest.fixture
def locale_env(monkeypatch):
    for var in ("LC_ALL", "LC_MESSAGES", "LANG"):
        monkeypatch.delenv(var, raising=False)

    def set_locale(value: str) -> None:
        monkeypatch.setenv("LANG", value)

    return set_locale


def test_default_language_uses_the_new_table_for_seven_letter_locales(locale_env):
    locale_env("haw_US.UTF-8")

    assert default_language() == "haw"


def test_default_language_still_falls_back_to_auto(locale_env):
    locale_env("xx_YY.UTF-8")

    assert default_language() == "auto"
