"""Every language whisper.cpp can transcribe, and a fuzzy matcher for them.

GTK-free on purpose: `shept.settings` and `shept.environment` import this
module, and both are tested without PyGObject installed.

The table is the one whisper.cpp itself uses, taken from the `g_lang` map in
`src/whisper.cpp` at tag `v1.8.4` (the tag Shept pins). It is the authoritative
list: every code here is accepted by `whisper-cli -l`, and nothing else is.
`auto` is not part of it: it is a Shept-level choice, not a whisper language,
so it lives in `AUTO` and is only offered alongside the real entries.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """One entry of whisper.cpp's language table."""

    code: str
    name: str


#: The "let whisper decide" choice. Not a whisper language, so it is kept out
#: of `LANGUAGES`; the picker and `search()` add it in front of the real ones.
AUTO = Language("auto", "Auto-detect")

#: whisper.cpp v1.8.4, `g_lang` in src/whisper.cpp, in the order whisper's own
#: language ids use (0 = en … 99 = yue). Keep this order: it is the order the
#: upstream table is written in, which makes a diff against a new tag easy.
LANGUAGES: tuple[Language, ...] = (
    Language("en", "English"),
    Language("zh", "Chinese"),
    Language("de", "German"),
    Language("es", "Spanish"),
    Language("ru", "Russian"),
    Language("ko", "Korean"),
    Language("fr", "French"),
    Language("ja", "Japanese"),
    Language("pt", "Portuguese"),
    Language("tr", "Turkish"),
    Language("pl", "Polish"),
    Language("ca", "Catalan"),
    Language("nl", "Dutch"),
    Language("ar", "Arabic"),
    Language("sv", "Swedish"),
    Language("it", "Italian"),
    Language("id", "Indonesian"),
    Language("hi", "Hindi"),
    Language("fi", "Finnish"),
    Language("vi", "Vietnamese"),
    Language("he", "Hebrew"),
    Language("uk", "Ukrainian"),
    Language("el", "Greek"),
    Language("ms", "Malay"),
    Language("cs", "Czech"),
    Language("ro", "Romanian"),
    Language("da", "Danish"),
    Language("hu", "Hungarian"),
    Language("ta", "Tamil"),
    Language("no", "Norwegian"),
    Language("th", "Thai"),
    Language("ur", "Urdu"),
    Language("hr", "Croatian"),
    Language("bg", "Bulgarian"),
    Language("lt", "Lithuanian"),
    Language("la", "Latin"),
    Language("mi", "Maori"),
    Language("ml", "Malayalam"),
    Language("cy", "Welsh"),
    Language("sk", "Slovak"),
    Language("te", "Telugu"),
    Language("fa", "Persian"),
    Language("lv", "Latvian"),
    Language("bn", "Bengali"),
    Language("sr", "Serbian"),
    Language("az", "Azerbaijani"),
    Language("sl", "Slovenian"),
    Language("kn", "Kannada"),
    Language("et", "Estonian"),
    Language("mk", "Macedonian"),
    Language("br", "Breton"),
    Language("eu", "Basque"),
    Language("is", "Icelandic"),
    Language("hy", "Armenian"),
    Language("ne", "Nepali"),
    Language("mn", "Mongolian"),
    Language("bs", "Bosnian"),
    Language("kk", "Kazakh"),
    Language("sq", "Albanian"),
    Language("sw", "Swahili"),
    Language("gl", "Galician"),
    Language("mr", "Marathi"),
    Language("pa", "Punjabi"),
    Language("si", "Sinhala"),
    Language("km", "Khmer"),
    Language("sn", "Shona"),
    Language("yo", "Yoruba"),
    Language("so", "Somali"),
    Language("af", "Afrikaans"),
    Language("oc", "Occitan"),
    Language("ka", "Georgian"),
    Language("be", "Belarusian"),
    Language("tg", "Tajik"),
    Language("sd", "Sindhi"),
    Language("gu", "Gujarati"),
    Language("am", "Amharic"),
    Language("yi", "Yiddish"),
    Language("lo", "Lao"),
    Language("uz", "Uzbek"),
    Language("fo", "Faroese"),
    Language("ht", "Haitian Creole"),
    Language("ps", "Pashto"),
    Language("tk", "Turkmen"),
    Language("nn", "Nynorsk"),
    Language("mt", "Maltese"),
    Language("sa", "Sanskrit"),
    Language("lb", "Luxembourgish"),
    Language("my", "Myanmar"),
    Language("bo", "Tibetan"),
    Language("tl", "Tagalog"),
    Language("mg", "Malagasy"),
    Language("as", "Assamese"),
    Language("tt", "Tatar"),
    Language("haw", "Hawaiian"),
    Language("ln", "Lingala"),
    Language("ha", "Hausa"),
    Language("ba", "Bashkir"),
    Language("jw", "Javanese"),
    Language("su", "Sundanese"),
    Language("yue", "Cantonese"),
)

_BY_CODE: dict[str, Language] = {lang.code: lang for lang in LANGUAGES}
_BY_CODE[AUTO.code] = AUTO

#: A favourite is only worth much when the match itself is close, so these
#: bonuses stay well below what a one-letter-better match is worth.
FAVOURITE_BONUS = 5
PREFERRED_BONUS = 3

#: Characters that start a new word in a language name ("Haitian Creole").
_WORD_BREAKS = " -_/."


def language_by_code(code: str) -> Language | None:
    """Return the entry for `code`, or None when whisper does not know it."""
    if not isinstance(code, str):
        return None
    return _BY_CODE.get(code.strip().lower())


def language_name(code: str) -> str:
    """English name for `code`; unknown codes are shown as they are."""
    language = language_by_code(code)
    return language.name if language is not None else code


def _match_positions(query: str, text: str) -> list[int] | None:
    """Positions of `query` as a subsequence of `text`, or None if it is not.

    Greedy from the left: the earliest possible position for each character
    leaves the most room for the rest, so it never misses a match that exists.
    """
    positions: list[int] = []
    start = 0
    for char in query:
        index = text.find(char, start)
        if index < 0:
            return None
        positions.append(index)
        start = index + 1
    return positions


def fuzzy_score(query: str, text: str) -> int:
    """Score how well `text` matches `query`; 0 means "no match".

    Matching is case-insensitive and by subsequence, so `ukr` finds
    `Ukrainian` and `haw` finds `Hawaiian`. The scoring keeps the rankings
    people expect from a picker:

    * a full prefix beats a word-start match, which beats a consecutive run
      in the middle, which beats scattered characters;
    * every gap costs points in proportion to its length;
    * shorter texts win over longer ones that needed the same characters.

    An empty query matches everything and returns 1, which keeps callers from
    special-casing it before sorting.
    """
    needle = query.strip().lower()
    if not needle:
        return 1
    haystack = text.lower()
    if not haystack:
        return 0

    positions = _match_positions(needle, haystack)
    if positions is None:
        return 0

    score = 10 * len(needle)

    if haystack.startswith(needle):
        score += 100
        if haystack == needle:
            score += 40

    word_starts = {0}
    for index, char in enumerate(haystack):
        if char in _WORD_BREAKS:
            word_starts.add(index + 1)
    at_word_start = sum(1 for position in positions if position in word_starts)
    score += 12 * at_word_start
    if at_word_start == len(positions):
        # Every character starts a word: an initialism like "Haitian Creole"
        # for "hc", which should not lose to a longer scattered match.
        score += 25

    runs = sum(1 for a, b in zip(positions, positions[1:], strict=False) if b == a + 1)
    score += 8 * runs

    gaps = sum(b - a - 1 for a, b in zip(positions, positions[1:], strict=False))
    score -= 2 * gaps
    score -= positions[0]
    score -= max(0, len(haystack) - len(needle)) // 2

    # Never let a real match score 0, which means "not a match" to callers.
    return max(score, 1)


def _dedupe(codes: Sequence[str]) -> list[Language]:
    """Known languages for `codes`, in order, without duplicates."""
    seen: set[str] = set()
    languages: list[Language] = []
    for code in codes:
        language = language_by_code(code)
        if language is None or language.code in seen:
            continue
        seen.add(language.code)
        languages.append(language)
    return languages


def search(
    query: str,
    favourites: Sequence[str] = (),
    preferred: str | None = None,
) -> list[Language]:
    """Languages matching `query`, best first.

    With no query the list is the whole table in the order a picker wants it:
    `Auto-detect`, then the user's favourites in the order they starred them,
    then the language the locale implies, then everything else alphabetically.
    With a query the ranking is by `fuzzy_score` over both the name and the
    code, and favourites plus the locale language get a small bonus so a tie
    stays where the user expects it.
    """
    favourite_languages = _dedupe(favourites)
    favourite_codes = {language.code for language in favourite_languages}

    if not query.strip():
        ordered: list[Language] = [AUTO]
        seen = {AUTO.code}
        for language in favourite_languages:
            if language.code not in seen:
                seen.add(language.code)
                ordered.append(language)
        if preferred is not None:
            language = language_by_code(preferred)
            if language is not None and language.code not in seen:
                seen.add(language.code)
                ordered.append(language)
        ordered.extend(
            sorted(
                (lang for lang in LANGUAGES if lang.code not in seen),
                key=lambda lang: lang.name.lower(),
            )
        )
        return ordered

    scored: list[tuple[int, Language]] = []
    for language in (AUTO, *LANGUAGES):
        score = max(
            fuzzy_score(query, language.name),
            fuzzy_score(query, language.code),
        )
        if score <= 0:
            continue
        if language.code in favourite_codes:
            score += FAVOURITE_BONUS
        if preferred is not None and language.code == preferred:
            score += PREFERRED_BONUS
        scored.append((score, language))

    # Sorted by score, then alphabetically so equally good matches keep a
    # stable, predictable order instead of the table's.
    scored.sort(key=lambda item: (-item[0], item[1].name.lower()))
    return [language for _score, language in scored]
