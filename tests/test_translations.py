from rat_tracer.translations import (
    DEFAULT_LANGUAGE,
    TRANSLATIONS,
    resolve_translations,
)

# ── resolve_translations ──────────────────────────────────────────────────────


def test_resolve_russian():
    assert resolve_translations("ru_RU") == TRANSLATIONS["ru"]


def test_resolve_english():
    assert resolve_translations("en_US") == TRANSLATIONS["en"]


def test_resolve_unknown_falls_back_to_default():
    assert resolve_translations("de_DE") == TRANSLATIONS[DEFAULT_LANGUAGE]


def test_resolve_language_only_name():
    assert resolve_translations("ru") == TRANSLATIONS["ru"]


def test_resolve_is_case_insensitive():
    assert resolve_translations("RU_ru") == TRANSLATIONS["ru"]


# ── key parity ────────────────────────────────────────────────────────────────


def test_all_locales_share_the_same_keys():
    expected = set(TRANSLATIONS[DEFAULT_LANGUAGE])
    for language, strings in TRANSLATIONS.items():
        assert set(strings) == expected, f"{language} has mismatched keys"
