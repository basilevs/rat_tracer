"""UI string translations keyed by language code.

The active language is derived from the OS locale at startup (see
``rat_tracer.ui.main``). English acts as the source language and the fallback
for any locale without a dedicated translation.
"""

DEFAULT_LANGUAGE = "en"

TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "open_video_title": "Open video",
        "video_files_filter": "Video files (*.mp4 *.mov *.avi *.mkv)",
        "all_files_filter": "All files (*)",
        "open_button": "Open…",
        "pause_button": "Pause",
        "play_button": "Play",
        "click_to_copy": "Click to copy",
        "cli_description": "Rat Tracer UI",
        "cli_video_help": "Video file to open on startup",
        "cli_video_not_found": "video file not found: {path}",
    },
    "ru": {
        "open_video_title": "Открыть видео",
        "video_files_filter": "Видеофайлы (*.mp4 *.mov *.avi *.mkv)",
        "all_files_filter": "Все файлы (*)",
        "open_button": "Открыть…",
        "pause_button": "Пауза",
        "play_button": "Воспроизвести",
        "click_to_copy": "Нажмите, чтобы скопировать",
        "cli_description": "Интерфейс Rat Tracer",
        "cli_video_help": "Видеофайл, открываемый при запуске",
        "cli_video_not_found": "видеофайл не найден: {path}",
    },
}


def resolve_translations(locale_name: str) -> dict[str, str]:
    """Return the translation dict for an OS locale name.

    ``locale_name`` is a value such as ``"ru_RU"`` or ``"en_US"`` as produced
    by ``QLocale.system().name()``. Only the language subtag is used, and
    unknown languages fall back to :data:`DEFAULT_LANGUAGE`.
    """
    language = locale_name.split("_", 1)[0].lower()
    return TRANSLATIONS.get(language, TRANSLATIONS[DEFAULT_LANGUAGE])
