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
        "no_file_open": "No file open",
        "cli_description": "Rat Tracer UI",
        "cli_video_help": "Video file to open on startup",
        "cli_video_not_found": "video file not found: {path}",
        "problem_mode_button": "Check detection",
        "problem_mode_tooltip": "Hide the visited-area track and show what the detector found on this frame alone",
        "mark_bad_frame": "Mark bad frame",
        "mark_bad_frame_tooltip": "Store this frame for retraining (F2)",
        "frame_already_marked_tooltip": "This frame is already stored",
        "previous_frame": "◀",
        "next_frame": "▶",
        "previous_frame_tooltip": "Previous frame (Left arrow)",
        "next_frame_tooltip": "Next frame (Right arrow)",
        "frame_label": "Frame {index}",
        "mark_saved_toast": "Frame {index} saved for retraining",
        "mark_failed_toast": "Frame {index} was NOT saved. Check free disk space.",
        "undo_button": "Undo",
        "collect_description": "Package every marked frame into one archive file",
        "collect_nothing_to_archive": "No marked frames found in {path}",
        "collect_done": "Archive written to: {path}",
    },
    "ru": {
        "open_video_title": "Открыть видео",
        "video_files_filter": "Видеофайлы (*.mp4 *.mov *.avi *.mkv)",
        "all_files_filter": "Все файлы (*)",
        "open_button": "Открыть…",
        "pause_button": "Пауза",
        "play_button": "Воспроизвести",
        "click_to_copy": "Нажмите, чтобы скопировать",
        "no_file_open": "Файл не открыт",
        "cli_description": "Интерфейс Rat Tracer",
        "cli_video_help": "Видеофайл, открываемый при запуске",
        "cli_video_not_found": "видеофайл не найден: {path}",
        "problem_mode_button": "Проверка распознавания",
        "problem_mode_tooltip": "Скрыть след посещённых мест и показать только то, что распознано на этом кадре",
        "mark_bad_frame": "Отметить кадр",
        "mark_bad_frame_tooltip": "Сохранить кадр для переобучения (F2)",
        "frame_already_marked_tooltip": "Этот кадр уже сохранён",
        "previous_frame": "◀",
        "next_frame": "▶",
        "previous_frame_tooltip": "Предыдущий кадр (стрелка влево)",
        "next_frame_tooltip": "Следующий кадр (стрелка вправо)",
        "frame_label": "Кадр {index}",
        "mark_saved_toast": "Кадр {index} сохранён для переобучения",
        "mark_failed_toast": "Кадр {index} НЕ сохранён. Проверьте свободное место на диске.",
        "undo_button": "Отменить",
        "collect_description": "Собрать все отмеченные кадры в один архив",
        "collect_nothing_to_archive": "Отмеченные кадры не найдены в {path}",
        "collect_done": "Архив записан: {path}",
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
