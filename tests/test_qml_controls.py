"""Tests for Main.qml's controls, loaded as the application loads it.

These exist because a whole class of defect is invisible to the Python tests:
a control can report a state its backend never had. Clicking a CheckBox flips
its own ``checked`` property, so ``checked: masker.frame_marked`` alone would
show a tick before -- or instead of -- anything reaching the disk.

Interaction is simulated by invoking the control's own ``toggle`` and
``clicked``, which is what a mouse release does internally, rather than by
synthesising input events: the point is what the handler leaves behind, not
Qt's event delivery.
"""

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_QUICK_BACKEND", "software")

import pytest
from PySide6.QtCore import QMetaObject
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtQuickControls2 import QQuickStyle
from rat_tracer.bad_frames import STORAGE_ENV_VAR
from rat_tracer.translations import resolve_translations
from rat_tracer.ui import VideoMasker

_QML = Path(__file__).resolve().parent.parent / "rat_tracer" / "Main.qml"


@pytest.fixture(scope="session")
def qapp():
    QQuickStyle.setStyle("Material")
    return QGuiApplication.instance() or QGuiApplication([])


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    """Load Main.qml exactly as ``ui.main`` does."""
    monkeypatch.setenv(STORAGE_ENV_VAR, str(tmp_path / "bad_frames"))
    engine = QQmlApplicationEngine()
    engine.rootContext().setContextProperty("tr", resolve_translations("en"))
    engine.load(_QML)
    roots = engine.rootObjects()
    assert roots, "Main.qml failed to load"
    root = roots[0]
    yield root
    del engine


def _find(window, name):
    control = window.findChild(object, name)
    assert control is not None, f"no control named {name!r} in Main.qml"
    return control


def _click(control):
    """What a mouse release does: toggle the control, then emit clicked."""
    QMetaObject.invokeMethod(control, "toggle")
    QMetaObject.invokeMethod(control, "clicked")


def test_the_mark_control_reports_storage_not_the_click(qapp, window):
    """Regression: the tick must never claim a frame is stored because it was
    clicked. Nothing is stored here -- there is no video open -- so the control
    must be back to unticked the moment the handler has run."""
    masker = window.findChild(VideoMasker)
    control = _find(window, "markBadFrameCheckBox")
    assert not control.property("checked")

    _click(control)
    qapp.processEvents()

    assert not masker.frame_marked, "nothing was stored"
    assert not control.property("checked"), (
        "the control is claiming the frame is stored while nothing is on disk"
    )


def test_the_mark_control_still_follows_the_backend_after_a_click(qapp, window):
    """The binding must survive the click, not merely be right once."""
    masker = window.findChild(VideoMasker)
    control = _find(window, "markBadFrameCheckBox")

    _click(control)
    qapp.processEvents()
    # Any later change of backend state must still reach the control.
    masker.mark_state_changed.emit()
    qapp.processEvents()

    assert control.property("checked") == masker.frame_marked


def test_controls_are_disabled_until_a_video_is_open(qapp, window):
    for name in (
        "markBadFrameCheckBox",
        "problemModeSwitch",
        "previousFrameButton",
        "nextFrameButton",
    ):
        assert not _find(window, name).property("enabled"), f"{name} should start disabled"


def test_the_toast_names_the_saved_frame_and_offers_undo(qapp, window):
    masker = window.findChild(VideoMasker)
    toast = _find(window, "markToast")
    label = _find(window, "markToastLabel")
    assert not toast.property("visible")

    masker.mark_saved.emit(4821)
    qapp.processEvents()

    assert toast.property("visible")
    assert "4821" in label.property("text")
    assert toast.property("undoable"), "a mark must be reversible for five seconds"


def test_a_failed_save_says_so_and_offers_no_undo(qapp, window):
    masker = window.findChild(VideoMasker)
    toast = _find(window, "markToast")
    label = _find(window, "markToastLabel")

    masker.mark_failed.emit(99)
    qapp.processEvents()

    text = label.property("text")
    assert "99" in text
    assert "NOT" in text, f"the researcher must be told it was not saved: {text!r}"
    assert not toast.property("undoable"), "there is nothing to undo"


def test_the_frame_index_is_shown(qapp, window):
    """A researcher and a technician need to name the same frame."""
    assert "0" in _find(window, "frameIndexLabel").property("text")


def test_the_slider_does_not_take_keyboard_focus(qapp, window):
    """Otherwise Left/Right would move the position by a slider step at the
    same time as the frame-step shortcut moved it by one frame."""
    slider = _find(window, "slider_here")

    assert not slider.property("activeFocusOnTab")
    assert not slider.property("activeFocus")
