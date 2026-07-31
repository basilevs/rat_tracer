// Versionless: Shortcut needs QtQuick 2.5 or later, and Qt 6 resolves
// versionless imports to the newest available module.
import QtQuick
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1
import QtQuick.Dialogs
import QtMultimedia
import MyBackend

ApplicationWindow {
    id: page
    width: 800
    height: 400
    visible: true
    Material.theme: Material.Dark
    Material.accent: Material.Red
    property bool playing: true
    VideoMasker {
        id: "masker"
        playing: page.playing
        video_output: videoOutput
        onMark_saved: (frameIndex) => toast.show(tr.mark_saved_toast.replace("{index}", frameIndex), true)
        onMark_failed: (frameIndex) => toast.show(tr.mark_failed_toast.replace("{index}", frameIndex), false)
        onProblem_mode_changed: (active) => { if (active) page.playing = false }
    }

    // Marking never navigates and never blocks: the researcher can seek or
    // step to the next frame of interest while the write is still in flight.
    function markBadFrame() {
        masker.markBadFrame()
    }

    Shortcut {
        // Layout-independent: Qt reports letter keys per the active keyboard
        // layout, so a letter shortcut would move under a Russian layout.
        sequence: "F2"
        enabled: masker.can_mark && !masker.frame_marked
        onActivated: markBadFrame()
    }
    Shortcut {
        sequence: StandardKey.MoveToPreviousChar
        onActivated: masker.stepFrame(-1)
    }
    Shortcut {
        sequence: StandardKey.MoveToNextChar
        onActivated: masker.stepFrame(1)
    }
    FileDialog {
        id: fileDialog
        title: tr.open_video_title
        nameFilters: [tr.video_files_filter, tr.all_files_filter]
        onAccepted: masker.openVideo(selectedFile)
    }
    ColumnLayout {
        spacing: 2
        anchors.fill: parent
        Layout.columnSpan: 1
        Layout.preferredWidth: 400
        Layout.fillWidth: true
        Layout.fillHeight: true
        VideoOutput {
            id: videoOutput
            objectName: "videoOutput"
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 200
            width: 200
            fillMode: VideoOutput.PreserveAspectFit
            MouseArea {
                anchors.fill: parent
                onClicked: {
                    page.playing = !page.playing
                }
            }
            DropArea {
                anchors.fill: parent
                onDropped: (drop) => {
                    if (drop.hasUrls) {
                        masker.openVideo(drop.urls[0])
                    }
                }
            }
        }
        Slider {
            id: "slider"
            Layout.fillWidth: true
            objectName: "slider_here"
            // A focused Slider consumes Left/Right itself, which would move
            // the position by a slider step at the same time as the
            // frame-step shortcut moves it by one frame. Dragging still works.
            focusPolicy: Qt.NoFocus
            onMoved: {
                masker.position = slider.value;
                page.playing = false;
            }
            value: masker.position
        }
        RowLayout {
            Button {
                text: tr.open_button
                Layout.alignment: Qt.AlignHCenter
                onClicked: fileDialog.open()
            }
            Button {
                text: page.playing ? tr.pause_button : tr.play_button
                Layout.alignment: Qt.AlignHCenter
                onClicked: {
                    page.playing = !page.playing
                }
            }
            Button {
                objectName: "previousFrameButton"
                text: tr.previous_frame
                enabled: masker.video != ""
                ToolTip.visible: hovered
                ToolTip.text: tr.previous_frame_tooltip
                onClicked: masker.stepFrame(-1)
            }
            Button {
                objectName: "nextFrameButton"
                text: tr.next_frame
                enabled: masker.video != ""
                ToolTip.visible: hovered
                ToolTip.text: tr.next_frame_tooltip
                onClicked: masker.stepFrame(1)
            }
            TextEdit {
                id: clipboardHelper
                visible: false
            }
            Button {
                text: masker.time_text
                ToolTip.visible: hovered
                ToolTip.text: tr.click_to_copy
                onClicked: {
                    clipboardHelper.text = masker.time_text
                    clipboardHelper.selectAll()
                    clipboardHelper.copy()
                }
            }
            Label {
                objectName: "frameIndexLabel"
                text: tr.frame_label.replace("{index}", masker.frame_index)
            }
            Item { Layout.fillWidth: true }
            Switch {
                objectName: "problemModeSwitch"
                text: tr.problem_mode_button
                enabled: masker.video != ""
                checked: masker.problem_mode
                ToolTip.visible: hovered
                ToolTip.text: tr.problem_mode_tooltip
                // Driven from the backend rather than from the checked state:
                // resuming playback leaves the mode, and the switch has to
                // follow that instead of fighting it.
                onToggled: masker.problem_mode = checked
            }
            CheckBox {
                objectName: "markBadFrameCheckBox"
                text: tr.mark_bad_frame
                // Stateful, not fire-and-forget: it shows whether the frame on
                // screen is already stored, so the researcher sees the answer
                // before acting rather than discovering it by pressing.
                checked: masker.frame_marked
                enabled: masker.can_mark && !masker.frame_marked
                ToolTip.visible: hovered
                ToolTip.text: masker.frame_marked ? tr.frame_already_marked_tooltip
                                                  : tr.mark_bad_frame_tooltip
                onClicked: markBadFrame()
            }
        }
        Button {
            text: masker.video || tr.no_file_open
            Layout.fillWidth: true
            ToolTip.visible: hovered
            ToolTip.text: tr.click_to_copy
            onClicked: {
                clipboardHelper.text = masker.video
                clipboardHelper.selectAll()
                clipboardHelper.copy()
            }
        }
    }

    // Floats over the video rather than sitting in the layout, so it never
    // blocks seeking or frame stepping while it is visible.
    Rectangle {
        id: toast
        objectName: "markToast"
        property bool undoable: false
        visible: false
        radius: 4
        color: undoable ? Material.color(Material.Grey, Material.Shade800)
                        : Material.color(Material.Red, Material.Shade800)
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.bottom: parent.bottom
        anchors.bottomMargin: 80
        implicitWidth: toastRow.implicitWidth + 24
        implicitHeight: toastRow.implicitHeight + 16

        function show(message, undoable) {
            toastLabel.text = message
            toast.undoable = undoable
            toast.visible = true
            dismissTimer.restart()
        }

        RowLayout {
            id: toastRow
            anchors.centerIn: parent
            spacing: 12
            Label {
                id: toastLabel
                objectName: "markToastLabel"
            }
            Button {
                objectName: "undoMarkButton"
                text: tr.undo_button
                visible: toast.undoable
                flat: true
                onClicked: {
                    masker.undoLastMark()
                    toast.visible = false
                    dismissTimer.stop()
                }
            }
        }

        Timer {
            id: dismissTimer
            // Five seconds is the whole correction window: a wrong mark not
            // undone here travels to the technician, since there is no
            // marked-frame navigation to find it again.
            interval: 5000
            onTriggered: toast.visible = false
        }
    }
}
