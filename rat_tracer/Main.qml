import QtQuick 2.0
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

}
