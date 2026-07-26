import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1
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
        }
        Slider {
            id: "slider"
            Layout.fillWidth: true
            objectName: "slider_here"
            onMoved: () => masker.position = slider.value
            //value: masker.position
        }
        Button {
            text: page.playing ? "Pause" : "Play"
            Layout.alignment: Qt.AlignHCenter
            onClicked: {
                page.playing = !page.playing
            }
        }
    }

}
