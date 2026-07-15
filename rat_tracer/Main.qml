import QtQuick 2.0
import QtQuick.Layouts 1.11
import QtQuick.Controls 2.1
import QtQuick.Window 2.1
import QtQuick.Controls.Material 2.1
import QtMultimedia

ApplicationWindow {
    id: page
    width: 800
    height: 400
    visible: true
    Material.theme: Material.Dark
    Material.accent: Material.Red
    ColumnLayout {
        spacing: 2
        anchors.fill: parent
        Layout.columnSpan: 1
        Layout.preferredWidth: 400
        Layout.fillWidth: true
        Layout.fillHeight: true
        MediaPlayer {
            id: player
            source: "file:///Users/vasiligulevich/git/rat_tracer/input/2025-10-10.mp4"
            videoOutput: videoOutput
        }
        VideoOutput {
            id: videoOutput
            Layout.fillWidth: true
            Layout.fillHeight: true
            height: 200
            width: 200
            fillMode: VideoOutput.PreserveAspectFit
        }

    }

}
