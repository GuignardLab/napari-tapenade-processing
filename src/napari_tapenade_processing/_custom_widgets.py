from qtpy.QtCore import QPoint, Qt, QRect
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
)
from qtpy import QtGui


class HoverTooltipButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(20, 20)  # Small square button
        self.setText("?")  # Add interrogation mark inside the button
        self.setCheckable(True)  # Toggle button to show/hide the tooltip

        # Create the custom tooltip as a top-level widget (not a child of the button)
        self.text_box = QLabel()
        self.text_box.setText(text)
        self.text_box.setStyleSheet(
            "background-color: yellow; border: 1px solid black; padding: 5px;"
        )
        self.text_box.setWindowFlags(Qt.ToolTip)  # Make it look like a tooltip
        self.text_box.hide()

        # Enable mouse tracking to track mouse movement inside the button
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.isChecked():
                self.setChecked(False)  # Uncheck to hide the tooltip
                self.text_box.hide()
            else:
                self.setChecked(True)  # Check to show the tooltip
                # move the tooltip to the mouse position
                self.adjust_tooltip_position(event.globalPos())
                self.text_box.show()  # Immediately show tooltip on click
        super().mousePressEvent(event)

    def leaveEvent(self, event):
        # Hide the text box when mouse leaves the button
        self.text_box.hide()
        self.setChecked(False)  # Uncheck to hide the tooltip
        super().leaveEvent(event)

    def mouseMoveEvent(self, event):
        # Update tooltip position to follow the mouse using global coordinates
        if self.isChecked():
            # self.text_box.move(event.globalPos() + QPoint(10, 10))  # Offset for better visibility
            self.adjust_tooltip_position(event.globalPos())
        super().mouseMoveEvent(event)

    def adjust_tooltip_position(self, cursor_pos):
        screen = QApplication.screenAt(cursor_pos)

        if screen is not None:
            screen_rect = (
                screen.availableGeometry()
            )  # Get the geometry of the screen with the cursor
        else:
            screen_rect = (
                QApplication.desktop().availableGeometry()
            )  # Fallback to primary screen

        # Get the size of the tooltip
        tooltip_size = self.text_box.sizeHint()

        # Calculate the desired position of the tooltip
        new_x = cursor_pos.x() + 10  # Offset for better visibility
        new_y = cursor_pos.y() + 10

        # Adjust the position if the tooltip goes beyond the screen's right edge
        if new_x + tooltip_size.width() > screen_rect.right():
            new_x = (
                screen_rect.right() - tooltip_size.width() - 10
            )  # Shift left

        # Adjust the position if the tooltip goes beyond the screen's bottom edge
        if new_y + tooltip_size.height() > screen_rect.bottom():
            new_y = (
                screen_rect.bottom() - tooltip_size.height() - 10
            )  # Shift up

        # Adjust the position if the tooltip goes beyond the screen's left edge
        if new_x < screen_rect.left():
            new_x = screen_rect.left() + 10  # Shift right

        # Adjust the position if the tooltip goes beyond the screen's top edge
        if new_y < screen_rect.top():
            new_y = screen_rect.top() + 10  # Shift down

        # Move the tooltip to the new adjusted position
        self.text_box.move(QPoint(new_x, new_y))



class SwitchButton(QPushButton):
    def __init__(self, parent = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(True)
        self.setMinimumWidth(66)
        self.setMinimumHeight(22)

    def paintEvent(self, event):
        label = "ON" if self.isChecked() else "OFF"
        bg_color = Qt.blue if self.isChecked() else Qt.blue

        radius = 10
        width = 32
        center = self.rect().center()

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.translate(center)
        painter.setBrush(QtGui.QColor(0,0,0))

        pen = QtGui.QPen(Qt.black)
        pen.setWidth(2)
        painter.setPen(pen)

        painter.drawRoundedRect(QRect(-width, -radius, 2*width, 2*radius), radius, radius)
        painter.setBrush(QtGui.QBrush(bg_color))
        sw_rect = QRect(-radius, -radius, width + radius, 2*radius)
        if not self.isChecked():
            sw_rect.moveLeft(-width)
        painter.drawRoundedRect(sw_rect, radius, radius)
        painter.drawText(sw_rect, Qt.AlignCenter, label)