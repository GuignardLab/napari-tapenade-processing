from qtpy.QtCore import Qt
from qtpy.QtWidgets import (QHBoxLayout, QPushButton, QScrollArea, QSizePolicy,
                            QStackedWidget, QVBoxLayout, QWidget, QLabel)


class MultiLineTabWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.tabBarLayout = QVBoxLayout()
        
        self.scrollArea = QScrollArea()
        self.scrollArea.setWidgetResizable(True)
        self.tabContainer = QWidget()
        self.tabContainer.setLayout(self.tabBarLayout)
        self.scrollArea.setWidget(self.tabContainer)
        
        self.layout.addWidget(self.scrollArea)

        self.stackedWidget = QStackedWidget()
        self.layout.addWidget(self.stackedWidget)

        self.setLayout(self.layout)

        self.tabs = []

    def addTab(self, widget, title):
        tabButton = QPushButton(title)
        tabButton.setCheckable(True)
        tabButton.clicked.connect(lambda: self.onTabClicked(tabButton))

        tabRowLayout = None
        if len(self.tabs) % 3 == 0:
            tabRowLayout = QHBoxLayout()
            self.tabBarLayout.addLayout(tabRowLayout)
        else:
            tabRowLayout = self.tabBarLayout.itemAt(self.tabBarLayout.count() - 1).layout()

        tabRowLayout.addWidget(tabButton)
        self.tabs.append(tabButton)
        
        self.stackedWidget.addWidget(widget)

    def onTabClicked(self, tabButton):
        for i, btn in enumerate(self.tabs):
            if btn == tabButton:
                self.stackedWidget.setCurrentIndex(i)
                btn.setChecked(True)
            else:
                btn.setChecked(False)


class RichTextPushButton(QPushButton):
    def __init__(self, parent=None, text=None):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__()
        self.__lbl = QLabel(self)
        if text is not None:
            self.__lbl.setText(text)
        self.__lyt = QHBoxLayout()
        self.__lyt.setContentsMargins(0, 0, 0, 0)
        self.__lyt.setSpacing(0)
        self.setLayout(self.__lyt)
        self.__lbl.setAttribute(Qt.WA_TranslucentBackground)
        self.__lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.__lbl.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Expanding,
        )
        self.__lbl.setTextFormat(Qt.RichText)
        self.__lyt.addWidget(self.__lbl)
        return

    def setText(self, text):
        self.__lbl.setText(text)
        self.updateGeometry()
        return

    def sizeHint(self):
        s = QPushButton.sizeHint(self)
        w = self.__lbl.sizeHint()
        s.setWidth(w.width())
        s.setHeight(w.height())
        return s
