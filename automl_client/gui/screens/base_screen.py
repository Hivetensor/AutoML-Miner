"""Base screen class for AutoML Miner GUI screens."""

from typing import TYPE_CHECKING
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Qt

if TYPE_CHECKING:
    from ..main_window import ModernMiningWindow

class BaseScreen(QWidget):
    """Base class for all application screens."""
    
    def __init__(self, main_window: 'ModernMiningWindow', title: str = ""):
        super().__init__()
        self.main_window = main_window
        self.title = title
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the basic UI structure. Override in subclasses."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(40, 40, 40, 40)
        self.layout.setSpacing(20)
        
        if self.title:
            title_label = QLabel(self.title)
            title_label.setObjectName("title")
            self.layout.addWidget(title_label)
    
    def on_screen_activated(self):
        """Called when this screen becomes active. Override in subclasses."""
        pass
    
    def create_card_widget(self) -> QWidget:
        """Create a card-style widget."""
        card = QWidget()
        card.setObjectName("card")
        return card
    
    def create_scrollable_content(self, content_widget: QWidget) -> QScrollArea:
        """Create a scrollable area for content."""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        return scroll 