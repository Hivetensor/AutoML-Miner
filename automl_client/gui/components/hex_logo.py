"""Hexagonal logo widget matching the website design."""

from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QBrush, QPolygonF, QColor
from ..theme import SolarTheme
import math

class HexWidget(QWidget):
    """Custom hexagonal shape widget matching website design."""
    
    def __init__(self, size=32, parent=None):
        super().__init__(parent)
        self.size = size
        self.setFixedSize(size, size)
        
    def paintEvent(self, event):
        """Paint the hexagonal shape using clip-path coordinates from website."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Website clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%)
        # Convert percentages to actual coordinates
        w, h = self.width(), self.height()
        
        # Create hexagon points matching website clip-path
        points = [
            QPointF(w * 0.5, h * 0.0),    # 50% 0% (top center)
            QPointF(w * 1.0, h * 0.25),   # 100% 25% (top right)
            QPointF(w * 1.0, h * 0.75),   # 100% 75% (bottom right)
            QPointF(w * 0.5, h * 1.0),    # 50% 100% (bottom center)
            QPointF(w * 0.0, h * 0.75),   # 0% 75% (bottom left)
            QPointF(w * 0.0, h * 0.25),   # 0% 25% (top left)
        ]
        
        # Create polygon and fill with solar gold
        hexagon = QPolygonF(points)
        
        # Parse the solar gold color
        color = QColor(SolarTheme.SOLAR_GOLD)
        brush = QBrush(color)
        
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPolygon(hexagon)

class HexLogo(QWidget):
    """Complete logo widget with hex shape and HIVETENSOR text."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the logo layout."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)
        
        # Real hexagonal shape
        self.hex_widget = HexWidget(32)
        
        # Logo text
        self.logo_text = QLabel("HIVETENSOR")
        self.logo_text.setObjectName("logo_text")
        
        # Add to layout
        layout.addWidget(self.hex_widget, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.logo_text, 0, Qt.AlignmentFlag.AlignVCenter)
        layout.addStretch()
        
        # Set cursor for hover effect
        self.setCursor(Qt.CursorShape.PointingHandCursor) 