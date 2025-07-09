"""Enhanced log manager component for modern GUI."""

import collections
from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import QTextEdit
from PySide6.QtGui import QTextCursor


class ModernLogManager:
    """Enhanced log manager with buffering and modern styling."""
    
    def __init__(self, text_edit: QTextEdit, max_lines: int = 1000):
        """
        Initialize the log manager.
        
        Args:
            text_edit: The QTextEdit widget to display logs
            max_lines: Maximum number of lines to keep
        """
        self.text_edit = text_edit
        self.max_lines = max_lines
        self.log_buffer = collections.deque(maxlen=max_lines)
        
        # Modern color scheme for logs
        self.colors = {
            "ERROR": "#ff6b6b",    # Red
            "WARNING": "#ffd93d",  # Yellow  
            "INFO": "#51cf66",     # Green
            "DEBUG": "#74c0fc",    # Blue
            "SUCCESS": "#69db7c",  # Bright green
        }
    
    def append(self, level: str, message: str, timestamp: Optional[datetime] = None):
        """
        Append a log message with appropriate formatting.
        
        Args:
            level: Log level ('INFO', 'WARNING', 'ERROR', etc.)
            message: The log message to append
            timestamp: Optional timestamp, defaults to now
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        time_str = timestamp.strftime("%H:%M:%S")
        color = self.colors.get(level.upper(), "#ffffff")
        
        formatted_msg = f'<span style="color:#888">[{time_str}]</span> <span style="color:{color};font-weight:bold">{level}:</span> <span style="color:#ffffff">{message}</span>'
        
        # Add to circular buffer
        self.log_buffer.append(formatted_msg)
        
        # Append to text edit
        self.text_edit.append(formatted_msg)
        self._trim_excess_lines()
        
        # Scroll to bottom
        self._scroll_to_bottom()
    
    def _trim_excess_lines(self):
        """Remove oldest lines if the count exceeds the maximum."""
        doc = self.text_edit.document()
        while doc.blockCount() > self.max_lines:
            cursor = QTextCursor(doc)
            cursor.select(QTextCursor.SelectionType.BlockUnderCursor)
            cursor.movePosition(QTextCursor.MoveOperation.NextCharacter, QTextCursor.MoveMode.KeepAnchor)
            cursor.removeSelectedText()
    
    def _scroll_to_bottom(self):
        """Scroll the text edit to show the latest logs."""
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.text_edit.setTextCursor(cursor)
    
    def clear(self):
        """Clear all logs."""
        self.log_buffer.clear()
        self.text_edit.clear()
    
    def get_logs(self) -> list:
        """Get all logs as a list."""
        return list(self.log_buffer) 