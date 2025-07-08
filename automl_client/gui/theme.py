"""Solar Theme System for AutoML Miner GUI."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor

class SolarTheme:
    """Solar-themed design system with black background and golden accents."""
    
    # Color Palette - matching the CSS
    SOLAR_GOLD = "#ffbe0b"
    SOLAR_AMBER = "#ff9b05" 
    ABYSS_BLACK = "#000000"
    OBSIDIAN_800 = "#0d0d0d"
    OBSIDIAN_700 = "#161616"
    OBSIDIAN_600 = "#1f1f1f"
    OBSIDIAN_500 = "#2a2a2a"
    WHITE = "#ffffff"
    GREY_400 = "#9ca3af"
    GREY_300 = "#d1d5db"
    
    @staticmethod
    def get_main_stylesheet() -> str:
        """Get the main application stylesheet."""
        return f"""
        /* Main Application */
        QMainWindow {{
            background-color: {SolarTheme.ABYSS_BLACK};
            color: {SolarTheme.WHITE};
            font-family: 'Inter', 'Helvetica Neue', 'Arial', sans-serif;
        }}
        
        /* Navigation Sidebar */
        QWidget#sidebar {{
            background-color: {SolarTheme.OBSIDIAN_800};
            border-right: 2px solid {SolarTheme.OBSIDIAN_600};
            min-width: 250px;
            max-width: 250px;
        }}
        
        /* Main Content Area */
        QWidget#content_area {{
            background-color: {SolarTheme.ABYSS_BLACK};
            padding: 20px;
        }}
        
        /* Navigation Buttons */
        QPushButton#nav_button {{
            background-color: transparent;
            color: {SolarTheme.GREY_400};
            border: none;
            padding: 15px 20px;
            text-align: left;
            font-size: 14px;
            font-weight: 500;
            border-radius: 0px;
        }}
        
        QPushButton#nav_button:hover {{
            background-color: {SolarTheme.OBSIDIAN_600};
            color: {SolarTheme.SOLAR_GOLD};
        }}
        
        QPushButton#nav_button:checked {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
            font-weight: 600;
        }}
        
        /* Primary Buttons */
        QPushButton#primary_button {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
            border: 2px solid {SolarTheme.SOLAR_GOLD};
            padding: 12px 24px;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-radius: 4px;
        }}
        
        QPushButton#primary_button:hover {{
            background-color: {SolarTheme.SOLAR_AMBER};
            border-color: {SolarTheme.SOLAR_AMBER};
        }}
        
        QPushButton#primary_button:pressed {{
            background-color: {SolarTheme.SOLAR_AMBER};
            transform: translateY(1px);
        }}
        
        QPushButton#primary_button:disabled {{
            background-color: {SolarTheme.OBSIDIAN_500};
            color: {SolarTheme.GREY_400};
            border-color: {SolarTheme.OBSIDIAN_500};
        }}
        
        /* Secondary Buttons */
        QPushButton#secondary_button {{
            background-color: transparent;
            color: {SolarTheme.SOLAR_GOLD};
            border: 2px solid {SolarTheme.SOLAR_GOLD};
            padding: 12px 24px;
            font-weight: 500;
            font-size: 13px;
            border-radius: 4px;
        }}
        
        QPushButton#secondary_button:hover {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
        }}
        
        /* Input Fields */
        QLineEdit {{
            background-color: {SolarTheme.OBSIDIAN_700};
            color: {SolarTheme.WHITE};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            padding: 12px 16px;
            font-size: 14px;
            border-radius: 4px;
        }}
        
        QLineEdit:focus {{
            border-color: {SolarTheme.SOLAR_GOLD};
            background-color: {SolarTheme.OBSIDIAN_600};
        }}
        
        /* Text Areas */
        QTextEdit {{
            background-color: {SolarTheme.OBSIDIAN_800};
            color: {SolarTheme.WHITE};
            border: 2px solid {SolarTheme.OBSIDIAN_600};
            padding: 12px;
            font-size: 13px;
            font-family: 'JetBrains Mono', 'Menlo', 'Consolas', monospace;
            border-radius: 4px;
        }}
        
        /* Progress Bars */
        QProgressBar {{
            background-color: {SolarTheme.OBSIDIAN_700};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            border-radius: 4px;
            text-align: center;
            color: {SolarTheme.WHITE};
            font-weight: 600;
        }}
        
        QProgressBar::chunk {{
            background-color: {SolarTheme.SOLAR_GOLD};
            border-radius: 2px;
        }}
        
        /* Group Boxes */
        QGroupBox {{
            font-size: 16px;
            font-weight: 600;
            color: {SolarTheme.WHITE};
            margin-top: 12px;
            padding-top: 20px;
            border: 2px solid {SolarTheme.OBSIDIAN_600};
            border-radius: 8px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 16px;
            padding: 0 8px 0 8px;
            color: {SolarTheme.SOLAR_GOLD};
        }}
        
        /* Labels */
        QLabel {{
            color: {SolarTheme.WHITE};
            font-size: 14px;
        }}
        
        QLabel#title {{
            font-size: 24px;
            font-weight: 700;
            color: {SolarTheme.SOLAR_GOLD};
            margin-bottom: 8px;
        }}
        
        QLabel#subtitle {{
            font-size: 16px;
            color: {SolarTheme.GREY_300};
            margin-bottom: 16px;
        }}
        
        QLabel#info {{
            color: {SolarTheme.GREY_400};
            font-size: 13px;
        }}
        
        /* Status Labels */
        QLabel#status_success {{
            color: #10b981;
            font-weight: 600;
        }}
        
        QLabel#status_error {{
            color: #ef4444;
            font-weight: 600;
        }}
        
        QLabel#status_warning {{
            color: {SolarTheme.SOLAR_AMBER};
            font-weight: 600;
        }}
        
        /* Cards/Panels */
        QWidget#card {{
            background-color: {SolarTheme.OBSIDIAN_800};
            border: 1px solid {SolarTheme.OBSIDIAN_600};
            border-radius: 8px;
            padding: 20px;
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {SolarTheme.OBSIDIAN_800};
            color: {SolarTheme.GREY_400};
            border-top: 1px solid {SolarTheme.OBSIDIAN_600};
        }}
        
        /* Combo Boxes */
        QComboBox {{
            background-color: {SolarTheme.OBSIDIAN_700};
            color: {SolarTheme.WHITE};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            padding: 8px 12px;
            border-radius: 4px;
        }}
        
        QComboBox:focus {{
            border-color: {SolarTheme.SOLAR_GOLD};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 20px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid {SolarTheme.GREY_400};
            margin-right: 5px;
        }}
        
        /* Scrollbars */
        QScrollBar:vertical {{
            background-color: {SolarTheme.OBSIDIAN_700};
            width: 12px;
            margin: 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {SolarTheme.OBSIDIAN_500};
            border-radius: 6px;
            min-height: 20px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {SolarTheme.SOLAR_GOLD};
        }}
        """
    
    @staticmethod
    def get_font_system():
        """Get the font system for the application."""
        fonts = {}
        
        # Primary font (Inter/Helvetica)
        primary_font = QFont()
        primary_font.setFamilies(["Inter", "Helvetica Neue", "Arial", "sans-serif"])
        fonts['primary'] = primary_font
        
        # Monospace font for code/logs
        mono_font = QFont()
        mono_font.setFamilies(["JetBrains Mono", "Menlo", "Consolas", "monospace"])
        fonts['mono'] = mono_font
        
        # Title font
        title_font = QFont()
        title_font.setFamilies(["Inter", "Helvetica Neue", "Arial", "sans-serif"])
        title_font.setWeight(QFont.Weight.Bold)
        title_font.setPointSize(18)
        fonts['title'] = title_font
        
        return fonts 