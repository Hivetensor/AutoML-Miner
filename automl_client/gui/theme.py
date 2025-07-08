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
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 15px;
            font-weight: bold;
        }}
        
        /* Navigation Sidebar */
        QWidget#sidebar {{
            background-color: {SolarTheme.OBSIDIAN_800};
            border-right: 2px solid {SolarTheme.OBSIDIAN_600};
            min-width: 280px;
            max-width: 280px;
        }}
        
        /* Logo Area */
        QWidget#logo_area {{
            background-color: {SolarTheme.OBSIDIAN_800};
            padding: 20px;
            border-bottom: 1px solid {SolarTheme.OBSIDIAN_600};
        }}
        
        QLabel#logo_text {{
            color: {SolarTheme.SOLAR_GOLD};
            font-family: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 20px;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        
        QLabel#logo_text:hover {{
            color: {SolarTheme.SOLAR_AMBER};
        }}
        
        /* Main Content Area */
        QWidget#content_area {{
            background-color: {SolarTheme.ABYSS_BLACK};
            padding: 30px;
        }}
        
        /* Navigation Buttons */
        QPushButton#nav_button {{
            background-color: transparent;
            color: {SolarTheme.GREY_400};
            border: none;
            padding: 18px 24px;
            text-align: left;
            font-size: 16px;
            font-weight: bold;
            border-radius: 0px;
        }}
        
        QPushButton#nav_button:hover {{
            background-color: {SolarTheme.OBSIDIAN_600};
            color: {SolarTheme.SOLAR_GOLD};
        }}
        
        QPushButton#nav_button:checked {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
            font-weight: 900;
        }}
        
        /* Primary Buttons */
        QPushButton#primary_button {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
            border: 2px solid {SolarTheme.SOLAR_GOLD};
            padding: 14px 28px;
            font-weight: 600;
            font-size: 15px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-radius: 6px;
            font-family: "JetBrains Mono", "Monaco", "Menlo", "Consolas", monospace;
        }}
        
        QPushButton#primary_button:hover {{
            background-color: {SolarTheme.SOLAR_AMBER};
            border-color: {SolarTheme.SOLAR_AMBER};
        }}
        
        QPushButton#primary_button:pressed {{
            background-color: {SolarTheme.SOLAR_AMBER};
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
            padding: 14px 28px;
            font-weight: 500;
            font-size: 15px;
            border-radius: 6px;
            font-family: "JetBrains Mono", "Monaco", "Menlo", "Consolas", monospace;
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
            padding: 14px 18px;
            font-size: 15px;
            border-radius: 6px;
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
            padding: 14px;
            font-size: 14px;
            font-family: "JetBrains Mono", "Monaco", "Menlo", "Consolas", monospace;
            border-radius: 6px;
        }}
        
        /* Progress Bars */
        QProgressBar {{
            background-color: {SolarTheme.OBSIDIAN_700};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            border-radius: 6px;
            text-align: center;
            color: {SolarTheme.WHITE};
            font-weight: 600;
            font-size: 14px;
        }}
        
        QProgressBar::chunk {{
            background-color: {SolarTheme.SOLAR_GOLD};
            border-radius: 4px;
        }}
        
        /* Group Boxes */
        QGroupBox {{
            font-size: 18px;
            font-weight: 600;
            color: {SolarTheme.WHITE};
            margin-top: 15px;
            padding-top: 25px;
            border: 2px solid {SolarTheme.OBSIDIAN_600};
            border-radius: 10px;
        }}
        
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 20px;
            padding: 0 10px 0 10px;
            color: {SolarTheme.SOLAR_GOLD};
        }}
        
        /* Labels */
        QLabel {{
            color: {SolarTheme.WHITE};
            font-size: 15px;
            font-weight: bold;
        }}
        
        QLabel#title {{
            font-size: 32px;
            font-weight: 700;
            color: {SolarTheme.SOLAR_GOLD};
            margin-bottom: 10px;
            letter-spacing: -0.025em;
        }}
        
        QLabel#subtitle {{
            font-size: 18px;
            color: {SolarTheme.GREY_300};
            margin-bottom: 20px;
            line-height: 1.4;
        }}
        
        QLabel#info {{
            color: {SolarTheme.GREY_400};
            font-size: 14px;
        }}
        
        /* Status Labels */
        QLabel#status_success {{
            color: #10b981;
            font-weight: 600;
            font-size: 15px;
        }}
        
        QLabel#status_error {{
            color: #ef4444;
            font-weight: 600;
            font-size: 15px;
        }}
        
        QLabel#status_warning {{
            color: {SolarTheme.SOLAR_AMBER};
            font-weight: 600;
            font-size: 15px;
        }}
        
        /* Cards/Panels */
        QWidget#card {{
            background-color: {SolarTheme.OBSIDIAN_800};
            border: 1px solid {SolarTheme.OBSIDIAN_600};
            border-radius: 10px;
            padding: 25px;
        }}
        
        /* Status Bar */
        QStatusBar {{
            background-color: {SolarTheme.OBSIDIAN_800};
            color: {SolarTheme.GREY_400};
            border-top: 1px solid {SolarTheme.OBSIDIAN_600};
            font-size: 13px;
        }}
        
        /* Combo Boxes */
        QComboBox {{
            background-color: {SolarTheme.OBSIDIAN_700};
            color: {SolarTheme.WHITE};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            padding: 12px 16px;
            border-radius: 6px;
            font-size: 15px;
        }}
        
        QComboBox:focus {{
            border-color: {SolarTheme.SOLAR_GOLD};
        }}
        
        QComboBox::drop-down {{
            border: none;
            width: 25px;
        }}
        
        QComboBox::down-arrow {{
            image: none;
            border-left: 6px solid transparent;
            border-right: 6px solid transparent;
            border-top: 6px solid {SolarTheme.GREY_400};
            margin-right: 8px;
        }}
        
        /* Spin Boxes */
        QSpinBox, QDoubleSpinBox {{
            background-color: {SolarTheme.OBSIDIAN_700};
            color: {SolarTheme.WHITE};
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            padding: 12px 16px;
            font-size: 15px;
            border-radius: 6px;
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {SolarTheme.SOLAR_GOLD};
        }}
        
        /* Check Boxes */
        QCheckBox {{
            color: {SolarTheme.WHITE};
            font-size: 15px;
            spacing: 8px;
        }}
        
        QCheckBox::indicator {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
            border: 2px solid {SolarTheme.OBSIDIAN_500};
            background-color: {SolarTheme.OBSIDIAN_700};
        }}
        
        QCheckBox::indicator:checked {{
            background-color: {SolarTheme.SOLAR_GOLD};
            border-color: {SolarTheme.SOLAR_GOLD};
        }}
        
        /* Tab Widget */
        QTabWidget::pane {{
            border: 2px solid {SolarTheme.OBSIDIAN_600};
            border-radius: 8px;
            background-color: {SolarTheme.OBSIDIAN_800};
        }}
        
        QTabBar::tab {{
            background-color: {SolarTheme.OBSIDIAN_700};
            color: {SolarTheme.GREY_400};
            padding: 12px 20px;
            margin-right: 2px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            font-size: 15px;
            font-weight: 500;
        }}
        
        QTabBar::tab:selected {{
            background-color: {SolarTheme.SOLAR_GOLD};
            color: {SolarTheme.ABYSS_BLACK};
            font-weight: 600;
        }}
        
        QTabBar::tab:hover:!selected {{
            background-color: {SolarTheme.OBSIDIAN_600};
            color: {SolarTheme.SOLAR_GOLD};
        }}
        
        /* Scrollbars */
        QScrollBar:vertical {{
            background-color: {SolarTheme.OBSIDIAN_700};
            width: 14px;
            margin: 0px;
        }}
        
        QScrollBar::handle:vertical {{
            background-color: {SolarTheme.OBSIDIAN_500};
            border-radius: 7px;
            min-height: 25px;
        }}
        
        QScrollBar::handle:vertical:hover {{
            background-color: {SolarTheme.SOLAR_GOLD};
        }}
        """
    
    @staticmethod
    def get_font_system():
        """Get the font system for the application."""
        fonts = {}
        
        # Primary font (Inter - matching website) - BOLD by default
        primary_font = QFont()
        primary_font.setFamilies([
            "Inter", "-apple-system", "BlinkMacSystemFont", 
            "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"
        ])
        primary_font.setPointSize(15)
        primary_font.setWeight(QFont.Weight.Bold)  # Make all text bold
        fonts['primary'] = primary_font
        
        # Monospace font for code/logs (JetBrains Mono - matching website) - BOLD
        mono_font = QFont()
        mono_font.setFamilies([
            "JetBrains Mono", "Monaco", "Menlo", "Consolas", 
            "Source Code Pro", "monospace"
        ])
        mono_font.setPointSize(14)
        mono_font.setWeight(QFont.Weight.Bold)  # Make monospace bold too
        fonts['mono'] = mono_font
        
        # Title font (Inter Bold - matching website) - EXTRA BOLD
        title_font = QFont()
        title_font.setFamilies([
            "Inter", "-apple-system", "BlinkMacSystemFont", 
            "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"
        ])
        title_font.setWeight(QFont.Weight.ExtraBold)  # Extra bold for titles
        title_font.setPointSize(22)
        fonts['title'] = title_font
        
        return fonts 