"""Modern main window with sidebar navigation for AutoML Miner."""

import sys
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QStackedWidget, QFrame, QApplication, QStatusBar
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QIcon

from .theme import SolarTheme
from .screens.wallet_screen import WalletScreen
from .screens.mining_screen import MiningScreen  
from .screens.settings_screen import SettingsScreen
from .screens.dashboard_screen import DashboardScreen

# Local imports for backend functionality
from ..stop_flag import StopFlag
from ..wallet import Wallet
from ..client import BittensorPoolClient

logger = logging.getLogger(__name__)

class ModernMiningWindow(QMainWindow):
    """Modern main window with sidebar navigation."""
    
    def __init__(self):
        super().__init__()
        
        # Backend state
        self.wallet: Optional[Wallet] = None
        self.client: Optional[BittensorPoolClient] = None
        self.stop_flag = StopFlag()
        
        # UI state
        self.current_screen = "dashboard"
        self.screens: Dict[str, QWidget] = {}
        self.nav_buttons: Dict[str, QPushButton] = {}
        
        self._setup_window()
        self._create_ui()
        self._setup_navigation()
        self._apply_theme()
        
        # Status updates
        self._setup_status_timer()
        
        logger.info("Modern Mining Window initialized")
    
    def _setup_window(self):
        """Configure main window properties."""
        self.setWindowTitle("AutoML Miner")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Center window on screen
        screen = QApplication.primaryScreen().geometry()
        window_geometry = self.frameGeometry()
        center_point = screen.center()
        window_geometry.moveCenter(center_point)
        self.move(window_geometry.topLeft())
    
    def _create_ui(self):
        """Create the main UI layout."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create sidebar
        self.sidebar = self._create_sidebar()
        main_layout.addWidget(self.sidebar)
        
        # Create content area
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("content_area")
        main_layout.addWidget(self.content_stack)
        
        # Create screens
        self._create_screens()
        
        # Create status bar
        self._create_status_bar()
    
    def _create_sidebar(self) -> QWidget:
        """Create the navigation sidebar."""
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Logo/Header area
        header_widget = self._create_header()
        layout.addWidget(header_widget)
        
        # Navigation buttons
        nav_data = [
            ("dashboard", "Dashboard", "📊"),
            ("wallet", "Wallet Setup", "💳"),
            ("mining", "Mining", "⛏️"),
            ("settings", "Settings", "⚙️"),
        ]
        
        for screen_id, title, icon in nav_data:
            btn = QPushButton(f"{icon}  {title}")
            btn.setObjectName("nav_button")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, s=screen_id: self._switch_screen(s))
            
            self.nav_buttons[screen_id] = btn
            layout.addWidget(btn)
        
        # Set dashboard as default
        self.nav_buttons["dashboard"].setChecked(True)
        
        # Spacer to push everything to top
        layout.addStretch()
        
        # Version info at bottom
        version_label = QLabel("v1.0.0")
        version_label.setObjectName("info")
        version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        version_label.setContentsMargins(20, 10, 20, 20)
        layout.addWidget(version_label)
        
        return sidebar
    
    def _create_header(self) -> QWidget:
        """Create the header area with logo and title."""
        header = QWidget()
        layout = QVBoxLayout(header)
        layout.setContentsMargins(20, 30, 20, 30)
        
        # App title
        title = QLabel("AutoML Miner")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Subtitle
        subtitle = QLabel("Neural Component Pool")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        
        # Separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"background-color: {SolarTheme.OBSIDIAN_600}; margin: 20px 0;")
        layout.addWidget(separator)
        
        return header
    
    def _create_screens(self):
        """Create and add all screen widgets."""
        # Dashboard screen
        dashboard_screen = DashboardScreen(self)
        self.screens["dashboard"] = dashboard_screen
        self.content_stack.addWidget(dashboard_screen)
        
        # Wallet screen  
        wallet_screen = WalletScreen(self)
        self.screens["wallet"] = wallet_screen
        self.content_stack.addWidget(wallet_screen)
        
        # Mining screen
        mining_screen = MiningScreen(self)
        self.screens["mining"] = mining_screen
        self.content_stack.addWidget(mining_screen)
        
        # Settings screen
        settings_screen = SettingsScreen(self)
        self.screens["settings"] = settings_screen
        self.content_stack.addWidget(settings_screen)
    
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _setup_navigation(self):
        """Setup navigation behavior."""
        # Connect screen-specific actions
        if "wallet" in self.screens:
            wallet_screen = self.screens["wallet"]
            if hasattr(wallet_screen, 'wallet_loaded'):
                wallet_screen.wallet_loaded.connect(self._on_wallet_loaded)
    
    def _apply_theme(self):
        """Apply the solar theme to the window."""
        self.setStyleSheet(SolarTheme.get_main_stylesheet())
        
        # Apply fonts
        fonts = SolarTheme.get_font_system()
        self.setFont(fonts['primary'])
    
    def _setup_status_timer(self):
        """Setup timer for status updates."""
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(5000)  # Update every 5 seconds
    
    def _switch_screen(self, screen_id: str):
        """Switch to a different screen."""
        if screen_id not in self.screens:
            logger.warning(f"Unknown screen: {screen_id}")
            return
        
        # Update navigation buttons
        for btn_id, btn in self.nav_buttons.items():
            btn.setChecked(btn_id == screen_id)
        
        # Switch stack widget
        screen_widget = self.screens[screen_id]
        self.content_stack.setCurrentWidget(screen_widget)
        self.current_screen = screen_id
        
        # Notify screen of activation
        if hasattr(screen_widget, 'on_screen_activated'):
            screen_widget.on_screen_activated()
        
        logger.debug(f"Switched to screen: {screen_id}")
    
    def _update_status(self):
        """Update status bar with current information."""
        status_parts = []
        
        # Wallet status
        if self.wallet:
            status_parts.append(f"Wallet: {self.wallet.name}")
        else:
            status_parts.append("No wallet loaded")
        
        # Client status
        if self.client:
            status_parts.append("Connected")
        else:
            status_parts.append("Disconnected")
        
        self.status_bar.showMessage(" | ".join(status_parts))
    
    def _on_wallet_loaded(self, wallet: Wallet):
        """Handle wallet being loaded."""
        self.wallet = wallet
        logger.info(f"Wallet loaded: {wallet.name}")
        
        # Create client with the wallet
        try:
            # TODO: Get API URL from settings
            api_url = "http://pool.hivetensor.com:1337"
            self.client = BittensorPoolClient(wallet=wallet, base_url=api_url)
            logger.info("Client created successfully")
        except Exception as e:
            logger.error(f"Failed to create client: {e}")
            self.client = None
        
        # Update status
        self._update_status()
        
        # Switch to mining screen if wallet is loaded
        self._switch_screen("mining")
    
    def get_wallet(self) -> Optional[Wallet]:
        """Get current wallet."""
        return self.wallet
    
    def get_client(self) -> Optional[BittensorPoolClient]:
        """Get current client."""
        return self.client
    
    def get_stop_flag(self) -> StopFlag:
        """Get the stop flag."""
        return self.stop_flag
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop any running operations
        self.stop_flag.stop()
        
        # Close client if exists
        if self.client:
            try:
                if hasattr(self.client, '__exit__'):
                    self.client.__exit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing client: {e}")
        
        logger.info("Application closing")
        event.accept() 