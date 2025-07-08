"""Dashboard screen for AutoML Miner."""

from typing import TYPE_CHECKING
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGridLayout, QGroupBox, QFrame
)
from PySide6.QtCore import Qt, QTimer
from .base_screen import BaseScreen

if TYPE_CHECKING:
    from ..main_window import ModernMiningWindow

class DashboardScreen(BaseScreen):
    """Dashboard screen showing system overview and status."""
    
    def __init__(self, main_window: 'ModernMiningWindow'):
        super().__init__(main_window, "Dashboard")
        self.setup_dashboard()
        
        # Timer for updating stats
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_stats)
        self.update_timer.start(10000)  # Update every 10 seconds
    
    def setup_dashboard(self):
        """Setup the dashboard layout."""
        # Welcome section
        welcome_widget = self._create_welcome_section()
        self.layout.addWidget(welcome_widget)
        
        # Stats grid
        stats_widget = self._create_stats_section()
        self.layout.addWidget(stats_widget)
        
        # Quick actions
        actions_widget = self._create_actions_section()
        self.layout.addWidget(actions_widget)
        
        # Status section
        status_widget = self._create_status_section()
        self.layout.addWidget(status_widget)
        
        self.layout.addStretch()
    
    def _create_welcome_section(self) -> QWidget:
        """Create welcome section."""
        card = self.create_card_widget()
        layout = QVBoxLayout(card)
        
        welcome_label = QLabel("Welcome to AutoML Miner")
        welcome_label.setObjectName("title")
        layout.addWidget(welcome_label)
        
        description = QLabel(
            "Participate in distributed evolutionary computing to evolve neural network components. "
            "Contribute your computing power to discover optimized loss functions and earn Alpha tokens."
        )
        description.setObjectName("subtitle")
        description.setWordWrap(True)
        layout.addWidget(description)
        
        return card
    
    def _create_stats_section(self) -> QWidget:
        """Create statistics section."""
        group = QGroupBox("System Statistics")
        layout = QGridLayout(group)
        
        # Wallet status
        layout.addWidget(QLabel("Wallet Status:"), 0, 0)
        self.wallet_status_label = QLabel("Not loaded")
        self.wallet_status_label.setObjectName("status_error")
        layout.addWidget(self.wallet_status_label, 0, 1)
        
        # Connection status
        layout.addWidget(QLabel("Pool Connection:"), 1, 0)
        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setObjectName("status_error")
        layout.addWidget(self.connection_status_label, 1, 1)
        
        # Mining status
        layout.addWidget(QLabel("Mining Status:"), 2, 0)
        self.mining_status_label = QLabel("Stopped")
        self.mining_status_label.setObjectName("status_warning")
        layout.addWidget(self.mining_status_label, 2, 1)
        
        # Balance (placeholder)
        layout.addWidget(QLabel("Alpha Balance:"), 3, 0)
        self.balance_label = QLabel("0.0 ALPHA")
        self.balance_label.setObjectName("info")
        layout.addWidget(self.balance_label, 3, 1)
        
        return group
    
    def _create_actions_section(self) -> QWidget:
        """Create quick actions section."""
        group = QGroupBox("Quick Actions")
        layout = QHBoxLayout(group)
        
        # Setup wallet button
        setup_wallet_btn = QPushButton("Setup Wallet")
        setup_wallet_btn.setObjectName("primary_button")
        setup_wallet_btn.clicked.connect(lambda: self.main_window._switch_screen("wallet"))
        layout.addWidget(setup_wallet_btn)
        
        # Start mining button (initially disabled)
        self.start_mining_btn = QPushButton("Start Mining")
        self.start_mining_btn.setObjectName("secondary_button")
        self.start_mining_btn.setEnabled(False)
        self.start_mining_btn.clicked.connect(lambda: self.main_window._switch_screen("mining"))
        layout.addWidget(self.start_mining_btn)
        
        # Settings button
        settings_btn = QPushButton("Settings")
        settings_btn.setObjectName("secondary_button")
        settings_btn.clicked.connect(lambda: self.main_window._switch_screen("settings"))
        layout.addWidget(settings_btn)
        
        layout.addStretch()
        
        return group
    
    def _create_status_section(self) -> QWidget:
        """Create system status section."""
        group = QGroupBox("System Information")
        layout = QVBoxLayout(group)
        
        # Version info
        version_info = QLabel("AutoML Miner v1.0.0")
        version_info.setObjectName("info")
        layout.addWidget(version_info)
        
        # System info
        import platform
        system_info = QLabel(f"Platform: {platform.system()} {platform.release()}")
        system_info.setObjectName("info")
        layout.addWidget(system_info)
        
        # Python version
        import sys
        python_info = QLabel(f"Python: {sys.version.split()[0]}")
        python_info.setObjectName("info")
        layout.addWidget(python_info)
        
        return group
    
    def update_stats(self):
        """Update dashboard statistics."""
        # Update wallet status
        wallet = self.main_window.get_wallet()
        if wallet:
            self.wallet_status_label.setText(f"Loaded: {wallet.name}")
            self.wallet_status_label.setObjectName("status_success")
            self.start_mining_btn.setEnabled(True)
        else:
            self.wallet_status_label.setText("Not loaded")
            self.wallet_status_label.setObjectName("status_error")
            self.start_mining_btn.setEnabled(False)
        
        # Update connection status
        client = self.main_window.get_client()
        if client:
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setObjectName("status_success")
        else:
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setObjectName("status_error")
        
        # Force style update
        self.wallet_status_label.style().unpolish(self.wallet_status_label)
        self.wallet_status_label.style().polish(self.wallet_status_label)
        self.connection_status_label.style().unpolish(self.connection_status_label)
        self.connection_status_label.style().polish(self.connection_status_label)
    
    def on_screen_activated(self):
        """Called when dashboard becomes active."""
        self.update_stats() 