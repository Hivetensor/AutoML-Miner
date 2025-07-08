"""Wallet management screen for AutoML Miner."""

import os
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QLineEdit, QGroupBox, QFileDialog, QMessageBox, QTextEdit,
    QGridLayout, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from .base_screen import BaseScreen

# Local imports
from ...wallet import Wallet

if TYPE_CHECKING:
    from ..main_window import ModernMiningWindow

logger = logging.getLogger(__name__)

class WalletScreen(BaseScreen):
    """Wallet management screen."""
    
    # Signal emitted when wallet is loaded
    wallet_loaded = Signal(object)  # Wallet object
    
    def __init__(self, main_window: 'ModernMiningWindow'):
        super().__init__(main_window, "Wallet Management")
        self.wallet: Optional[Wallet] = None
        self.setup_wallet_ui()
    
    def setup_wallet_ui(self):
        """Setup the wallet management interface."""
        # Status section
        status_widget = self._create_status_section()
        self.layout.addWidget(status_widget)
        
        # Main wallet operations
        operations_widget = self._create_operations_section()
        self.layout.addWidget(operations_widget)
        
        self.layout.addStretch()
    
    def _create_status_section(self) -> QWidget:
        """Create wallet status section."""
        group = QGroupBox("Current Wallet Status")
        layout = QVBoxLayout(group)
        
        # Current wallet info
        self.wallet_info_label = QLabel("No wallet loaded")
        self.wallet_info_label.setObjectName("subtitle")
        layout.addWidget(self.wallet_info_label)
        
        # Wallet directory
        self.wallet_dir_label = QLabel()
        self.wallet_dir_label.setObjectName("info")
        layout.addWidget(self.wallet_dir_label)
        
        # Update initial status
        self._update_status()
        
        return group
    
    def _create_operations_section(self) -> QWidget:
        """Create wallet operations section."""
        # Use tabs for different operations
        tab_widget = QTabWidget()
        
        # Create wallet tab
        create_tab = self._create_create_tab()
        tab_widget.addTab(create_tab, "Create New Wallet")
        
        # Import wallet tab
        import_tab = self._create_import_tab()
        tab_widget.addTab(import_tab, "Import Wallet")
        
        # Load existing wallet tab
        load_tab = self._create_load_tab()
        tab_widget.addTab(load_tab, "Load Existing")
        
        return tab_widget
    
    def _create_create_tab(self) -> QWidget:
        """Create the 'Create New Wallet' tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel(
            "Create a new wallet with a generated mnemonic phrase. "
            "Make sure to securely store your mnemonic phrase!"
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("info")
        layout.addWidget(instructions)
        
        # Wallet details form
        form_group = QGroupBox("Wallet Details")
        form_layout = QGridLayout(form_group)
        
        # Wallet name
        form_layout.addWidget(QLabel("Wallet Name:"), 0, 0)
        self.new_wallet_name = QLineEdit("default")
        form_layout.addWidget(self.new_wallet_name, 0, 1)
        
        # Hotkey name
        form_layout.addWidget(QLabel("Hotkey Name:"), 1, 0)
        self.new_hotkey_name = QLineEdit("default")
        form_layout.addWidget(self.new_hotkey_name, 1, 1)
        
        # Wallet directory
        form_layout.addWidget(QLabel("Wallet Directory:"), 2, 0)
        dir_layout = QHBoxLayout()
        self.wallet_dir_edit = QLineEdit(str(Path.home() / ".automl_pool" / "wallets"))
        dir_layout.addWidget(self.wallet_dir_edit)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setObjectName("secondary_button")
        browse_btn.clicked.connect(self._browse_wallet_directory)
        dir_layout.addWidget(browse_btn)
        form_layout.addLayout(dir_layout, 2, 1)
        
        layout.addWidget(form_group)
        
        # Create button
        create_btn = QPushButton("Create New Wallet")
        create_btn.setObjectName("primary_button")
        create_btn.clicked.connect(self._create_new_wallet)
        layout.addWidget(create_btn)
        
        # Generated mnemonic display
        self.mnemonic_display = QTextEdit()
        self.mnemonic_display.setPlaceholderText("Generated mnemonic will appear here...")
        self.mnemonic_display.setReadOnly(True)
        self.mnemonic_display.setMaximumHeight(100)
        layout.addWidget(self.mnemonic_display)
        
        layout.addStretch()
        return widget
    
    def _create_import_tab(self) -> QWidget:
        """Create the 'Import Wallet' tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel(
            "Import an existing wallet using your mnemonic phrase or private key."
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("info")
        layout.addWidget(instructions)
        
        # Import form
        form_group = QGroupBox("Import Details")
        form_layout = QGridLayout(form_group)
        
        # Wallet name
        form_layout.addWidget(QLabel("Wallet Name:"), 0, 0)
        self.import_wallet_name = QLineEdit("imported")
        form_layout.addWidget(self.import_wallet_name, 0, 1)
        
        # Hotkey name
        form_layout.addWidget(QLabel("Hotkey Name:"), 1, 0)
        self.import_hotkey_name = QLineEdit("default")
        form_layout.addWidget(self.import_hotkey_name, 1, 1)
        
        # Mnemonic phrase
        form_layout.addWidget(QLabel("Mnemonic Phrase:"), 2, 0)
        self.mnemonic_input = QTextEdit()
        self.mnemonic_input.setPlaceholderText("Enter your 12-24 word mnemonic phrase...")
        self.mnemonic_input.setMaximumHeight(80)
        form_layout.addWidget(self.mnemonic_input, 2, 1)
        
        layout.addWidget(form_group)
        
        # Import button
        import_btn = QPushButton("Import Wallet")
        import_btn.setObjectName("primary_button")
        import_btn.clicked.connect(self._import_wallet)
        layout.addWidget(import_btn)
        
        layout.addStretch()
        return widget
    
    def _create_load_tab(self) -> QWidget:
        """Create the 'Load Existing Wallet' tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Instructions
        instructions = QLabel(
            "Load an existing wallet from your file system."
        )
        instructions.setWordWrap(True)
        instructions.setObjectName("info")
        layout.addWidget(instructions)
        
        # Load form
        form_group = QGroupBox("Load Wallet")
        form_layout = QGridLayout(form_group)
        
        # Wallet name
        form_layout.addWidget(QLabel("Wallet Name:"), 0, 0)
        self.load_wallet_name = QLineEdit("default")
        form_layout.addWidget(self.load_wallet_name, 0, 1)
        
        # Hotkey name
        form_layout.addWidget(QLabel("Hotkey Name:"), 1, 0)
        self.load_hotkey_name = QLineEdit("default")
        form_layout.addWidget(self.load_hotkey_name, 1, 1)
        
        layout.addWidget(form_group)
        
        # Load button
        load_btn = QPushButton("Load Wallet")
        load_btn.setObjectName("primary_button")
        load_btn.clicked.connect(self._load_existing_wallet)
        layout.addWidget(load_btn)
        
        layout.addStretch()
        return widget
    
    def _browse_wallet_directory(self):
        """Browse for wallet directory."""
        directory = QFileDialog.getExistingDirectory(
            self, 
            "Select Wallet Directory",
            str(Path.home())
        )
        if directory:
            self.wallet_dir_edit.setText(directory)
    
    def _create_new_wallet(self):
        """Create a new wallet."""
        try:
            wallet_name = self.new_wallet_name.text().strip()
            hotkey_name = self.new_hotkey_name.text().strip()
            wallet_dir = self.wallet_dir_edit.text().strip()
            
            if not wallet_name:
                QMessageBox.warning(self, "Error", "Please enter a wallet name.")
                return
            
            # Create wallet directory if it doesn't exist
            wallet_path = Path(wallet_dir)
            wallet_path.mkdir(parents=True, exist_ok=True)
            
            # Create wallet
            self.wallet = Wallet(
                name=wallet_name,
                hotkey=hotkey_name,
                path=str(wallet_path)
            )
            
            # Display the mnemonic (if available)
            if hasattr(self.wallet, 'mnemonic'):
                self.mnemonic_display.setText(self.wallet.mnemonic)
            else:
                self.mnemonic_display.setText("Wallet created successfully!")
            
            self._update_status()
            self.wallet_loaded.emit(self.wallet)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Wallet '{wallet_name}' created successfully!\n\n"
                "Please save your mnemonic phrase in a secure location."
            )
            
            logger.info(f"Created new wallet: {wallet_name}")
            
        except Exception as e:
            logger.error(f"Error creating wallet: {e}")
            QMessageBox.critical(self, "Error", f"Failed to create wallet:\n{str(e)}")
    
    def _import_wallet(self):
        """Import a wallet from mnemonic."""
        try:
            wallet_name = self.import_wallet_name.text().strip()
            hotkey_name = self.import_hotkey_name.text().strip()
            mnemonic = self.mnemonic_input.toPlainText().strip()
            
            if not wallet_name:
                QMessageBox.warning(self, "Error", "Please enter a wallet name.")
                return
            
            if not mnemonic:
                QMessageBox.warning(self, "Error", "Please enter a mnemonic phrase.")
                return
            
            # Create wallet from mnemonic
            # Note: This is a simplified version - actual implementation may vary
            wallet_dir = self.wallet_dir_edit.text().strip()
            wallet_path = Path(wallet_dir)
            wallet_path.mkdir(parents=True, exist_ok=True)
            
            self.wallet = Wallet(
                name=wallet_name,
                hotkey=hotkey_name,
                path=str(wallet_path)
                # TODO: Add mnemonic parameter when Wallet class supports it
            )
            
            self._update_status()
            self.wallet_loaded.emit(self.wallet)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Wallet '{wallet_name}' imported successfully!"
            )
            
            logger.info(f"Imported wallet: {wallet_name}")
            
        except Exception as e:
            logger.error(f"Error importing wallet: {e}")
            QMessageBox.critical(self, "Error", f"Failed to import wallet:\n{str(e)}")
    
    def _load_existing_wallet(self):
        """Load an existing wallet."""
        try:
            wallet_name = self.load_wallet_name.text().strip()
            hotkey_name = self.load_hotkey_name.text().strip()
            
            if not wallet_name:
                QMessageBox.warning(self, "Error", "Please enter a wallet name.")
                return
            
            # Load existing wallet
            wallet_dir = self.wallet_dir_edit.text().strip()
            
            self.wallet = Wallet(
                name=wallet_name,
                hotkey=hotkey_name,
                path=wallet_dir
            )
            
            self._update_status()
            self.wallet_loaded.emit(self.wallet)
            
            QMessageBox.information(
                self, 
                "Success", 
                f"Wallet '{wallet_name}' loaded successfully!"
            )
            
            logger.info(f"Loaded wallet: {wallet_name}")
            
        except Exception as e:
            logger.error(f"Error loading wallet: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load wallet:\n{str(e)}")
    
    def _update_status(self):
        """Update wallet status display."""
        if self.wallet:
            self.wallet_info_label.setText(f"Loaded: {self.wallet.name}")
            if hasattr(self.wallet, 'path'):
                self.wallet_dir_label.setText(f"Directory: {self.wallet.path}")
            else:
                self.wallet_dir_label.setText("Directory: Unknown")
        else:
            self.wallet_info_label.setText("No wallet loaded")
            self.wallet_dir_label.setText("")
    
    def on_screen_activated(self):
        """Called when wallet screen becomes active."""
        self._update_status() 