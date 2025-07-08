"""Settings screen for AutoML Miner."""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QLineEdit, QSpinBox, QCheckBox, QComboBox,
    QTextEdit, QMessageBox, QGridLayout, QDoubleSpinBox
)
from PySide6.QtCore import Qt
from .base_screen import BaseScreen

if TYPE_CHECKING:
    from ..main_window import ModernMiningWindow

logger = logging.getLogger(__name__)

class SettingsScreen(BaseScreen):
    """Application settings screen."""
    
    def __init__(self, main_window: 'ModernMiningWindow'):
        super().__init__(main_window, "Settings")
        self.settings_file = Path.home() / ".automl_pool" / "settings.json"
        self.settings = self._load_settings()
        self.setup_settings_ui()
    
    def setup_settings_ui(self):
        """Setup the settings interface."""
        # Mining settings
        mining_widget = self._create_mining_settings()
        self.layout.addWidget(mining_widget)
        
        # Network settings
        network_widget = self._create_network_settings()
        self.layout.addWidget(network_widget)
        
        # Advanced settings
        advanced_widget = self._create_advanced_settings()
        self.layout.addWidget(advanced_widget)
        
        # Actions
        actions_widget = self._create_actions_section()
        self.layout.addWidget(actions_widget)
        
        self.layout.addStretch()
    
    def _create_mining_settings(self) -> QWidget:
        """Create mining configuration settings."""
        group = QGroupBox("Mining Configuration")
        layout = QGridLayout(group)
        
        # Default pool URL
        layout.addWidget(QLabel("Default Pool URL:"), 0, 0)
        self.pool_url_edit = QLineEdit(self.settings.get("pool_url", "http://pool.hivetensor.com:1337"))
        layout.addWidget(self.pool_url_edit, 0, 1)
        
        # Default max cycles
        layout.addWidget(QLabel("Default Max Cycles:"), 1, 0)
        self.max_cycles_spin = QSpinBox()
        self.max_cycles_spin.setMinimum(0)
        self.max_cycles_spin.setMaximum(100000)
        self.max_cycles_spin.setValue(self.settings.get("max_cycles", 0))
        self.max_cycles_spin.setSpecialValueText("Unlimited")
        layout.addWidget(self.max_cycles_spin, 1, 1)
        
        # Cycle timeout
        layout.addWidget(QLabel("Cycle Timeout (seconds):"), 2, 0)
        self.cycle_timeout_spin = QSpinBox()
        self.cycle_timeout_spin.setMinimum(30)
        self.cycle_timeout_spin.setMaximum(3600)
        self.cycle_timeout_spin.setValue(self.settings.get("cycle_timeout", 120))
        layout.addWidget(self.cycle_timeout_spin, 2, 1)
        
        # Auto-restart on error
        layout.addWidget(QLabel("Auto-restart on Error:"), 3, 0)
        self.auto_restart_check = QCheckBox()
        self.auto_restart_check.setChecked(self.settings.get("auto_restart", False))
        layout.addWidget(self.auto_restart_check, 3, 1)
        
        # Preferred task type
        layout.addWidget(QLabel("Preferred Task Type:"), 4, 0)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["alternate", "evolve", "evaluate"])
        current_task = self.settings.get("task_type", "alternate")
        index = self.task_type_combo.findText(current_task)
        if index >= 0:
            self.task_type_combo.setCurrentIndex(index)
        layout.addWidget(self.task_type_combo, 4, 1)
        
        return group
    
    def _create_network_settings(self) -> QWidget:
        """Create network configuration settings."""
        group = QGroupBox("Network Configuration")
        layout = QGridLayout(group)
        
        # Connection timeout
        layout.addWidget(QLabel("Connection Timeout (seconds):"), 0, 0)
        self.connection_timeout_spin = QSpinBox()
        self.connection_timeout_spin.setMinimum(5)
        self.connection_timeout_spin.setMaximum(300)
        self.connection_timeout_spin.setValue(self.settings.get("connection_timeout", 30))
        layout.addWidget(self.connection_timeout_spin, 0, 1)
        
        # Max retries
        layout.addWidget(QLabel("Max Retries:"), 1, 0)
        self.max_retries_spin = QSpinBox()
        self.max_retries_spin.setMinimum(1)
        self.max_retries_spin.setMaximum(10)
        self.max_retries_spin.setValue(self.settings.get("max_retries", 3))
        layout.addWidget(self.max_retries_spin, 1, 1)
        
        # Delay between cycles
        layout.addWidget(QLabel("Delay Between Cycles (seconds):"), 2, 0)
        self.cycle_delay_spin = QDoubleSpinBox()
        self.cycle_delay_spin.setMinimum(0.1)
        self.cycle_delay_spin.setMaximum(60.0)
        self.cycle_delay_spin.setValue(self.settings.get("cycle_delay", 5.0))
        layout.addWidget(self.cycle_delay_spin, 2, 1)
        
        return group
    
    def _create_advanced_settings(self) -> QWidget:
        """Create advanced configuration settings."""
        group = QGroupBox("Advanced Configuration")
        layout = QGridLayout(group)
        
        # Debug logging
        layout.addWidget(QLabel("Debug Logging:"), 0, 0)
        self.debug_logging_check = QCheckBox()
        self.debug_logging_check.setChecked(self.settings.get("debug_logging", False))
        layout.addWidget(self.debug_logging_check, 0, 1)
        
        # Log level
        layout.addWidget(QLabel("Log Level:"), 1, 0)
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        current_level = self.settings.get("log_level", "INFO")
        index = self.log_level_combo.findText(current_level)
        if index >= 0:
            self.log_level_combo.setCurrentIndex(index)
        layout.addWidget(self.log_level_combo, 1, 1)
        
        # Auto-update check
        layout.addWidget(QLabel("Check for Updates:"), 2, 0)
        self.auto_update_check = QCheckBox()
        self.auto_update_check.setChecked(self.settings.get("auto_update", True))
        layout.addWidget(self.auto_update_check, 2, 1)
        
        # Data directory
        layout.addWidget(QLabel("Data Directory:"), 3, 0)
        data_dir_layout = QHBoxLayout()
        self.data_dir_edit = QLineEdit(self.settings.get("data_dir", str(Path.home() / ".automl_pool")))
        data_dir_layout.addWidget(self.data_dir_edit)
        
        browse_data_btn = QPushButton("Browse")
        browse_data_btn.setObjectName("secondary_button")
        browse_data_btn.clicked.connect(self._browse_data_directory)
        data_dir_layout.addWidget(browse_data_btn)
        layout.addLayout(data_dir_layout, 3, 1)
        
        return group
    
    def _create_actions_section(self) -> QWidget:
        """Create settings action buttons."""
        group = QGroupBox("Actions")
        layout = QHBoxLayout(group)
        
        # Save settings
        save_btn = QPushButton("Save Settings")
        save_btn.setObjectName("primary_button")
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)
        
        # Reset to defaults
        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setObjectName("secondary_button")
        reset_btn.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_btn)
        
        # Export settings
        export_btn = QPushButton("Export Settings")
        export_btn.setObjectName("secondary_button")
        export_btn.clicked.connect(self._export_settings)
        layout.addWidget(export_btn)
        
        # Import settings
        import_btn = QPushButton("Import Settings")
        import_btn.setObjectName("secondary_button")
        import_btn.clicked.connect(self._import_settings)
        layout.addWidget(import_btn)
        
        layout.addStretch()
        
        return group
    
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file."""
        default_settings = {
            "pool_url": "http://pool.hivetensor.com:1337",
            "max_cycles": 0,
            "cycle_timeout": 120,
            "auto_restart": False,
            "task_type": "alternate",
            "connection_timeout": 30,
            "max_retries": 3,
            "cycle_delay": 5.0,
            "debug_logging": False,
            "log_level": "INFO",
            "auto_update": True,
            "data_dir": str(Path.home() / ".automl_pool")
        }
        
        try:
            if self.settings_file.exists():
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    default_settings.update(loaded_settings)
                    return default_settings
        except Exception as e:
            logger.warning(f"Failed to load settings: {e}")
        
        return default_settings
    
    def _save_settings(self):
        """Save current settings to file."""
        try:
            # Gather current settings from UI
            current_settings = {
                "pool_url": self.pool_url_edit.text().strip(),
                "max_cycles": self.max_cycles_spin.value(),
                "cycle_timeout": self.cycle_timeout_spin.value(),
                "auto_restart": self.auto_restart_check.isChecked(),
                "task_type": self.task_type_combo.currentText(),
                "connection_timeout": self.connection_timeout_spin.value(),
                "max_retries": self.max_retries_spin.value(),
                "cycle_delay": self.cycle_delay_spin.value(),
                "debug_logging": self.debug_logging_check.isChecked(),
                "log_level": self.log_level_combo.currentText(),
                "auto_update": self.auto_update_check.isChecked(),
                "data_dir": self.data_dir_edit.text().strip()
            }
            
            # Ensure settings directory exists
            self.settings_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            with open(self.settings_file, 'w') as f:
                json.dump(current_settings, f, indent=2)
            
            self.settings = current_settings
            
            QMessageBox.information(
                self,
                "Settings Saved",
                "Settings have been saved successfully."
            )
            
            logger.info("Settings saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to save settings:\n{str(e)}"
            )
    
    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to their default values?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # Reset UI to defaults
            self.pool_url_edit.setText("http://pool.hivetensor.com:1337")
            self.max_cycles_spin.setValue(0)
            self.cycle_timeout_spin.setValue(120)
            self.auto_restart_check.setChecked(False)
            self.task_type_combo.setCurrentText("alternate")
            self.connection_timeout_spin.setValue(30)
            self.max_retries_spin.setValue(3)
            self.cycle_delay_spin.setValue(5.0)
            self.debug_logging_check.setChecked(False)
            self.log_level_combo.setCurrentText("INFO")
            self.auto_update_check.setChecked(True)
            self.data_dir_edit.setText(str(Path.home() / ".automl_pool"))
    
    def _export_settings(self):
        """Export settings to a file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Settings",
            str(Path.home() / "automl_settings.json"),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                # Get current settings
                current_settings = {
                    "pool_url": self.pool_url_edit.text().strip(),
                    "max_cycles": self.max_cycles_spin.value(),
                    "cycle_timeout": self.cycle_timeout_spin.value(),
                    "auto_restart": self.auto_restart_check.isChecked(),
                    "task_type": self.task_type_combo.currentText(),
                    "connection_timeout": self.connection_timeout_spin.value(),
                    "max_retries": self.max_retries_spin.value(),
                    "cycle_delay": self.cycle_delay_spin.value(),
                    "debug_logging": self.debug_logging_check.isChecked(),
                    "log_level": self.log_level_combo.currentText(),
                    "auto_update": self.auto_update_check.isChecked(),
                    "data_dir": self.data_dir_edit.text().strip()
                }
                
                with open(file_path, 'w') as f:
                    json.dump(current_settings, f, indent=2)
                
                QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Settings exported to:\n{file_path}"
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Export Failed",
                    f"Failed to export settings:\n{str(e)}"
                )
    
    def _import_settings(self):
        """Import settings from a file."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Settings",
            str(Path.home()),
            "JSON Files (*.json)"
        )
        
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    imported_settings = json.load(f)
                
                # Update UI with imported settings
                self.pool_url_edit.setText(imported_settings.get("pool_url", ""))
                self.max_cycles_spin.setValue(imported_settings.get("max_cycles", 0))
                self.cycle_timeout_spin.setValue(imported_settings.get("cycle_timeout", 120))
                self.auto_restart_check.setChecked(imported_settings.get("auto_restart", False))
                
                task_type = imported_settings.get("task_type", "alternate")
                index = self.task_type_combo.findText(task_type)
                if index >= 0:
                    self.task_type_combo.setCurrentIndex(index)
                
                self.connection_timeout_spin.setValue(imported_settings.get("connection_timeout", 30))
                self.max_retries_spin.setValue(imported_settings.get("max_retries", 3))
                self.cycle_delay_spin.setValue(imported_settings.get("cycle_delay", 5.0))
                self.debug_logging_check.setChecked(imported_settings.get("debug_logging", False))
                
                log_level = imported_settings.get("log_level", "INFO")
                index = self.log_level_combo.findText(log_level)
                if index >= 0:
                    self.log_level_combo.setCurrentIndex(index)
                
                self.auto_update_check.setChecked(imported_settings.get("auto_update", True))
                self.data_dir_edit.setText(imported_settings.get("data_dir", ""))
                
                QMessageBox.information(
                    self,
                    "Import Successful",
                    "Settings imported successfully. Don't forget to save them."
                )
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Failed to import settings:\n{str(e)}"
                )
    
    def _browse_data_directory(self):
        """Browse for data directory."""
        from PySide6.QtWidgets import QFileDialog
        
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            self.data_dir_edit.text() or str(Path.home())
        )
        
        if directory:
            self.data_dir_edit.setText(directory)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        return self.settings
    
    def on_screen_activated(self):
        """Called when settings screen becomes active."""
        pass 