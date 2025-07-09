"""Mining screen for AutoML Miner."""

import logging
import queue
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QGroupBox, QProgressBar, QTextEdit, QGridLayout, QFrame,
    QMessageBox, QLineEdit, QComboBox
)
from PySide6.QtCore import Qt, QTimer, QRunnable, QThreadPool, QObject, Signal, Slot
from .base_screen import BaseScreen
from ..components import ModernLogManager

if TYPE_CHECKING:
    from ..main_window import ModernMiningWindow

logger = logging.getLogger(__name__)

class MiningTask(QRunnable):
    """Background task for mining operations."""
    
    class Signals(QObject):
        progress = Signal(int, str)
        log = Signal(str, str)
        finished = Signal()
        error = Signal(str)
        stopping = Signal()
    
    def __init__(self, client, max_cycles: int, cycle_timeout: int, stop_flag):
        super().__init__()
        self.client = client
        self.max_cycles = max_cycles
        self.cycle_timeout = cycle_timeout
        self.stop_flag = stop_flag
        self.signals = self.Signals()
        self.setAutoDelete(True)
    
    def stop(self):
        """Stop the mining task."""
        self.stop_flag.stop()
        if hasattr(self.client, "stop_mining"):
            self.client.stop_mining()
        self.signals.stopping.emit()
    
    @Slot()
    def run(self):
        """Run the mining task."""
        try:
            self.signals.progress.emit(0, "Starting mining cycles...")
            self.signals.log.emit("INFO", "Mining started")
            
            if hasattr(self.client, 'run_continuous_mining'):
                self.client.run_continuous_mining(
                    cycles=self.max_cycles,
                    alternate=True,
                    delay=5.0,
                    max_retries=3,
                    stop_flag=self.stop_flag,
                )
            else:
                # Fallback simulation for testing
                import time
                for i in range(self.max_cycles if self.max_cycles > 0 else 100):
                    if self.stop_flag.is_stopped():
                        break
                    time.sleep(1)
                    progress = min(100, (i + 1) * 100 // (self.max_cycles if self.max_cycles > 0 else 100))
                    self.signals.progress.emit(progress, f"Mining cycle {i + 1}")
                    
                    if i % 10 == 0:
                        self.signals.log.emit("INFO", f"Completed {i + 1} mining cycles")
            
            if self.stop_flag.is_stopped():
                self.signals.progress.emit(100, "Stopped by user")
                self.signals.log.emit("INFO", "Mining stopped by user")
            else:
                self.signals.progress.emit(100, "Mining completed")
                self.signals.log.emit("INFO", "Mining completed successfully")
                
        except Exception as e:
            self.signals.error.emit(f"Mining error: {e}")
            self.signals.log.emit("ERROR", f"Mining failed: {e}")
        finally:
            self.signals.finished.emit()

class MiningScreen(BaseScreen):
    """Mining operations screen."""
    
    def __init__(self, main_window: 'ModernMiningWindow'):
        super().__init__(main_window, "Mining Operations")
        self.mining_task: Optional[MiningTask] = None
        self.is_mining = False
        self.setup_mining_ui()
        
        # Setup periodic updates
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_mining_status)
        self.update_timer.start(2000)  # Update every 2 seconds
    
    def setup_mining_ui(self):
        """Setup the mining interface."""
        # Mining controls section
        controls_widget = self._create_controls_section()
        self.layout.addWidget(controls_widget)
        
        # Status section
        status_widget = self._create_status_section()
        self.layout.addWidget(status_widget)
        
        # Progress section
        progress_widget = self._create_progress_section()
        self.layout.addWidget(progress_widget)
        
        # Logs section
        logs_widget = self._create_logs_section()
        self.layout.addWidget(logs_widget)
    
    def _create_controls_section(self) -> QWidget:
        """Create mining controls section."""
        group = QGroupBox("Mining Controls")
        layout = QVBoxLayout(group)
        
        # Settings row
        settings_layout = QGridLayout()
        
        # Pool URL
        settings_layout.addWidget(QLabel("Pool URL:"), 0, 0)
        self.pool_url_edit = QLineEdit("http://pool.hivetensor.com:1337")
        settings_layout.addWidget(self.pool_url_edit, 0, 1)
        
        # Task type
        settings_layout.addWidget(QLabel("Task Type:"), 1, 0)
        self.task_type_combo = QComboBox()
        self.task_type_combo.addItems(["Alternate", "Evolve Only", "Evaluate Only"])
        settings_layout.addWidget(self.task_type_combo, 1, 1)
        
        # Max cycles
        settings_layout.addWidget(QLabel("Max Cycles:"), 2, 0)
        self.max_cycles_edit = QLineEdit("0")
        self.max_cycles_edit.setPlaceholderText("0 = unlimited")
        settings_layout.addWidget(self.max_cycles_edit, 2, 1)
        
        layout.addLayout(settings_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Mining")
        self.start_button.setObjectName("primary_button")
        self.start_button.clicked.connect(self._start_mining)
        button_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Mining")
        self.stop_button.setObjectName("secondary_button")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_mining)
        button_layout.addWidget(self.stop_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        return group
    
    def _create_status_section(self) -> QWidget:
        """Create status monitoring section."""
        group = QGroupBox("Mining Status")
        layout = QGridLayout(group)
        
        # Wallet status
        layout.addWidget(QLabel("Wallet:"), 0, 0)
        self.wallet_status_label = QLabel("Not loaded")
        self.wallet_status_label.setObjectName("status_error")
        layout.addWidget(self.wallet_status_label, 0, 1)
        
        # Connection status
        layout.addWidget(QLabel("Connection:"), 1, 0)
        self.connection_status_label = QLabel("Disconnected")
        self.connection_status_label.setObjectName("status_error")
        layout.addWidget(self.connection_status_label, 1, 1)
        
        # Mining status
        layout.addWidget(QLabel("Mining:"), 2, 0)
        self.mining_status_label = QLabel("Stopped")
        self.mining_status_label.setObjectName("status_warning")
        layout.addWidget(self.mining_status_label, 2, 1)
        
        # Stats
        layout.addWidget(QLabel("Tasks Completed:"), 0, 2)
        self.tasks_completed_label = QLabel("0")
        layout.addWidget(self.tasks_completed_label, 0, 3)
        
        layout.addWidget(QLabel("Uptime:"), 1, 2)
        self.uptime_label = QLabel("00:00:00")
        layout.addWidget(self.uptime_label, 1, 3)
        
        layout.addWidget(QLabel("Last Activity:"), 2, 2)
        self.last_activity_label = QLabel("Never")
        layout.addWidget(self.last_activity_label, 2, 3)
        
        return group
    
    def _create_progress_section(self) -> QWidget:
        """Create progress monitoring section."""
        group = QGroupBox("Progress")
        layout = QVBoxLayout(group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        layout.addWidget(self.progress_bar)
        
        # Current operation
        self.current_operation_label = QLabel("Ready to start mining")
        self.current_operation_label.setObjectName("info")
        layout.addWidget(self.current_operation_label)
        
        return group
    
    def _create_logs_section(self) -> QWidget:
        """Create logs monitoring section."""
        group = QGroupBox("Mining Logs")
        layout = QVBoxLayout(group)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)
        
        # Initialize enhanced log manager
        self.log_manager = ModernLogManager(self.log_text, max_lines=1000)
        
        # Log controls
        log_controls = QHBoxLayout()
        
        clear_logs_btn = QPushButton("Clear Logs")
        clear_logs_btn.setObjectName("secondary_button")
        clear_logs_btn.clicked.connect(self._clear_logs)
        log_controls.addWidget(clear_logs_btn)
        
        log_controls.addStretch()
        layout.addLayout(log_controls)
        
        return group
    
    def _start_mining(self):
        """Start mining operations."""
        # Validate prerequisites
        wallet = self.main_window.get_wallet()
        if not wallet:
            QMessageBox.warning(
                self, 
                "Wallet Required", 
                "Please setup a wallet before starting mining."
            )
            return
        
        client = self.main_window.get_client()
        if not client:
            QMessageBox.warning(
                self, 
                "Connection Required", 
                "Client connection not available. Please check your wallet setup."
            )
            return
        
        try:
            # Get settings
            max_cycles = int(self.max_cycles_edit.text() or "0")
            
            # Reset stop flag
            stop_flag = self.main_window.get_stop_flag()
            stop_flag.reset()
            
            # Create and start mining task
            self.mining_task = MiningTask(
                client=client,
                max_cycles=max_cycles,
                cycle_timeout=120,
                stop_flag=stop_flag
            )
            
            # Connect signals
            self.mining_task.signals.progress.connect(self._update_progress)
            self.mining_task.signals.log.connect(self._add_log)
            self.mining_task.signals.finished.connect(self._mining_finished)
            self.mining_task.signals.error.connect(self._mining_error)
            
            # Start task
            QThreadPool.globalInstance().start(self.mining_task)
            
            # Update UI state
            self.is_mining = True
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.mining_status_label.setText("Running")
            self.mining_status_label.setObjectName("status_success")
            
            self._add_log("INFO", "Mining started successfully")
            
        except ValueError:
            QMessageBox.warning(
                self, 
                "Invalid Settings", 
                "Please enter a valid number for max cycles."
            )
        except Exception as e:
            QMessageBox.critical(
                self, 
                "Error", 
                f"Failed to start mining:\n{str(e)}"
            )
            logger.error(f"Error starting mining: {e}")
    
    def _stop_mining(self):
        """Stop mining operations."""
        if self.mining_task:
            self.mining_task.stop()
            self._add_log("INFO", "Stop signal sent to mining task")
    
    def _mining_finished(self):
        """Handle mining task completion."""
        self.is_mining = False
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.mining_status_label.setText("Stopped")
        self.mining_status_label.setObjectName("status_warning")
        self.mining_task = None
    
    def _mining_error(self, error_msg: str):
        """Handle mining errors."""
        QMessageBox.critical(self, "Mining Error", error_msg)
        self._mining_finished()
    
    def _update_progress(self, value: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(value)
        self.current_operation_label.setText(message)
    
    def _add_log(self, level: str, message: str):
        """Add a log entry using the enhanced log manager."""
        self.log_manager.append(level.upper(), message)
    
    def _clear_logs(self):
        """Clear the log text area."""
        self.log_manager.clear()
    
    def update_mining_status(self):
        """Update mining status displays."""
        # Update wallet status
        wallet = self.main_window.get_wallet()
        if wallet:
            self.wallet_status_label.setText(f"Loaded: {wallet.name}")
            self.wallet_status_label.setObjectName("status_success")
        else:
            self.wallet_status_label.setText("Not loaded")
            self.wallet_status_label.setObjectName("status_error")
        
        # Update connection status
        client = self.main_window.get_client()
        if client:
            self.connection_status_label.setText("Connected")
            self.connection_status_label.setObjectName("status_success")
        else:
            self.connection_status_label.setText("Disconnected")
            self.connection_status_label.setObjectName("status_error")
        
        # Force style refresh
        self.wallet_status_label.style().unpolish(self.wallet_status_label)
        self.wallet_status_label.style().polish(self.wallet_status_label)
        self.connection_status_label.style().unpolish(self.connection_status_label)
        self.connection_status_label.style().polish(self.connection_status_label)
    
    def on_screen_activated(self):
        """Called when mining screen becomes active."""
        self.update_mining_status() 