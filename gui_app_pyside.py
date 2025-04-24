"""Enhanced PySide6 GUI for the Neural Network Miner – now with flexible wallet
handling (create / import by phrase / load from disk) and user‑selectable wallet
root directory. Compatible with the new cross‑platform wallet implementation.
"""

import sys
import os
import time
import logging
import queue
import traceback
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QProgressBar, QGroupBox, QLineEdit, QTextEdit, QHBoxLayout, QMessageBox,
    QFileDialog, QInputDialog, QStatusBar
)
from PySide6.QtCore import Qt, Signal, QObject, QTimer, QRunnable, QThreadPool, Slot
from PySide6.QtGui import QTextCursor

from substrateinterface import Keypair

# Local imports
from automl_client.stop_flag import StopFlag
from automl_client.wallet import Wallet
try:
    from automl_client.client import BittensorPoolClient
except ImportError:
    print("Error: Required modules not found. Please install automl_client.")
    sys.exit(1)

LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_queue: "queue.Queue[str]" = queue.Queue()

SETTINGS = {
    "api_url": "http://localhost:8000",
    "max_cycles": 100,
    "cycle_timeout": 120,
}

# --------------------------------------------------------------------------------------
#  Logging helpers
# --------------------------------------------------------------------------------------
class QueueHandler(logging.Handler):
    def __init__(self, log_queue: "queue.Queue[str]"):
        super().__init__()
        self.log_queue = log_queue
        self.setFormatter(logging.Formatter(LOG_FORMAT))

    def emit(self, record):
        self.log_queue.put(self.format(record))

class LogProcessor(QObject):
    log_signal = Signal(str, str)

    @Slot()
    def process_logs(self):
        while not log_queue.empty():
            message = log_queue.get_nowait()
            level = "info"
            if "ERROR" in message:
                level = "error"
            elif "WARNING" in message:
                level = "warning"
            self.log_signal.emit(message, level)

# --------------------------------------------------------------------------------------
#  Miner task wrapper (unchanged except name refactor)
# --------------------------------------------------------------------------------------
class MiningTask(QRunnable):
    class Signals(QObject):
        update = Signal(dict)
        log = Signal(str, str)
        progress = Signal(int, str)
        finished = Signal()
        error = Signal(str)
        stopping = Signal()

    def __init__(self, client, *, max_cycles: int, cycle_timeout: int, stop_flag: StopFlag):
        super().__init__()
        self.client = client
        self.max_cycles = max_cycles
        self.cycle_timeout = cycle_timeout
        self.stop_flag = stop_flag
        self.signals = self.Signals()
        self.setAutoDelete(True)

    def stop(self):
        self.stop_flag.stop()
        if hasattr(self.client, "stop_mining"):
            self.client.stop_mining()
        self.signals.stopping.emit()

    @Slot()
    def run(self):
        try:
            self.signals.progress.emit(0, "Starting mining cycles …")
            self.client.run_continuous_mining(
                cycles=self.max_cycles,
                alternate=True,
                delay=5.0,
                max_retries=3,
                stop_flag=self.stop_flag,
            )
            if self.stop_flag.is_stopped():
                self.signals.progress.emit(100, "Stopped by user.")
            else:
                self.signals.progress.emit(100, "Mining finished ✅")
        except Exception as e:
            self.signals.error.emit(f"Mining error: {e}")
        finally:
            self.signals.finished.emit()

# --------------------------------------------------------------------------------------
#  Retro stylesheet
# --------------------------------------------------------------------------------------
class RetroTerminalStyle:
    STYLESHEET = """
    QWidget           { background:#000; color:#0F0; font-family:"Courier New",monospace; font-size:12px; }
    QGroupBox         { border:1px solid #060; border-radius:4px; margin-top:1em; padding:8px; }
    QGroupBox::title  { color:#0C0; font-weight:bold; }
    QPushButton       { background:#020; border:1px solid #0A0; padding:4px 10px; }
    QProgressBar      { border:1px solid #060; background:#010; }
    QProgressBar::chunk{ background:#040; }
    """

# --------------------------------------------------------------------------------------
#  Main window
# --------------------------------------------------------------------------------------
class MiningWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.wallet: Optional[Wallet] = None
        self.client: Optional[BittensorPoolClient] = None
        self.mining_task: Optional[MiningTask] = None
        self.stop_flag = StopFlag()

        self._build_ui()
        self._configure_logging()

    # ------------------------------------------------------------------ UI construction
    def _build_ui(self):
        self.setWindowTitle("AutoML Miner")
        self.resize(900, 650)
        self.setStyleSheet(RetroTerminalStyle.STYLESHEET)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ------------------------- Server settings
        srv_box = QGroupBox("Server")
        srv_lay = QVBoxLayout(srv_box)
        self.pool_url_edit = QLineEdit(SETTINGS["api_url"])
        srv_lay.addWidget(QLabel("Pool URL:"))
        srv_lay.addWidget(self.pool_url_edit)
        layout.addWidget(srv_box)

        # ------------------------- Wallet settings
        w_box = QGroupBox("Wallet")
        w_lay = QVBoxLayout(w_box)

        self.wallet_dir_edit = QLineEdit(str(Path.home() / ".bittensor" / "wallets"))
        browse_btn = QPushButton("Browse …")
        browse_btn.clicked.connect(self._browse_wallet_dir)
        dir_row = QHBoxLayout(); dir_row.addWidget(self.wallet_dir_edit); dir_row.addWidget(browse_btn)
        w_lay.addWidget(QLabel("Wallet root directory:"))
        w_lay.addLayout(dir_row)

        self.wallet_name_edit = QLineEdit("default")
        self.hotkey_name_edit = QLineEdit("default")
        w_lay.addWidget(QLabel("Wallet name:"));  w_lay.addWidget(self.wallet_name_edit)
        w_lay.addWidget(QLabel("Hotkey name:"));  w_lay.addWidget(self.hotkey_name_edit)

        # action buttons
        actions_row = QHBoxLayout()
        self.load_wallet_btn   = QPushButton("Load from Disk")
        self.new_wallet_btn    = QPushButton("Create New Wallet")
        self.import_phrase_btn = QPushButton("Import by Phrase")
        for b in (self.load_wallet_btn, self.new_wallet_btn, self.import_phrase_btn):
            actions_row.addWidget(b)
        w_lay.addLayout(actions_row)

        self.load_wallet_btn.clicked.connect(self._load_wallet)
        self.new_wallet_btn.clicked.connect(self._create_wallet)
        self.import_phrase_btn.clicked.connect(self._import_wallet)

        layout.addWidget(w_box)

        # ------------------------- Mining settings
        m_box = QGroupBox("Mining")
        m_lay = QVBoxLayout(m_box)
        self.max_cycles_edit   = QLineEdit(str(SETTINGS["max_cycles"]))
        self.cycle_timeout_edit= QLineEdit(str(SETTINGS["cycle_timeout"]))
        m_lay.addWidget(QLabel("Max cycles (0 = ∞):"));   m_lay.addWidget(self.max_cycles_edit)
        m_lay.addWidget(QLabel("Cycle timeout (s):"));     m_lay.addWidget(self.cycle_timeout_edit)

        self.mining_btn = QPushButton("Start Mining")
        self.mining_btn.setEnabled(False)
        self.mining_btn.clicked.connect(self._toggle_mining)
        m_lay.addWidget(self.mining_btn)
        layout.addWidget(m_box)

        # ------------------------- Progress + log
        prog_row = QHBoxLayout()
        prog_row.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar(); self.progress_bar.setValue(0)
        prog_row.addWidget(self.progress_bar)
        layout.addLayout(prog_row)

        self.log_view = QTextEdit(); self.log_view.setReadOnly(True)
        layout.addWidget(self.log_view, stretch=1)

        # Status bar
        self.status_label = QLabel("Status: ready")
        sb = QStatusBar(); sb.addWidget(self.status_label)
        self.setStatusBar(sb)

        # log polling timer
        self.log_processor = LogProcessor(self)
        self.log_processor.log_signal.connect(self._append_log)
        self.log_timer = QTimer(self); self.log_timer.timeout.connect(self.log_processor.process_logs); self.log_timer.start(150)

    # ------------------------------------------------------------------ Logging setup
    def _configure_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(QueueHandler(log_queue))

    # ------------------------------------------------------------------ Wallet helpers
    def _wallet_root(self) -> str:
        root = os.path.expanduser(self.wallet_dir_edit.text().strip())
        os.makedirs(root, exist_ok=True)
        return root if root.endswith(os.sep) else root + os.sep

    def _browse_wallet_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select wallet directory", self._wallet_root())
        if dir_path:
            self.wallet_dir_edit.setText(dir_path)

    def _build_wallet(self) -> Wallet:
        return Wallet(
            name=self.wallet_name_edit.text().strip() or "default",
            hotkey=self.hotkey_name_edit.text().strip() or "default",
            path=self._wallet_root(),
        )

    # ---------------- create new wallet ----------------
    def _create_wallet(self):
        try:
            w = self._build_wallet()
            # safety: refuse to overwrite existing
            if w.coldkey_file.exists():
                QMessageBox.warning(self, "Wallet exists", "A wallet with that name already exists in this directory.")
                return
            # -- generate keys manually so we can show the mnemonics
            cold_phrase = Keypair.generate_mnemonic(12)
            hot_phrase  = Keypair.generate_mnemonic(12)
            w.import_coldkey_from_mnemonic(cold_phrase, password=None, overwrite=False)
            w.import_hotkey_from_mnemonic(hot_phrase, password=None, overwrite=False)
            self.wallet = w
            QMessageBox.information(self, "New wallet created",
                f"Coldkey mnemonic (WRITE THIS DOWN):\n\n{cold_phrase}\n\nHotkey mnemonic (optional – WRITE THIS TOO):\n\n{hot_phrase}")
            self._connect_client()
        except Exception as e:
            self._append_log(f"Create wallet error: {e}", "error")

    # ---------------- import by phrase ---------------
    def _import_wallet(self):
        phrase, ok = QInputDialog.getMultiLineText(self, "Import wallet", "Enter 12/24‑word mnemonic (coldkey):")
        if not ok or not phrase.strip():
            return
        try:
            w = self._build_wallet()
            w.import_coldkey_from_mnemonic(phrase.strip(), password=None, overwrite=True)
            # Ensure hotkey exists
            if not w.hotkey_file.exists():
                hot_phrase = Keypair.generate_mnemonic(12)
                w.import_hotkey_from_mnemonic(hot_phrase, overwrite=False)
                QMessageBox.information(self, "Hotkey created",
                    f"No hotkey found, generated one automatically:\n\n{hot_phrase}")
            self.wallet = w
            QMessageBox.information(self, "Wallet imported", "Coldkey imported successfully.")
            self._connect_client()
        except Exception as e:
            self._append_log(f"Import error: {e}", "error")

    # ---------------- load from disk ----------------
    def _load_wallet(self):
        try:
            w = self._build_wallet()
            if not w.coldkey_file.exists():
                QMessageBox.warning(self, "Not found", "Coldkey file not found in the specified directory.")
                return
            self.wallet = w
            self._connect_client()
        except Exception as e:
            self._append_log(f"Load wallet error: {e}", "error")

    # ------------------------------------------------------------------ Client connect
    def _connect_client(self):
        try:
            pool_url = self.pool_url_edit.text().strip()
            if not pool_url:
                raise ValueError("Pool URL cannot be empty")
            self.client = BittensorPoolClient(wallet=self.wallet, base_url=pool_url)
            self.status_label.setText("Status: connected")
            self._append_log(f"Wallet connected to {pool_url}")
            self.mining_btn.setEnabled(True)
        except Exception as e:
            self._append_log(f"Connection error: {e}", "error")
            self.status_label.setText("Status: connection failed")
            self.mining_btn.setEnabled(False)

    # ------------------------------------------------------------------ Mining control
    def _toggle_mining(self):
        if self.mining_task:
            # stop
            self.stop_flag.stop()
            self.mining_btn.setEnabled(False)
            self.status_label.setText("Status: stopping…")
            self._append_log("Stopping mining …", "warning")
        else:
            # start
            try:
                max_cycles   = int(self.max_cycles_edit.text())
                cycle_to     = int(self.cycle_timeout_edit.text())
            except ValueError:
                self._append_log("Invalid numeric input", "error")
                return
            self.stop_flag.reset()
            self.mining_task = MiningTask(
                self.client,
                max_cycles=max_cycles,
                cycle_timeout=cycle_to,
                stop_flag=self.stop_flag,
            )
            self.mining_task.signals.progress.connect(self._update_progress)
            self.mining_task.signals.error.connect(lambda msg: self._append_log(msg, "error"))
            self.mining_task.signals.finished.connect(self._mining_finished)
            self.mining_task.signals.stopping.connect(lambda: self._append_log("Stop signal sent", "warning"))
            QThreadPool.globalInstance().start(self.mining_task)
            self.mining_btn.setText("Stop Mining")
            self.status_label.setText("Status: mining …")
            self.progress_bar.setValue(0)

    def _mining_finished(self):
        self.mining_task = None
        self.mining_btn.setText("Start Mining")
        self.mining_btn.setEnabled(True)
        self.status_label.setText("Status: ready")
        self.progress_bar.setValue(100)

    # ------------------------------------------------------------------ UI helpers
    def _update_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self._append_log(f"{pct}% – {msg}")

    def _append_log(self, message: str, level: str = "info"):
        color = {"error": "#F55", "warning": "#FA0", "info": "#0F0"}.get(level, "#0F0")
        self.log_view.append(f"<span style='color:{color}'>{message}</span>")
        cursor = self.log_view.textCursor(); cursor.movePosition(QTextCursor.End); self.log_view.setTextCursor(cursor)

    # ------------------------------------------------------------------ window close
    def closeEvent(self, event):
        if self.mining_task and not self.stop_flag.is_stopped():
            self.stop_flag.stop()
            self._append_log("Gracefully stopping before exit …", "warning")
            for _ in range(10):
                QApplication.processEvents(); time.sleep(0.1)
        event.accept()

# --------------------------------------------------------------------------------------
#  Entrypoint
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MiningWindow(); win.show()
    sys.exit(app.exec())
