#!/usr/bin/env python3
"""Test script for the PySide6 Neural Network Miner GUI."""

import unittest
from unittest.mock import patch, MagicMock
from PySide6.QtWidgets import QApplication
from gui_app_pyside import MiningWindow, RetroTerminalStyle

# Initialize QApplication once for all tests
app = QApplication([])

class TestMiningGUI(unittest.TestCase):
    def test_retro_style(self):
        """Test retro terminal style stylesheet is valid."""
        self.assertIn("background-color: #000000", RetroTerminalStyle.STYLESHEET)
        self.assertIn("color: #00FF00", RetroTerminalStyle.STYLESHEET)
        self.assertIn("QGroupBox", RetroTerminalStyle.STYLESHEET)
        
    def test_window_creation(self):
        """Test main window initialization."""
        window = MiningWindow()
        self.assertIsNotNone(window)
        self.assertEqual(window.windowTitle(), "Neural Network Miner")
        
    @patch('bittensor.wallet')
    @patch('automl_client.client.BittensorPoolClient')
    def test_wallet_connection(self, mock_client, mock_wallet):
        """Test wallet connection functionality."""
        window = MiningWindow()
        mock_wallet.return_value.hotkey.ss58_address = "test_address"
        mock_client.return_value.get_balance.return_value = {"balance": 100.0}
        
        # Simulate connect button click
        window.on_connect_wallet()
        
        # Verify wallet was created
        mock_wallet.assert_called_once_with(name="default", hotkey="default")
        mock_client.assert_called_once()
        self.assertEqual(window.wallet_status.text(), "Status: Connected")
        
    @patch('gui_app_pyside.MiningThread')
    def test_mining_controls(self, mock_thread):
        """Test mining start/stop functionality."""
        window = MiningWindow()
        window.client = MagicMock()
        
        # Mock thread instance
        thread_instance = MagicMock()
        mock_thread.return_value = thread_instance
        
        # Simulate start mining
        window.on_toggle_mining()
        mock_thread.assert_called_once()
        self.assertEqual(window.mining_btn.text(), "Stop Mining")
        
        # Simulate stop mining
        window.on_toggle_mining()
        thread_instance.stop.assert_called_once()
        self.assertEqual(window.mining_btn.text(), "Start Mining")

if __name__ == "__main__":
    unittest.main()
