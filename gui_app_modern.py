"""Modern GUI entry point for AutoML Miner."""

import sys
import logging
from pathlib import Path

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt

# Setup logging before importing application components
def setup_logging():
    """Setup application logging."""
    # Create logs directory
    log_dir = Path.home() / ".automl_pool" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "automl_miner.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific logger levels
    logging.getLogger("automl_client.gui").setLevel(logging.DEBUG)
    logging.getLogger("automl_client.client").setLevel(logging.INFO)
    logging.getLogger("automl_client.wallet").setLevel(logging.INFO)

def main():
    """Main entry point for the modern GUI application."""
    # Setup logging first
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting AutoML Miner Modern GUI")
    
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("AutoML Miner")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Neural Component Pool")
        
        # Enable high DPI scaling
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
        # Import and create main window
        from automl_client.gui import ModernMiningWindow
        
        # Create and show main window
        window = ModernMiningWindow()
        window.show()
        
        logger.info("GUI initialized successfully")
        
        # Run the application event loop
        exit_code = app.exec()
        
        logger.info(f"Application exiting with code: {exit_code}")
        return exit_code
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        print(f"Error: Failed to import required modules: {e}")
        print("Make sure PySide6 is installed: pip install PySide6")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 