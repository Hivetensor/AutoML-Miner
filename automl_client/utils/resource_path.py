"""Utility functions for handling resource paths correctly in bundled applications."""

import os
import sys
import logging

logger = logging.getLogger(__name__)

def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for development and for PyInstaller bundles.
    
    Args:
        relative_path: Path relative to the application's root directory
        
    Returns:
        Absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
        
        # In case we're in a macOS app bundle
        if os.path.exists(os.path.join(base_path, 'Contents', 'Resources')):
            base_path = os.path.join(base_path, 'Contents', 'Resources')
        elif os.path.exists(os.path.join(base_path, 'Resources')):
            base_path = os.path.join(base_path, 'Resources')
            
        abs_path = os.path.join(base_path, relative_path)
        logger.info(f"Resource path resolved: {relative_path} -> {abs_path}")
        return abs_path
    except Exception as e:
        logger.error(f"Error getting resource path for {relative_path}: {e}")
        return os.path.join(os.path.abspath("."), relative_path) 