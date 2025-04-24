"""Stop flag implementation for the Neural Network Mining application."""

import logging

logger = logging.getLogger(__name__)

class StopFlag:
    """A simple flag that can be shared across components to signal stopping."""
    
    def __init__(self):
        """Initialize the flag as not stopped."""
        self._is_stopped = False
        logger.debug("StopFlag initialized")
    
    def stop(self):
        """Set the flag to stopped."""
        self._is_stopped = True
        logger.info("StopFlag set to stopped")
    
    def is_stopped(self):
        """Check if the flag is stopped."""
        return self._is_stopped
    
    def reset(self):
        """Reset the flag to not stopped."""
        self._is_stopped = False
        logger.debug("StopFlag reset to not stopped")