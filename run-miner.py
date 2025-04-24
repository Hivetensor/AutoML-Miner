#!/usr/bin/env python3
# run-miner.py - Neural Component Pool mining client

import json
import uuid
import argparse
import time
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Import our client implementation
from automl_client.client import BittensorPoolClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Default to INFO level, can be overridden by command line args
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Module-specific logging levels will be set based on command line args


# BittensorPoolClient is now imported from automl_client.client


class BittensorNetwork:
    """Interface to the Bittensor network."""

    @staticmethod
    def create_wallet(wallet_name: str, hotkey_name: str) -> "Any":
        """
        Create or load a Bittensor wallet.

        Args:
            wallet_name: Name of the wallet
            hotkey_name: Name of the hotkey

        Returns:
            Bittensor wallet
        """
        # Import bittensor here to avoid top-level import
        import bittensor as bt
        try:
            # Create configuration
            config = bt.config()
            config.wallet = bt.config()
            config.wallet.name = wallet_name
            config.wallet.hotkey = hotkey_name

            # Check if wallet exists or create it
            wallet = wallet(config=config)

            # Verify wallet was properly created
            if not hasattr(wallet, 'name') or not hasattr(wallet, 'hotkey'):
                logger.error("Wallet missing required attributes")
                return None

            # # Ensure the hotkey exists within the wallet
            # if not wallet.hotkey_file.exists():
            #     logger.info(f"Creating new hotkey: {hotkey_name}")
            #     wallet.create_hotkey(hotkey_name)

            return wallet
        except Exception as e:
            logger.error(f"Error in wallet creation: {e}")
            return None


def run_miner(args):
    """
    Run the mining client with the provided arguments.
    
    Args:
        args: Command line arguments
    """
    # Create a Bittensor wallet
    try:
        wallet = BittensorNetwork.create_wallet(args.wallet, args.hotkey)
        if wallet is None:
            logger.error("Failed to create wallet - wallet object is None")
            return 1
            
        logger.info(f"Using wallet: {wallet.name} with hotkey: {wallet.hotkey_str}")
        logger.info(f"Public address: {wallet.hotkey.ss58_address}")
    except Exception as e:
        logger.error(f"Error creating wallet: {e}")
        return 1
    
    # Initialize pool client
    with BittensorPoolClient(wallet=wallet, base_url=args.pool_url) as client:
        try:
            # Check if connected to pool
            logger.info(f"Connecting to pool at {args.pool_url}")
            
            # Run mining cycles
            logger.info(f"Starting mining with {args.cycles} cycles (0 = infinite)")
            client.run_continuous_mining(
                cycles=args.cycles,
                alternate=not args.no_alternate,
                delay=args.delay
            )
            
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            return 1
    
    return 0


def main():
    """
    Main entry point with argument parsing.
    """
    parser = argparse.ArgumentParser(description="Neural Component Pool Miner")
    
    parser.add_argument(
        "--wallet", 
        type=str, 
        default="default", 
        help="Name of the Bittensor wallet to use"
    )
    parser.add_argument(
        "--hotkey", 
        type=str, 
        default="default", 
        help="Name of the hotkey to use within the wallet"
    )
    parser.add_argument(
        "--pool_url", 
        type=str, 
        default="http://localhost:8000", 
        help="URL of the neural component pool server"
    )
    parser.add_argument(
        "--cycles", 
        type=int, 
        default=0, 
        help="Number of mining cycles to run (0 for infinite)"
    )
    parser.add_argument(
        "--no_alternate", 
        action="store_true", 
        help="Don't alternate between evolution and evaluation tasks"
    )
    parser.add_argument(
        "--delay", 
        type=float, 
        default=5.0, 
        help="Delay between mining cycles in seconds"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="",
        help="Log to the specified file in addition to console"
    )
    
    args = parser.parse_args()

    # Import Bittensor after parsing arguments to avoid CLI hijacking
    import bittensor as bt

    # Set up logging level and file handler if specified
    # Determine log level (--debug flag takes precedence for backward compatibility)
    log_level = logging.DEBUG if args.debug else getattr(logging, args.log_level)
    logging.getLogger().setLevel(log_level)
    
    # Only set specific loggers to DEBUG if debug mode is enabled
    if args.debug:
        for module in ['automl_client.evaluation_strategy', 'automl_client.genetic.interpreter']:
            logging.getLogger(module).setLevel(logging.DEBUG)
        
    if args.log_file:
        # Add file handler to root logger
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {args.log_file}")
    
    # Run the miner
    try:
        exit_code = run_miner(args)
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("Miner stopped by user")
        exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        exit(1)


if __name__ == "__main__":
    main()
