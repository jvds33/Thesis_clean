#!/usr/bin/env python3
"""
Run the ACOS pipeline test with detailed output.
This script runs the test_pipeline.py test with maximum verbosity
to show the detailed execution of the pipeline.
"""

import os
import sys
import pytest
import logging
import colorama
from colorama import Fore, Back, Style

# Initialize colorama
colorama.init(autoreset=True)

# Configure custom formatter with colors
class ColoredFormatter(logging.Formatter):
    FORMATS = {
        logging.DEBUG: Fore.CYAN + "%(message)s" + Style.RESET_ALL,
        logging.INFO: "%(message)s",
        logging.WARNING: Fore.YELLOW + "%(message)s" + Style.RESET_ALL,
        logging.ERROR: Fore.RED + "%(message)s" + Style.RESET_ALL,
        logging.CRITICAL: Fore.RED + Back.WHITE + "%(message)s" + Style.RESET_ALL
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        if "DUMMY LLM" in record.getMessage():
            log_fmt = Fore.GREEN + "%(message)s" + Style.RESET_ALL
        elif "STEP 1:" in record.getMessage():
            log_fmt = Fore.BLUE + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        elif "STEP 2:" in record.getMessage():
            log_fmt = Fore.MAGENTA + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        elif "STEP 3:" in record.getMessage():
            log_fmt = Fore.YELLOW + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        elif "FINAL ACOS" in record.getMessage():
            log_fmt = Fore.CYAN + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        elif "TEST SUMMARY" in record.getMessage():
            log_fmt = Fore.WHITE + Back.BLUE + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        elif "PASSED" in record.getMessage() and "Test result:" in record.getMessage():
            log_fmt = Fore.BLACK + Back.GREEN + Style.BRIGHT + "%(message)s" + Style.RESET_ALL
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logging with custom formatter
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print(Fore.CYAN + Style.BRIGHT + "\n" + "="*80)
    print(Fore.WHITE + Back.BLUE + Style.BRIGHT + "                ACOS PIPELINE TEST WITH DETAILED OUTPUT                " + Style.RESET_ALL)
    print(Fore.CYAN + Style.BRIGHT + "="*80 + "\n")
    
    # Run the test with maximum verbosity
    pytest.main(["-vvs", "src/tests/test_pipeline.py", "--log-cli-level=INFO"]) 