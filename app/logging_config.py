"""Shared logging configuration for the RAG application."""

import logging
import os


def get_logger(name: str) -> logging.Logger:
    """Return an application logger, configuring console output on first use."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
