"""Centralized logging configuration for Talon Assistant."""
import logging
import logging.handlers
import os
import sys


def setup_logging(log_dir="data/logs", level=logging.DEBUG):
    """Initialize logging with console + rotating file handlers."""
    os.makedirs(log_dir, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Clear any existing handlers (prevents duplicates on reload)
    root.handlers.clear()

    # Console handler — INFO and above, concise format
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(
        "%(message)s"  # Keep console output clean like current print()
    ))
    root.addHandler(console)

    # File handler — DEBUG and above, detailed format, rotating
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "talon.log"),
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in ("chromadb", "httpx", "urllib3", "sentence_transformers",
                 "transformers", "torch", "onnxruntime", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.info("Logging initialized (file: %s)",
                 os.path.join(log_dir, "talon.log"))
