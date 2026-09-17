"""
GUI Mode Entry Point

Kreatív Diktáló indítása grafikus felülettel
"""
import sys
import os
import ctypes
from pathlib import Path

# Fix Windows symlink issues for HuggingFace cache
os.environ['HF_HUB_DISABLE_SYMLINKS'] = '1'

# Windows: Set custom AppUserModelID so the taskbar shows the app icon,
# not the Python interpreter icon.
if sys.platform == 'win32':
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "KreativDiktalo.DiktaloProg.1"
        )
    except Exception:
        pass

# Projekt gyökér hozzáadása a path-hoz
sys.path.insert(0, str(Path(__file__).parent.parent))

# IMPORTANT: Import torch before PyQt6 to avoid DLL loading issues on Windows.
# Torch is only needed for the local Whisper provider. For the Groq (cloud)
# provider it is unnecessary, so make the import optional to allow a lightweight,
# torch-free install.
try:
    import torch
except ImportError:
    torch = None

from src.gui.app import KreativDiktaloGUI
from src.gui.main_window import MainWindow
from src.utils.logger import setup_logger, get_logger


def main():
    """Main entry point for GUI mode"""
    # Projekt gyökér
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"

    # Logger setup
    log_file = project_root / "data" / "logs" / "app.log"
    setup_logger(level="INFO", log_file=str(log_file))

    logger = get_logger()
    logger.info("=" * 60)
    logger.info("Kreatív Diktáló GUI Mode - Indítás")
    logger.info("=" * 60)

    try:
        # Create Qt Application
        app = KreativDiktaloGUI(sys.argv)

        # Create main window
        main_window = MainWindow(str(config_path))

        # IMPORTANT: Show and activate the window!
        main_window.show()
        main_window.raise_()
        main_window.activateWindow()

        # Start event loop
        sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Kritikus hiba: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
