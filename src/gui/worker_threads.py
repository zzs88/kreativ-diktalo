"""
Worker Threads for Background Operations

QThread-alapú worker osztályok hosszan futó műveletek háttérben történő
végrehajtásához, hogy a GUI ne fagyjon be.
"""
from PyQt6.QtCore import QThread, pyqtSignal
from pathlib import Path
from typing import Optional, Any
import ctypes
import sys
import traceback

# NOTE: SpeechToText is intentionally NOT imported here at module level.
# Importing it would trigger faster_whisper -> ctranslate2 -> torch -> c10.dll
# which crashes on some Windows systems. Instead, it is imported lazily
# inside _load_whisper() only when the whisper provider is actually used.
from src.core.llm_cleaner import LLMCleaner
from src.core.keyboard_sim import KeyboardSimulator
from src.utils.logger import get_logger
from src.utils.config_manager import ConfigManager

logger = get_logger()


class STTLoadWorker(QThread):
    """
    Speech-to-Text modell betöltése (univerzális - Whisper vagy AssemblyAI)

    Különböző STT provider-eket támogat a config alapján.
    """

    # Signals
    progress_updated = pyqtSignal(int, str)  # (percentage, message)
    model_loaded = pyqtSignal(object)  # STT instance
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, config_path: str):
        super().__init__()
        self.config_path = config_path
        self.config: Optional[ConfigManager] = None
        self.stt = None

    def run(self):
        """Thread fő futási ciklusa"""
        try:
            # Config betöltése
            self.progress_updated.emit(10, "Konfiguráció betöltése...")
            self.config = ConfigManager(self.config_path)

            # Check provider
            provider = self.config.get('stt.provider', 'whisper')
            logger.info(f"STT provider: {provider}")

            if provider == 'groq':
                self._load_groq()
            elif provider == 'assemblyai':
                self._load_assemblyai()
            elif provider == 'whisper':
                self._load_whisper()
            else:
                raise ValueError(f"Ismeretlen STT provider: {provider}")

            self.progress_updated.emit(100, "Betöltés kész!")
            self.model_loaded.emit(self.stt)

        except Exception as e:
            error_msg = f"STT modell betöltési hiba: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)

    def _load_groq(self):
        """Groq Whisper inicializálása (WISPR FLOW!)"""
        from src.core.groq_stt import GroqSpeechToText

        self.progress_updated.emit(30, "Groq Whisper inicializálása...")

        api_key = self.config.get('stt.groq.api_key', '')
        language = self.config.get('stt.groq.language', 'hu')
        prompt = self.config.get('stt.groq.prompt', '')

        if not api_key:
            raise ValueError(
                "Groq API key hiányzik!\n\n"
                "1. Regisztrálj: https://console.groq.com/\n"
                "2. Másold ki az API key-t\n"
                "3. Add hozzá a config.yaml-hoz:\n"
                "   stt:\n"
                "     groq:\n"
                "       api_key: \"YOUR_API_KEY\""
            )

        logger.info(f"Groq Whisper inicializálása (language: {language})")
        self.stt = GroqSpeechToText(api_key=api_key, language=language, prompt=prompt)
        logger.info("Groq Whisper készen áll (WISPR FLOW!)")

    def _load_assemblyai(self):
        """AssemblyAI inicializálása"""
        from src.core.assemblyai_stt import AssemblyAISpeechToText

        self.progress_updated.emit(30, "AssemblyAI inicializálása...")

        api_key = self.config.get('stt.assemblyai.api_key', '')
        language = self.config.get('stt.assemblyai.language', 'hu')

        if not api_key:
            raise ValueError(
                "AssemblyAI API key hiányzik!\n\n"
                "1. Regisztrálj: https://www.assemblyai.com/\n"
                "2. Másold ki az API key-t\n"
                "3. Add hozzá a config.yaml-hoz:\n"
                "   stt:\n"
                "     assemblyai:\n"
                "       api_key: \"YOUR_API_KEY\""
            )

        logger.info(f"AssemblyAI inicializálása (language: {language})")
        self.stt = AssemblyAISpeechToText(api_key=api_key, language=language)
        logger.info("AssemblyAI készen áll")

    def _load_whisper(self):
        """Whisper modell betöltése (helyi)"""
        self.progress_updated.emit(30, "Whisper modell betöltése...")

        # Lazy import: only load faster_whisper/ctranslate2/torch when actually needed
        from src.core.speech_to_text import SpeechToText

        model_name = self.config.get('stt.whisper.model', 'small')
        device = self.config.get('stt.whisper.device', 'cpu')
        language = self.config.get('stt.whisper.language', 'hu')

        logger.info(f"Whisper modell betöltése: {model_name} ({device})")

        self.stt = SpeechToText(
            model_name=model_name,
            device=device,
            language=language
        )

        logger.info("✅ Whisper modell betöltve")


class WhisperLoadWorker(STTLoadWorker):
    """
    Whisper modell betöltése háttér thread-ben

    DEPRECATED: Use STTLoadWorker instead.
    This is kept for backwards compatibility.
    """

    def __init__(self, config_path: str):
        super().__init__(config_path)


class ProcessingWorker(QThread):
    """
    Teljes feldolgozási pipeline: STT → LLM → Keyboard

    Ez a worker végzi a fő munkát:
    1. Audio fájl → Whisper STT
    2. Nyers szöveg → LLM tisztítás
    3. Tisztított szöveg → Billentyűzet beírás
    """

    # Signals
    stage_changed = pyqtSignal(str)  # "transcribing", "cleaning", "typing"
    transcription_complete = pyqtSignal(str)  # raw_text
    cleaning_complete = pyqtSignal(str)  # cleaned_text
    processing_complete = pyqtSignal()
    clipboard_fallback = pyqtSignal(str)  # message for toast notification
    error_occurred = pyqtSignal(str, str)  # (title, message)

    def __init__(
        self,
        audio_data,  # numpy array
        sample_rate: int,
        stt: Any,
        llm: LLMCleaner,
        keyboard: KeyboardSimulator,
        config: ConfigManager = None,
        target_hwnd: int = 0
    ):
        super().__init__()
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.stt = stt
        self.llm = llm
        self.keyboard = keyboard
        self.config = config if config else ConfigManager()
        self.target_hwnd = target_hwnd  # Target window to restore focus before paste

    def run(self):
        """Thread fő futási ciklusa"""
        raw_text = ""
        cleaned_text = ""

        try:
            # === 1. SPEECH-TO-TEXT ===
            logger.info("Processing: STT indítása...")
            self.stage_changed.emit("transcribing")

            # Use transcribe_array instead of transcribe_file (no ffmpeg needed!)
            stt_result = self.stt.transcribe_array(self.audio_data, self.sample_rate)
            raw_text = stt_result.get('text', '').strip()

            if not raw_text:
                logger.warning("STT nem adott vissza szöveget")
                self.error_occurred.emit(
                    "Nincs felismert szöveg",
                    "A Whisper nem tudott szöveget felismerni az audiból."
                )
                return

            logger.info(f"STT kész: '{raw_text[:100]}...'")
            logger.info("Emitting transcription_complete signal")
            self.transcription_complete.emit(raw_text)

            # === 2. LLM CLEANING (ha engedélyezve) ===
            enable_cleaning = self.config.get('text_processing.enable_cleaning', True)

            if enable_cleaning:
                logger.info("Processing: LLM tisztítás...")
                self.stage_changed.emit("cleaning")
                cleaned_text = self.llm.clean_text(raw_text)
                logger.info(f"LLM kész: '{cleaned_text[:100]}...'")
            else:
                logger.info("Processing: LLM tisztítás kihagyva (beállítás szerint)")
                cleaned_text = raw_text

            logger.info("Emitting cleaning_complete signal")
            self.cleaning_complete.emit(cleaned_text)

            # === 3. KEYBOARD TYPING (WISPR FLOW STYLE) ===
            logger.info("Processing: Szöveg beírása...")
            self.stage_changed.emit("typing")

            # Restore focus to the original target window before pasting.
            # During STT+LLM processing (which can take several seconds), the
            # dictation app itself may have stolen focus from the target textbox.
            # Without this, Ctrl+V would go to the wrong window.
            if self.target_hwnd and sys.platform == 'win32':
                try:
                    import time as _time
                    ctypes.windll.user32.SetForegroundWindow(self.target_hwnd)
                    _time.sleep(0.15)  # Give Windows time to actually switch focus
                    logger.debug(f"Focus restored to HWND: {self.target_hwnd}")
                except Exception as e:
                    logger.warning(f"Focus restore sikertelen (HWND: {self.target_hwnd}): {e}")

            result = self.keyboard.type_text(cleaned_text, smart_paste=True)

            if not result['success']:
                # Check if text is on clipboard (fallback worked)
                if result.get('clipboard'):
                    # Text is on clipboard - notify user
                    logger.info("Szöveg clipboard-on, auto-paste sikertelen")
                    self.clipboard_fallback.emit(result.get('message', 'Szöveg clipboard-on'))
                    # Still count as success (text is available)
                    self.processing_complete.emit()
                else:
                    # Complete failure
                    logger.error(f"Billentyűzet beírás sikertelen: {result.get('message')}")
                    self.error_occurred.emit(
                        "Beírási hiba",
                        result.get('message', 'A szöveg beírása sikertelen volt.')
                    )
                return

            logger.info("Processing: Sikeres befejezés")
            self.processing_complete.emit()

        except Exception as e:
            error_msg = f"Feldolgozási hiba:\n\n{str(e)}\n\n{traceback.format_exc()}"
            logger.error(error_msg)
            self.error_occurred.emit("Feldolgozási hiba", str(e))


class TranscriptionWorker(QThread):
    """
    Csak STT transcription (önálló használatra)

    Néhány esetben lehet hogy csak az STT-re van szükség külön.
    """

    # Signals
    transcription_complete = pyqtSignal(dict)  # teljes Whisper result
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, audio_file: str, stt: Any):
        super().__init__()
        self.audio_file = audio_file
        self.stt = stt

    def run(self):
        """Thread fő futási ciklusa"""
        try:
            logger.info(f"Transcribing: {self.audio_file}")
            result = self.stt.transcribe_file(self.audio_file)
            self.transcription_complete.emit(result)
        except Exception as e:
            error_msg = f"STT hiba: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)


class LLMCleaningWorker(QThread):
    """
    Csak LLM cleaning (önálló használatra)

    Tisztítás előre megadott szövegen
    """

    # Signals
    cleaning_complete = pyqtSignal(str)  # cleaned text
    error_occurred = pyqtSignal(str)  # error message

    def __init__(self, raw_text: str, llm: LLMCleaner, config: ConfigManager = None):
        super().__init__()
        self.raw_text = raw_text
        self.llm = llm
        self.config = config if config else ConfigManager()

    def run(self):
        """Thread fő futási ciklusa"""
        try:
            enable_cleaning = self.config.get('text_processing.enable_cleaning', True)

            if enable_cleaning:
                logger.info(f"LLM cleaning: '{self.raw_text[:50]}...'")
                cleaned = self.llm.clean_text(self.raw_text)
            else:
                logger.info("LLM cleaning kihagyva (beállítás szerint)")
                cleaned = self.raw_text

            self.cleaning_complete.emit(cleaned)
        except Exception as e:
            error_msg = f"LLM hiba: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.error_occurred.emit(error_msg)
