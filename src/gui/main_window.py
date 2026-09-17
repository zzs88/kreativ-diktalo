"""
Main Window

Kreatív Diktáló fő alkalmazás ablak
"""
import sys
import ctypes
from pathlib import Path
from typing import Optional, Any

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QMenuBar, QMenu, QStatusBar, QMessageBox, QLabel, QSplitter, QSystemTrayIcon, QApplication
)
from PyQt6.QtCore import Qt, pyqtSlot, QObject, QTimer
from PyQt6.QtGui import QAction, QCloseEvent, QIcon

from src.main import KreativDiktalo
# NOTE: SpeechToText is NOT imported here - lazy import in _on_whisper_loaded/_create_backend_with_stt
# to avoid crashing c10.dll (torch) when whisper provider is not used
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger

from src.gui.signals import ApplicationSignals
from src.gui.state_machine import StateMachine, AppState
from src.gui.worker_threads import WhisperLoadWorker, ProcessingWorker
from src.gui.widgets import StatusIndicator, WaveformWidget, TranscriptionDisplay, ToastManager, HistoryPanel
from src.gui.settings_dialog import SettingsDialog
from src.gui.splash_screen import SplashScreen

logger = get_logger()


class CallbackBridge(QObject):
    """
    Bridge between non-Qt callbacks and Qt signals

    A core modulok callback-jei (amelyek nem Qt thread-ben futnak)
    át lesznek irányítva Qt signal-okra thread-safe módon.
    """

    def __init__(self, signals: ApplicationSignals):
        super().__init__()
        self.signals = signals

    def recording_started_callback(self):
        """AudioRecorder.on_recording_started bridge"""
        self.signals.recording_started.emit()

    def recording_stopped_callback(self):
        """AudioRecorder.on_recording_stopped bridge"""
        self.signals.recording_stopped.emit()

    def audio_chunk_callback(self, chunk):
        """AudioRecorder.on_audio_chunk bridge"""
        self.signals.audio_chunk_received.emit(chunk)


class MainWindow(QMainWindow):
    """
    Fő alkalmazás ablak

    Koordinálja az összes GUI komponenst és integrálja a KreativDiktalo backend-et.
    """

    def __init__(self, config_path: str = "config.yaml"):
        super().__init__()

        self.config_path = config_path
        self.config: Optional[ConfigManager] = None
        self.backend: Optional[KreativDiktalo] = None
        self.callback_bridge: Optional[CallbackBridge] = None
        self.splash: Optional[SplashScreen] = None
        self.current_worker: Optional[ProcessingWorker] = None
        self._target_hwnd: int = 0  # Target window HWND to restore focus before paste

        # Signal hub
        self.signals = ApplicationSignals()

        # State machine
        self.state_machine = StateMachine(self.signals)

        # Toast notification manager
        self.toast_manager = ToastManager()

        # Listener restart guard flag
        self._listener_restarting = False

        # UI setup
        try:
            logger.info(">>> _setup_ui indul")
            self._setup_ui()
            logger.info(">>> _setup_ui KESZ")
            self._connect_signals()
            logger.info(">>> _connect_signals KESZ")
            self._setup_system_tray()
            logger.info(">>> _setup_system_tray KESZ")
            # Session monitoring halasztva - az ablak megjelenítése UTÁN fut (event loop kell)
            QTimer.singleShot(1500, self._setup_session_monitoring)
            logger.info(">>> _setup_session_monitoring kezelessel halasztva")
            self._start_health_check_timer()
            logger.info(">>> _start_health_check_timer KESZ")
            self._initialize_backend()
            logger.info(">>> _initialize_backend ELINDULT")
        except Exception as e:
            logger.error(f"KRITO HIBA az __init__ soran: {e}", exc_info=True)
            raise

    def _setup_ui(self):
        """UI komponensek felépítése"""
        self.setWindowTitle("Kreatív Diktáló")

        # Set window icon
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.ico"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Load window size from config (will be loaded later, use defaults for now)
        self.resize(900, 700)

        # === Menu Bar ===
        self._create_menu_bar()

        # === Central Widget ===
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # === Status Indicator ===
        self.status_indicator = StatusIndicator()
        main_layout.addWidget(self.status_indicator)

        # === Waveform Widget ===
        waveform_label = QLabel("Valós idejű audio:")
        waveform_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(waveform_label)

        self.waveform_widget = WaveformWidget()
        main_layout.addWidget(self.waveform_widget)

        # === Transcription Display + History Panel (Splitter) ===
        transcription_label = QLabel("Diktálás eredmény:")
        transcription_label.setStyleSheet("font-weight: bold;")
        main_layout.addWidget(transcription_label)

        # Horizontal splitter for transcription and history
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: Transcription display
        self.transcription_display = TranscriptionDisplay()
        content_splitter.addWidget(self.transcription_display)

        # Right: History panel
        self.history_panel = HistoryPanel(max_items=50)
        content_splitter.addWidget(self.history_panel)

        # Set initial sizes (60% transcription, 40% history)
        content_splitter.setSizes([600, 400])

        main_layout.addWidget(content_splitter, stretch=1)

        # === Status Bar ===
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Inicializálás...", 3000)

        # Apply dark theme
        self._apply_dark_theme()

    def _setup_system_tray(self):
        """Setup system tray icon and menu"""
        # Create tray icon
        icon_path = Path(__file__).parent.parent.parent / "assets" / "icon.ico"
        if not icon_path.exists():
            # Fallback to default icon if custom icon doesn't exist
            self.tray_icon = QSystemTrayIcon(self)
        else:
            self.tray_icon = QSystemTrayIcon(QIcon(str(icon_path)), self)

        # Create tray menu
        tray_menu = QMenu()

        # Show/Hide action
        show_action = QAction("Megjelenítés", self)
        show_action.triggered.connect(self.show)
        tray_menu.addAction(show_action)

        hide_action = QAction("Elrejtés", self)
        hide_action.triggered.connect(self.hide)
        tray_menu.addAction(hide_action)

        tray_menu.addSeparator()

        # Settings action
        settings_action = QAction("Beállítások", self)
        settings_action.triggered.connect(self._open_settings)
        tray_menu.addAction(settings_action)

        # Restart listener action
        restart_action = QAction("Listener újraindítása", self)
        restart_action.triggered.connect(self._restart_listener)
        tray_menu.addAction(restart_action)

        tray_menu.addSeparator()

        # Quit action
        quit_action = QAction("Kilépés", self)
        quit_action.triggered.connect(self._quit_application)
        tray_menu.addAction(quit_action)

        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.setToolTip("Kreatív Diktáló")

        # Double-click to show/hide
        self.tray_icon.activated.connect(self._on_tray_activated)

        # Show tray icon
        self.tray_icon.show()

    def _on_tray_activated(self, reason):
        """Handle tray icon activation"""
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            if self.isVisible():
                self.hide()
            else:
                self.show()
                self.activateWindow()

    def _create_menu_bar(self):
        """Menu bar létrehozása"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&Fájl")

        exit_action = QAction("&Kilépés", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Settings menu
        settings_menu = menubar.addMenu("&Beállítások")

        preferences_action = QAction("&Preferenciák...", self)
        preferences_action.setShortcut("Ctrl+,")
        preferences_action.triggered.connect(self._open_settings)
        settings_menu.addAction(preferences_action)

        # Help menu
        help_menu = menubar.addMenu("&Súgó")

        about_action = QAction("&Névjegy", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _connect_signals(self):
        """Signal/slot kapcsolatok létrehozása"""
        # State changes -> Status indicator
        self.signals.state_changed.connect(self.status_indicator.on_state_changed)

        # Recording signals -> Waveform widget
        self.signals.recording_started.connect(self.waveform_widget.on_recording_started)
        self.signals.recording_stopped.connect(self.waveform_widget.on_recording_stopped)
        self.signals.audio_chunk_received.connect(self.waveform_widget.on_audio_chunk)

        # Processing signals -> Transcription display
        self.signals.transcription_complete.connect(self.transcription_display.set_raw_text)
        self.signals.cleaning_complete.connect(self.transcription_display.set_cleaned_text)

        # Processing signals -> History panel (add completed dictations)
        self.signals.transcription_complete.connect(self._on_transcription_for_history)
        self.signals.cleaning_complete.connect(self._on_cleaning_for_history)

        # History panel -> Paste action
        self.history_panel.paste_requested.connect(self._on_history_paste_requested)

        # Error signals -> Error dialog
        self.signals.error_occurred.connect(self._show_error_dialog)

        # Status messages -> Status bar
        self.signals.status_message.connect(self.status_bar.showMessage)

        # DEBUG: Check signal connections
        logger.info("=== Signal connections established ===")
        self.signals.debug_signal_connections()

    def _initialize_backend(self):
        """Backend inicializálás (háttérben)"""
        # Show splash screen
        self.splash = SplashScreen()
        self.splash.show()

        # Load config
        try:
            self.config = ConfigManager(self.config_path)
        except Exception as e:
            logger.error(f"Config betöltési hiba: {e}", exc_info=True)
            QMessageBox.critical(
                self,
                "Konfigurációs hiba",
                f"Nem sikerült betölteni a konfigurációt:\n\n{str(e)}"
            )
            sys.exit(1)

        # Apply window settings from config
        width = self.config.get('ui.window_width', 900)
        height = self.config.get('ui.window_height', 700)
        self.resize(width, height)

        # Start Whisper loading in background
        self.whisper_worker = WhisperLoadWorker(self.config_path)
        self.whisper_worker.progress_updated.connect(self.splash.update_progress)
        self.whisper_worker.model_loaded.connect(self._on_whisper_loaded)
        self.whisper_worker.error_occurred.connect(self._on_whisper_load_error)
        self.whisper_worker.start()

    @pyqtSlot(object)
    def _on_whisper_loaded(self, stt: Any):
        """
        Whisper modell sikeresen betöltve

        Args:
            stt: Betöltött SpeechToText instance
        """
        logger.info("Whisper modell betöltve, backend inicializálása...")

        try:
            # Create backend (pass pre-loaded STT)
            self._create_backend_with_stt(stt)

            # Bridge callbacks to signals
            self._bridge_callbacks()

            # Replace blocking _process_recording
            self.backend._process_recording = self._process_recording_threaded

            # Close splash
            self.splash.finish_loading(self)

            # Transition to IDLE
            self.state_machine.transition_to(AppState.IDLE)
            self.signals.emit_status("Készen áll!", 3000)

            logger.info("Alkalmazás sikeresen inicializálva")

        except Exception as e:
            logger.error(f"Backend inicializálási hiba: {e}", exc_info=True)
            self.splash.close()
            QMessageBox.critical(
                self,
                "Inicializálási hiba",
                f"Nem sikerült inicializálni az alkalmazást:\n\n{str(e)}"
            )
            sys.exit(1)

    def _create_backend_with_stt(self, stt: Any):
        """
        Backend létrehozása pre-loaded STT-vel

        Args:
            stt: Betöltött SpeechToText instance
        """
        # Import here to avoid circular dependency at module level
        import platform
        from src.core.audio_recorder import AudioRecorder
        from src.core.llm_cleaner import build_llm_cleaner
        from src.core.keyboard_sim import KeyboardSimulator

        # Platform-specifikus hotkey listener
        if platform.system() == 'Windows':
            try:
                from src.core.hotkey_listener_windows import WindowsHotkeyListener as HotkeyListener
                logger.info("🪟 Windows-optimalizált hotkey listener használata")
            except ImportError:
                from src.core.hotkey_listener import HotkeyListener
                logger.warning("⚠️ Windows hotkey listener nem elérhető, fallback pynput-ra")
        else:
            from src.core.hotkey_listener import HotkeyListener

        # Create a minimal KreativDiktalo instance
        # We'll manually initialize modules instead of using __init__
        self.backend = KreativDiktalo.__new__(KreativDiktalo)
        self.backend.config = self.config
        self.backend.logger = logger
        self.backend.is_running = False
        self.backend.temp_audio_file = None

        # Audio Recorder
        self.backend.audio_recorder = AudioRecorder(
            sample_rate=self.config.get('audio.sample_rate', 16000),
            channels=self.config.get('audio.channels', 1),
            vad_enabled=self.config.get('audio.vad_enabled', True),
            silence_threshold=self.config.get('audio.silence_threshold', 2.0)
        )

        # Use pre-loaded STT
        self.backend.stt = stt

        # LLM Cleaner
        self.backend.llm = build_llm_cleaner(self.config)

        # Keyboard Simulator
        self.backend.keyboard = KeyboardSimulator(
            typing_speed=self.config.get('keyboard.typing_speed', 0.01),
            paste_mode=self.config.get('keyboard.paste_mode', True),
            delay_before_type=self.config.get('keyboard.delay_before_type', 0.1)
        )

        # Hotkey Listener
        self.backend.hotkey_listener = HotkeyListener()
        dictation_hotkey = self.config.get('hotkeys.dictation', 'F8')
        self.backend.hotkey_listener.register_hotkey(
            dictation_hotkey,
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release
        )

        # Start hotkey listener
        self.backend.hotkey_listener.start()

        logger.info(f"Backend inicializálva - Hotkey: {dictation_hotkey}")

    def _bridge_callbacks(self):
        """Core callback-ek áthidalása Qt signal-okra"""
        self.callback_bridge = CallbackBridge(self.signals)

        # Audio recorder callbacks
        self.backend.audio_recorder.on_recording_started = \
            self.callback_bridge.recording_started_callback
        self.backend.audio_recorder.on_recording_stopped = \
            self.callback_bridge.recording_stopped_callback
        self.backend.audio_recorder.on_audio_chunk = \
            self.callback_bridge.audio_chunk_callback

    def _on_hotkey_press(self):
        """Hotkey lenyomva - rögzítés indítása"""
        logger.info("Hotkey pressed")

        # Capture the target window HWND IMMEDIATELY (before any Qt UI updates
        # that could steal focus from the target application)
        try:
            self._target_hwnd = ctypes.windll.user32.GetForegroundWindow()
            logger.debug(f"Target HWND captured: {self._target_hwnd}")
        except Exception:
            self._target_hwnd = 0

        # Check state - don't start if still processing
        current_state = self.state_machine.current_state
        if current_state != AppState.IDLE:
            logger.warning(f"Hotkey pressed but state is {current_state}, ignoring")
            self.toast_manager.show(f"⏳ Várj, még feldolgozás folyik...", duration=2000, toast_type='warning')
            return

        if not self.backend.audio_recorder.start_recording():
            logger.error("Rögzítés indítása sikertelen (mikrofon hiba), visszaállás idle állapotba")
            self.toast_manager.show("⚠️ Mikrofon hiba, próbáld újra", duration=3000, toast_type='warning')
            return

        self.state_machine.transition_to(AppState.RECORDING)
        self.signals.emit_status("Rögzítés...", 0)

    def _on_hotkey_release(self):
        """Hotkey felengedve - rögzítés leállítása"""
        logger.info("Hotkey released")
        self.backend.audio_recorder.stop_recording()
        self.signals.emit_status("Feldolgozás...", 0)

        # Start processing
        self._process_recording_threaded()

    def _process_recording_threaded(self):
        """
        Feldolgozás worker thread-ben (non-blocking)

        Ez a metódus helyettesíti a backend blocking _process_recording-ját
        """
        try:
            # Get audio data directly (no file, no ffmpeg needed!)
            audio_data = self.backend.audio_recorder.get_recorded_audio()
            if audio_data is None or len(audio_data) == 0:
                logger.error("Nincs rögzített audio adat")
                self.signals.emit_error("Audio hiba", "Nincs rögzített audio")
                self.state_machine.transition_to(AppState.IDLE)
                return

            sample_rate = self.backend.audio_recorder.sample_rate

            # Create processing worker with numpy array
            self.current_worker = ProcessingWorker(
                audio_data,
                sample_rate,
                self.backend.stt,
                self.backend.llm,
                self.backend.keyboard,
                self.backend.config,
                target_hwnd=self._target_hwnd
            )

            # Connect signals
            self.current_worker.stage_changed.connect(self._on_processing_stage_changed)
            self.current_worker.transcription_complete.connect(self._on_transcription_complete)
            self.current_worker.cleaning_complete.connect(self._on_cleaning_complete)
            self.current_worker.clipboard_fallback.connect(self._on_clipboard_fallback)
            self.current_worker.processing_complete.connect(self._on_processing_complete)
            self.current_worker.error_occurred.connect(self._on_processing_error)

            # Transition to PROCESSING
            self.state_machine.transition_to(AppState.PROCESSING)

            # Start worker
            self.current_worker.start()

        except Exception as e:
            logger.error(f"Processing hiba: {e}", exc_info=True)
            self.signals.emit_error("Feldolgozási hiba", str(e))
            self.state_machine.transition_to(AppState.IDLE)

    @pyqtSlot(str)
    def _on_transcription_complete(self, raw_text: str):
        """Worker transcription complete - forward to ApplicationSignals"""
        logger.info(f"=== MAIN WINDOW === Transcription complete received: '{raw_text[:50]}'")
        logger.info("Emitting ApplicationSignals.transcription_complete")
        self.signals.transcription_complete.emit(raw_text)
        logger.info("Signal emitted!")

    @pyqtSlot(str)
    def _on_cleaning_complete(self, cleaned_text: str):
        """Worker cleaning complete - forward to ApplicationSignals"""
        logger.info(f"=== MAIN WINDOW === Cleaning complete received: '{cleaned_text[:50]}'")
        logger.info("Emitting ApplicationSignals.cleaning_complete")
        self.signals.cleaning_complete.emit(cleaned_text)
        logger.info("Signal emitted!")

    @pyqtSlot(str)
    def _on_processing_stage_changed(self, stage: str):
        """Processing stage changed"""
        stage_names = {
            "transcribing": "Beszédfelismerés...",
            "cleaning": "Szövegtisztítás...",
            "typing": "Beírás..."
        }
        message = stage_names.get(stage, stage)
        self.signals.emit_status(message, 0)

    @pyqtSlot()
    def _on_processing_complete(self):
        """Processing successfully completed"""
        logger.info("Processing complete")
        self.state_machine.transition_to(AppState.IDLE)
        self.signals.emit_status("Kész!", 3000)

        # Proper worker cleanup
        if self.current_worker:
            self.current_worker.quit()
            self.current_worker.wait()
            self.current_worker.deleteLater()
            self.current_worker = None

    @pyqtSlot(str, str)
    def _on_processing_error(self, title: str, message: str):
        """Processing error occurred"""
        logger.error(f"Processing error: {title} - {message}")
        self.signals.emit_error(title, message)
        self.state_machine.transition_to(AppState.ERROR)

        # Proper worker cleanup
        if self.current_worker:
            self.current_worker.quit()
            self.current_worker.wait()
            self.current_worker.deleteLater()
            self.current_worker = None

        # Recovery to IDLE after 3 seconds (only if still in ERROR state)
        def _recover_from_error():
            if self.state_machine.current_state == AppState.ERROR:
                self.state_machine.transition_to(AppState.IDLE)
        QTimer.singleShot(3000, _recover_from_error)

    @pyqtSlot(str)
    def _on_clipboard_fallback(self, message: str):
        """Clipboard fallback - show toast notification"""
        logger.info(f"Clipboard fallback: {message}")
        self.toast_manager.show(
            "📋 " + message,
            duration=5000,
            toast_type='warning'
        )

    # === HISTORY PANEL SLOTS ===

    def _on_transcription_for_history(self, raw_text: str):
        """Store raw text temporarily for history"""
        self._pending_history_raw = raw_text

    @pyqtSlot(str)
    def _on_cleaning_for_history(self, cleaned_text: str):
        """Add completed dictation to history"""
        raw_text = getattr(self, '_pending_history_raw', '')
        if raw_text:
            self.history_panel.add_item(raw_text, cleaned_text)
            delattr(self, '_pending_history_raw')

    @pyqtSlot(str)
    def _on_history_paste_requested(self, text: str):
        """User wants to paste text from history"""
        logger.info(f"History paste requested: {text[:50]}")
        # Use keyboard simulator to paste
        result = self.backend.keyboard.type_text(text, smart_paste=True)
        if not result['success']:
            if result.get('clipboard'):
                self.toast_manager.show("📋 Szöveg clipboard-on - Ctrl+V", duration=3000, toast_type='warning')
            else:
                self.toast_manager.show("❌ Beillesztés sikertelen", duration=3000, toast_type='error')

    @pyqtSlot(str)
    def _on_whisper_load_error(self, error_msg: str):
        """Whisper loading failed"""
        logger.error(f"Whisper load error: {error_msg}")
        self.splash.close()
        QMessageBox.critical(
            self,
            "Whisper betöltési hiba",
            f"Nem sikerült betölteni a Whisper modellt:\n\n{error_msg}"
        )
        sys.exit(1)

    @pyqtSlot(str, str)
    def _show_error_dialog(self, title: str, message: str):
        """Show error dialog"""
        QMessageBox.critical(self, title, message)

    def _open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self.config, self)
        dialog.settings_changed.connect(self._on_settings_changed)
        dialog.exec()

    def _on_settings_changed(self):
        """Handle settings changes"""
        logger.info("Settings changed - applying theme")

        # Apply theme immediately
        self._apply_theme()

        self.toast_manager.show(
            "Beállítások mentve és alkalmazva!",
            duration=3000,
            toast_type='success'
        )

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "Névjegy - Kreatív Diktáló",
            "<h2>Kreatív Diktáló</h2>"
            "<p>Intelligens magyar diktáló alkalmazás</p>"
            "<p><b>Funkciók:</b></p>"
            "<ul>"
            "<li>Lokális Whisper beszédfelismerés</li>"
            "<li>LLM-alapú szövegtisztítás (Ollama)</li>"
            "<li>Automatikus beírás</li>"
            "</ul>"
            "<p>Verzió: 1.0.0</p>"
            "<p>Licensz: MIT</p>"
        )

    def _apply_theme(self):
        """Apply theme stylesheet based on config"""
        theme = self.config.get('ui.theme', 'dark') if self.config else 'dark'

        if theme == 'dark':
            stylesheet = """
                QMainWindow {
                    background-color: #1E1E1E;
                }
                QWidget {
                    background-color: #1E1E1E;
                    color: #FFFFFF;
                }
                QLabel {
                    color: #FFFFFF;
                }
                QMenuBar {
                    background-color: #2B2B2B;
                    color: #FFFFFF;
                }
                QMenuBar::item:selected {
                    background-color: #404040;
                }
                QMenu {
                    background-color: #2B2B2B;
                    color: #FFFFFF;
                }
                QMenu::item:selected {
                    background-color: #404040;
                }
                QStatusBar {
                    background-color: #2B2B2B;
                    color: #AAAAAA;
                }
            """
        else:  # light theme
            stylesheet = """
                QMainWindow {
                    background-color: #FFFFFF;
                }
                QWidget {
                    background-color: #FFFFFF;
                    color: #000000;
                }
                QLabel {
                    color: #000000;
                }
                QMenuBar {
                    background-color: #F0F0F0;
                    color: #000000;
                }
                QMenuBar::item:selected {
                    background-color: #E0E0E0;
                }
                QMenu {
                    background-color: #F0F0F0;
                    color: #000000;
                }
                QMenu::item:selected {
                    background-color: #E0E0E0;
                }
                QStatusBar {
                    background-color: #F0F0F0;
                    color: #666666;
                }
            """

        self.setStyleSheet(stylesheet)
        logger.info(f"Theme applied: {theme}")

    def _apply_dark_theme(self):
        """Legacy method for backwards compatibility"""
        self._apply_theme()

    # === SESSION MONITORING & WATCHDOG ===

    def _setup_session_monitoring(self):
        """Windows session lock/unlock értesítések regisztrálása (WM_WTSSESSION_CHANGE)"""
        if sys.platform != 'win32':
            return
        try:
            hwnd = int(self.winId())
            NOTIFY_FOR_THIS_SESSION = 0
            result = ctypes.windll.wtsapi32.WTSRegisterSessionNotification(hwnd, NOTIFY_FOR_THIS_SESSION)
            if result:
                logger.info("Windows session monitoring aktiválva (lock/unlock detektálás)")
            else:
                logger.warning("Session monitoring regisztráció sikertelen")
        except Exception as e:
            logger.warning(f"Session monitoring setup hiba: {e}")

    def nativeEvent(self, eventType, message):
        """Windows natív üzenetek kezelése – session lock/unlock detektálás

        IMPORTANT: Do NOT call super().nativeEvent() here!
        Calling super() crashes when PortAudio (sounddevice) is loaded alongside Qt,
        due to a COM/DLL conflict. Returning (False, 0) tells Qt to process the
        message through its own C++ pipeline — which is exactly what Qt does by
        default — so NOT calling super() is safe and correct.
        """
        WM_WTSSESSION_CHANGE = 0x02B1
        WTS_SESSION_UNLOCK = 0x8

        # Only parse MSG structure for Windows generic messages with valid pointer.
        # ctypes.from_address() with a null/invalid pointer causes a C-level access
        # violation that Python try/except cannot catch, so guard with addr check.
        if sys.platform == 'win32' and eventType == b'windows_generic_MSG':
            addr = int(message)
            if addr:
                try:
                    class _MSG(ctypes.Structure):
                        _fields_ = [
                            ('hwnd', ctypes.c_size_t),
                            ('message', ctypes.c_uint),
                            ('wParam', ctypes.c_size_t),
                            ('lParam', ctypes.c_size_t),
                            ('time', ctypes.c_uint),
                            ('ptx', ctypes.c_long),
                            ('pty', ctypes.c_long),
                        ]
                    msg = _MSG.from_address(addr)
                    if msg.message == WM_WTSSESSION_CHANGE and msg.wParam == WTS_SESSION_UNLOCK:
                        logger.info("Képernyő feloldva – hotkey listener újraindítása 2mp múlva")
                        QTimer.singleShot(2000, self._restart_listener)
                except Exception:
                    pass  # Natív üzenet feldolgozási hiba nem kritikus

        return False, 0

    def _start_health_check_timer(self):
        """Periodikus watchdog timer: elakadt state detektálás"""
        import time as _time
        self._state_entered_at = _time.time()

        # State-változás időpontjának követése
        self.signals.state_changed.connect(
            lambda old, new: setattr(self, '_state_entered_at', _time.time())
        )

        # 30 másodpercenként ellenőrzés
        self._health_timer = QTimer(self)
        self._health_timer.timeout.connect(self._check_health)
        self._health_timer.start(30000)
        logger.info("Watchdog timer elindult (30mp)")

    def _check_health(self):
        """Watchdog: ha > 120mp óta nem IDLE/INITIALIZING, auto-reset"""
        import time as _time
        if self.backend is None:
            return  # Backend még nem kész

        current = self.state_machine.current_state
        if current in [AppState.IDLE, AppState.INITIALIZING]:
            return  # Normális állapot

        time_in_state = _time.time() - getattr(self, '_state_entered_at', _time.time())
        if time_in_state > 120:
            logger.warning(
                f"Watchdog: '{current.value}' állapot {time_in_state:.0f} másodperce elakadt – auto-reset"
            )
            self._restart_listener()

    def _restart_listener(self):
        """Hotkey listener és state machine újraindítása (session unlock / watchdog / manuális)"""
        if self.backend is None:
            logger.warning("Backend még nincs inicializálva – újraindítás kihagyva")
            return

        if self._listener_restarting:
            logger.debug("Listener újraindítás már folyamatban van")
            return

        self._listener_restarting = True
        logger.info("Listener újraindítása...")

        try:
            # Aktív rögzítés leállítása
            if self.backend.audio_recorder.is_recording:
                self.backend.audio_recorder.stop_recording()

            # State machine reset ha elakadt
            current = self.state_machine.current_state
            if current != AppState.IDLE:
                logger.info(f"State reset: {current.value} → IDLE")
                self.state_machine.transition_to(AppState.IDLE, force=True)

            # Hotkey listener leállítása, majd késleltett újraindítás
            self.backend.hotkey_listener.stop()
            QTimer.singleShot(500, self._do_restart_listener)

        except Exception as e:
            logger.error(f"Listener újraindítás hiba: {e}", exc_info=True)
            self._listener_restarting = False

    def _do_restart_listener(self):
        """Listener tényleges újraindítása (500ms késleltetéssel hívva)"""
        try:
            self.backend.hotkey_listener.start()
            logger.info("✅ Hotkey listener sikeresen újraindult")
            self.signals.emit_status("Listener újraindítva", 3000)
        except Exception as e:
            logger.error(f"Listener start hiba: {e}", exc_info=True)
            self.signals.emit_status("Listener újraindítás sikertelen!", 5000)
        finally:
            self._listener_restarting = False

    def _quit_application(self):
        """Properly quit the application (bypass minimize to tray)"""
        # Cleanup
        if self.backend and self.backend.hotkey_listener:
            self.backend.hotkey_listener.stop()
        if self.backend and self.backend.audio_recorder:
            self.backend.audio_recorder.cleanup()

        # Quit the application
        QApplication.quit()

    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        # Check if should minimize to tray
        minimize_to_tray = self.config.get('ui.minimize_to_tray', True) if self.config else True

        if minimize_to_tray and self.tray_icon.isVisible():
            # Minimize to tray instead of closing
            self.hide()
            self.tray_icon.showMessage(
                "Kreatív Diktáló",
                "Az alkalmazás a tálcán fut tovább.",
                QSystemTrayIcon.MessageIcon.Information,
                2000
            )
            event.ignore()
        else:
            # Actually close the app
            # Cleanup
            if self.backend and self.backend.hotkey_listener:
                self.backend.hotkey_listener.stop()
            if self.backend and self.backend.audio_recorder:
                self.backend.audio_recorder.cleanup()

            event.accept()
