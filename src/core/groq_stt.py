"""
Groq Whisper Speech-to-Text Integration

Ultra-fast Whisper API (same as Wispr Flow uses!)
- <1 second transcription time
- 14,400 minutes/day free tier
- Whisper Large-v3 model
"""
from groq import Groq
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
import soundfile as sf
from src.utils.logger import get_logger

logger = get_logger()


class GroqSpeechToText:
    """
    Groq Whisper-based speech recognition

    This is what Wispr Flow uses!
    Features:
    - Ultra-fast: <1 second transcription
    - Excellent Hungarian support (Whisper Large-v3)
    - Huge free tier: 14,400 minutes/day
    """

    def __init__(self, api_key: str, language: str = "hu", prompt: str = ""):
        """
        Initialize Groq client

        Args:
            api_key: Groq API key
            language: Language code (hu, en, etc.)
            prompt: Optional context/vocabulary hint passed to Whisper to bias
                recognition towards expected terms and punctuation style
        """
        self.api_key = api_key
        self.language = language
        self.prompt = prompt or ""
        self.client = Groq(api_key=api_key)

        logger.info(f"Groq Whisper inicializálva (language: {language}, prompt: {'igen' if self.prompt else 'nincs'})")

    def transcribe_file(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio file

        Args:
            audio_path: Path to audio file

        Returns:
            dict with 'text' and metadata
        """
        logger.info(f"Groq Whisper transcription: {audio_path}")

        try:
            with open(audio_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(Path(audio_path).name, file.read()),
                    model="whisper-large-v3",
                    language=self.language,
                    response_format="verbose_json",
                    prompt=self.prompt,
                    temperature=0.0
                )

            result = {
                'text': transcription.text,
                'language': self.language,
                'duration': getattr(transcription, 'duration', None)
            }

            logger.info(f"Groq transcription kész: {len(result['text'])} karakter")
            return result

        except Exception as e:
            logger.error(f"Groq hiba: {str(e)}", exc_info=True)
            raise

    def transcribe_array(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> Dict[str, Any]:
        """
        Transcribe audio from numpy array

        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate in Hz

        Returns:
            dict with 'text' and metadata
        """
        logger.info(f"Groq transcription (array: {audio.shape}, sr: {sample_rate})")

        # Flatten if needed
        if audio.ndim > 1:
            logger.info(f"Audio shape before flatten: {audio.shape}")
            audio = audio.flatten()
            logger.info(f"Audio shape after flatten: {audio.shape}")

        # Save to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Write audio to temp file
            sf.write(tmp_path, audio, sample_rate)
            logger.info(f"Temp audio file: {tmp_path}")

            # Transcribe
            result = self.transcribe_file(tmp_path)

            return result

        finally:
            # Clean up temp file
            try:
                Path(tmp_path).unlink()
                logger.debug(f"Temp file deleted: {tmp_path}")
            except:
                pass

    def is_available(self) -> bool:
        """
        Check if API is available

        Returns:
            True if API key is set
        """
        return bool(self.api_key)
