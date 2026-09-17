"""
LLM-alapú szövegtisztító modul Ollama vagy Groq használatával
"""
import re
from typing import Optional
from src.utils.logger import get_logger

logger = get_logger()


class LLMCleaner:
    """LLM-mel történő szövegtisztítás (Ollama helyi vagy Groq felhő backend)"""

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        timeout: int = 30,
        temperature: float = 0.3,
        backend: str = "ollama",
        groq_api_key: str = "",
        vocabulary: str = ""
    ):
        """
        Args:
            host: Ollama szerver URL (csak backend="ollama" esetén)
            model: Használandó modell (Ollama modellnév vagy Groq chat modellnév)
            timeout: Timeout másodpercben
            temperature: LLM temperature (0.0-1.0)
            backend: "ollama" (helyi) vagy "groq" (felhő, nem kell hozzá helyi telepítés)
            groq_api_key: Groq API kulcs (csak backend="groq" esetén)
            vocabulary: Vesszővel elválasztott szakszavak/nevek lista, amit az LLM
                helyesen kell hogy felismerjen/megtartson javítás közben
        """
        self.host = host
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.backend = backend
        self.vocabulary = vocabulary
        self.ollama_available = False
        self.groq_client = None

        if self.backend == "groq":
            self._init_groq(groq_api_key)
        else:
            import ollama
            self.client = ollama.Client(host=self.host)
            self._check_ollama_connection()

    def _init_groq(self, api_key: str):
        """Groq backend inicializálása (nincs szükség kapcsolat-ellenőrzésre, csak kulcsra)"""
        if not api_key:
            logger.warning("Groq API kulcs hiányzik a szövegtisztításhoz, fallback regex tisztításra")
            return

        try:
            from groq import Groq
            self.groq_client = Groq(api_key=api_key)
            self.ollama_available = True  # generikus "LLM elérhető" jelző, lásd is_available()
            logger.info(f"Groq LLM szövegtisztítás inicializálva (model: {self.model})")
        except Exception as e:
            logger.warning(f"Groq LLM inicializálás sikertelen: {e}")

    def _check_ollama_connection(self):
        """Ollama szerver elérhetőség ellenőrzése"""
        try:
            # Modellek lekérése (teszt) - self.client újrafelhasználás
            models = self.client.list()

            self.ollama_available = True
            logger.info(f"Ollama szerver elérhető: {self.host}")

            # Modellek listája - különböző API verziók kezelése
            if hasattr(models, 'models'):
                model_list = models.models
            elif isinstance(models, dict):
                model_list = models.get('models', [])
            else:
                model_list = []

            model_names = []

            for m in model_list:
                # Extract model name from different formats
                if hasattr(m, 'model'):
                    name = m.model
                elif hasattr(m, 'name'):
                    name = m.name
                elif isinstance(m, dict):
                    name = m.get('model', m.get('name', str(m)))
                else:
                    name = str(m)
                model_names.append(name)

            logger.info(f"Elérhető modellek: {model_names}")

            # Ellenőrizzük hogy a kért modell elérhető-e
            if self.model not in model_names:
                logger.warning(f"A kért modell ({self.model}) nincs telepítve. Kérlek futtasd: ollama pull {self.model}")

        except Exception as e:
            self.ollama_available = False
            logger.warning(f"Ollama nem elérhető: {e}")
            logger.info("Fallback: Alapvető regex-alapú tisztítás lesz használva")

    def _build_cleaning_prompt(self, text: str) -> str:
        """
        Prompt generálása szövegtisztításhoz

        Args:
            text: Nyers szöveg

        Returns:
            Prompt string
        """
        vocab_hint = ""
        if self.vocabulary:
            vocab_hint = (
                f"\nGyakran előforduló szakszavak/nevek, amiket a beszédfelismerő "
                f"eltorzíthatott — ha egy szó ezekre hasonlít, javítsd a helyes alakra: "
                f"{self.vocabulary}\n"
            )

        prompt = f"""Ez egy magyar nyelvű diktált szöveg gépi beszédfelismerésből. Javítsd ki a nyilvánvaló félrehallásokat és elgépeléseket, távolítsd el a töltelékszavakat (hát, szóval, ööö, izé) és a szóismétléseket, tedd rendbe a helyesírást és az írásjeleket, mondatkezdéskor nagybetű. NE fogalmazd át a mondatokat, NE adj hozzá új tartalmat, csak a felismerési hibákat javítsd és tisztítsd a szöveget. Válaszolj KIZÁRÓLAG a javított szöveggel, semmi mással (ne írj bevezetőt vagy magyarázatot).
{vocab_hint}
Szöveg: {text}

Javított:"""

        return prompt

    def _build_command_prompt(self, text: str, command: str) -> str:
        """
        Prompt generálása command mode-hoz

        Args:
            text: Eredeti szöveg
            command: Felhasználói parancs

        Returns:
            Prompt string
        """
        prompt = f"""Te egy szövegszerkesztő asszisztens vagy. A feladatod, hogy módosítsd a megadott szöveget a felhasználó parancsa szerint.

PARANCS: {command}

EREDETI SZÖVEG:
{text}

MÓDOSÍTOTT SZÖVEG (csak a szöveget írd, semmi mást):"""

        return prompt

    def clean_text(self, text: str) -> str:
        """
        Szöveg tisztítása LLM-mel vagy fallback-kel

        Args:
            text: Nyers szöveg

        Returns:
            Tisztított szöveg
        """
        if not text or not text.strip():
            return text

        text = text.strip()
        logger.info(f"Szövegtisztítás indítása ({self.backend}): {len(text)} karakter")

        # LLM használat ha elérhető (Ollama vagy Groq)
        if self.ollama_available:
            try:
                if self.backend == "groq":
                    cleaned = self._clean_with_groq(text)
                else:
                    cleaned = self._clean_with_ollama(text)
                logger.info("LLM tisztítás sikeres")
                return cleaned
            except Exception as e:
                logger.error(f"LLM tisztítás hiba: {e}, fallback használata")

        # Fallback: Regex-alapú tisztítás
        cleaned = self._basic_clean(text)
        logger.info("Regex alapú tisztítás használva")
        return cleaned

    def _clean_with_groq(self, text: str) -> str:
        """
        Szöveg tisztítása Groq felhő LLM-mel

        Args:
            text: Nyers szöveg

        Returns:
            Tisztított szöveg
        """
        prompt = self._build_cleaning_prompt(text)

        response = self.groq_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
        )

        cleaned_text = response.choices[0].message.content.strip()

        # Biztonság: ha túl rövid vagy üres, ne használjuk
        if len(cleaned_text) < len(text) * 0.3:
            logger.warning("LLM válasz túl rövid, fallback használata")
            return self._basic_clean(text)

        return cleaned_text

    def _clean_with_ollama(self, text: str) -> str:
        """
        Szöveg tisztítása Ollama LLM-mel

        Args:
            text: Nyers szöveg

        Returns:
            Tisztított szöveg
        """
        prompt = self._build_cleaning_prompt(text)

        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': self.temperature,
                'top_p': 0.9,
                'top_k': 40,
            }
        )

        cleaned_text = response['response'].strip()

        # Biztonság: ha túl rövid vagy üres, ne használjuk
        if len(cleaned_text) < len(text) * 0.3:
            logger.warning("LLM válasz túl rövid, fallback használata")
            return self._basic_clean(text)

        return cleaned_text

    def _basic_clean(self, text: str) -> str:
        """
        Alapvető regex-alapú szövegtisztítás (fallback)

        Args:
            text: Nyers szöveg

        Returns:
            Tisztított szöveg
        """
        cleaned = text

        # 1. Töltelékszavak eltávolítása
        filler_words = [
            r'\bhát\b', r'\bszóval\b', r'\búgy\b', r'\bna\b', r'\bnos\b',
            r'\bööö+\b', r'\bum+\b', r'\buh+\b', r'\behm+\b',
            r'\bte tudod\b', r'\bugye\b', r'\bhogy is mondjam\b'
        ]

        for pattern in filler_words:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

        # 2. Többszörös szóközök eltávolítása
        cleaned = re.sub(r'\s+', ' ', cleaned)

        # 3. Szóköz írásjelek előtt/után
        cleaned = re.sub(r'\s+([.,!?;:])', r'\1', cleaned)  # Szóköz eltávolítása írásjelek előtt
        cleaned = re.sub(r'([.,!?;:])\s*', r'\1 ', cleaned)  # Szóköz írásjelek után

        # 4. Mondatkezdés nagybetűvel
        sentences = re.split(r'([.!?]\s+)', cleaned)
        cleaned = ''
        for i, part in enumerate(sentences):
            if i % 2 == 0 and part:  # Mondat (nem delimiter)
                part = part[0].upper() + part[1:] if part else part
            cleaned += part

        # 5. Trim
        cleaned = cleaned.strip()

        # 6. Pont a végére ha nincs
        if cleaned and cleaned[-1] not in '.!?':
            cleaned += '.'

        return cleaned

    def process_command(self, text: str, command: str) -> str:
        """
        Szöveg módosítása parancs alapján (Command Mode)

        Args:
            text: Eredeti szöveg
            command: Felhasználói parancs

        Returns:
            Módosított szöveg
        """
        if not text or not command:
            return text

        logger.info(f"Command feldolgozás: '{command}'")

        if not self.ollama_available:
            logger.warning("LLM nem elérhető, command mode nem működik")
            return text

        try:
            prompt = self._build_command_prompt(text, command)

            if self.backend == "groq":
                response = self.groq_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                modified_text = response.choices[0].message.content.strip()
            else:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options={
                        'temperature': self.temperature,
                    }
                )
                modified_text = response['response'].strip()

            logger.info("Command feldolgozás sikeres")
            return modified_text

        except Exception as e:
            logger.error(f"Hiba a command feldolgozáskor: {e}")
            return text

    def is_available(self) -> bool:
        """
        Ollama elérhetőség lekérése

        Returns:
            True ha Ollama elérhető
        """
        return self.ollama_available

    def get_status(self) -> dict:
        """
        Státusz információk lekérése

        Returns:
            Státusz dictionary
        """
        return {
            'backend': self.backend,
            'ollama_available': self.ollama_available,
            'host': self.host,
            'model': self.model,
            'temperature': self.temperature
        }


def build_llm_cleaner(config) -> "LLMCleaner":
    """
    LLMCleaner létrehozása a config alapján, backend-választással (ollama vagy groq)

    Args:
        config: ConfigManager instance

    Returns:
        Konfigurált LLMCleaner
    """
    backend = config.get('text_processing.backend', 'ollama')

    if backend == 'groq':
        return LLMCleaner(
            backend='groq',
            groq_api_key=config.get('stt.groq.api_key', ''),
            model=config.get('text_processing.groq.model', 'openai/gpt-oss-120b'),
            temperature=config.get('text_processing.groq.temperature', 0.2),
            vocabulary=config.get('text_processing.vocabulary', '')
        )

    return LLMCleaner(
        backend='ollama',
        host=config.get('ollama.host', 'http://localhost:11434'),
        model=config.get('ollama.model', 'llama3.1:8b'),
        timeout=config.get('ollama.timeout', 30),
        temperature=config.get('ollama.temperature', 0.3),
        vocabulary=config.get('text_processing.vocabulary', '')
    )


# Teszt funkció
def _test_llm_cleaner():
    """Teszt futtatás"""

    test_texts = [
        "hát szóval úgy gondolom hogy ez egy teszt szöveg ööö igen",
        "holnap nem várj inkább pénteken találkozunk",
        "te tudod ez a dolog ami úgy van hogy nos hát igen",
    ]

    cleaner = LLMCleaner()

    print("LLM Cleaner teszt\n")
    print(f"Ollama elérhető: {cleaner.is_available()}\n")

    for i, text in enumerate(test_texts, 1):
        print(f"Teszt {i}:")
        print(f"  Eredeti: {text}")
        cleaned = cleaner.clean_text(text)
        print(f"  Tisztított: {cleaned}")
        print()

    # Command mode teszt
    if cleaner.is_available():
        print("Command Mode teszt:")
        original = "Ez egy hosszú és bonyolult mondat, amit szeretnék egyszerűsíteni."
        print(f"  Eredeti: {original}")
        result = cleaner.process_command(original, "rövidítsd le")
        print(f"  Rövidítve: {result}")


if __name__ == "__main__":
    _test_llm_cleaner()
