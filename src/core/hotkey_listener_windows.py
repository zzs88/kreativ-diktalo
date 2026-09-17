"""
Windows-optimalizált globális hotkey figyelő keyboard library-vel

A 'keyboard' library sokkal jobban működik Windowson mint a pynput,
de ADMIN JOGOKAT IGÉNYEL a globális hotkey funkcióhoz!
"""
import keyboard
from threading import Thread
from typing import Callable, Optional
from src.utils.logger import get_logger

logger = get_logger()


class WindowsHotkeyListener:
    """
    Windows-specifikus hotkey listener keyboard library-vel

    FONTOS: Admin jogokat igényel!
    """

    def __init__(self):
        """WindowsHotkeyListener inicializálás"""
        self.is_listening = False
        self.hotkeys: dict[str, dict] = {}
        self.registered_hooks = []  # Track registered hooks for cleanup
        self.press_states = {}  # Track if key is currently pressed (debouncing)
        logger.info("WindowsHotkeyListener inicializálva")

    def register_hotkey(
        self,
        hotkey: str,
        on_press: Optional[Callable] = None,
        on_release: Optional[Callable] = None
    ):
        """
        Hotkey regisztrálása

        Args:
            hotkey: Hotkey string (pl. "F8", "ctrl+shift+space")
            on_press: Callback lenyomáskor
            on_release: Callback felengedéskor
        """
        # keyboard library formátumra konvertálás
        # F8 -> f8, Ctrl+Space -> ctrl+space
        normalized_hotkey = hotkey.lower().replace('control', 'ctrl')

        self.hotkeys[hotkey] = {
            'normalized': normalized_hotkey,
            'on_press': on_press,
            'on_release': on_release
        }

        logger.info(f"Hotkey regisztrálva: {hotkey} -> {normalized_hotkey}")

    def start(self):
        """Hotkey listener indítása"""
        if self.is_listening:
            logger.warning("Listener már fut")
            return

        try:
            # Minden regisztrált hotkey-re feliratkozás
            for hotkey_str, hotkey_info in self.hotkeys.items():
                normalized = hotkey_info['normalized']
                self.press_states[hotkey_str] = False

                # Press callback wrapper (debounced)
                def make_press_callback(key, cb):
                    def wrapper():
                        if not self.press_states[key]:  # Only trigger once
                            self.press_states[key] = True
                            self._safe_callback(cb)
                    return wrapper

                # Release callback wrapper
                def make_release_callback(key, cb):
                    def wrapper():
                        if self.press_states[key]:  # Only trigger if was pressed
                            self.press_states[key] = False
                            self._safe_callback(cb)
                    return wrapper

                # Register press event
                if hotkey_info['on_press']:
                    press_cb = make_press_callback(hotkey_str, hotkey_info['on_press'])
                    keyboard.add_hotkey(normalized, press_cb, suppress=False)
                    logger.debug(f"Press hotkey registered: {normalized}")

                # Register release event using on_release_key (add_hotkey doesn't support release)
                if hotkey_info['on_release']:
                    release_cb = make_release_callback(hotkey_str, hotkey_info['on_release'])
                    keyboard.on_release_key(normalized, lambda e, cb=release_cb: cb(), suppress=False)
                    logger.debug(f"Release hotkey registered: {normalized}")

            self.is_listening = True
            logger.info("✅ Windows hotkey listener elindult (keyboard library)")

        except Exception as e:
            logger.error(f"❌ Hiba a listener indításakor: {e}")
            logger.error("FONTOS: Windows-on ADMIN JOGOK szükségesek!")
            raise

    def _safe_callback(self, callback: Callable):
        """
        Biztonságos callback hívás hibakezeléssel

        FONTOS: külön szálban futtatjuk, mert a `keyboard` library a hotkey
        callback-eket a natív alacsony szintű billentyűzet-hook szálán hívja
        meg szinkronban. Azon a szálon a sounddevice/PortAudio megbízhatatlanul
        elhasal ("Error querying device -1") mikrofon-stream nyitásakor, míg
        egy sima háttérszálból ugyanaz a hívás azonnal sikeres.
        """
        def _run():
            try:
                callback()
            except Exception as e:
                logger.error(f"Hiba a hotkey callback-ben: {e}", exc_info=True)

        Thread(target=_run, daemon=True).start()

    def stop(self):
        """Hotkey listener leállítása"""
        if not self.is_listening:
            logger.warning("Listener nincs elindítva")
            return

        try:
            # Minden hook eltávolítása
            keyboard.unhook_all()
            self.is_listening = False
            logger.info("Windows hotkey listener leállt")

        except Exception as e:
            logger.error(f"Hiba a listener leállításakor: {e}")

    def restart(self):
        """Listener újraindítása (stop + start) - session unlock / watchdog esetén"""
        logger.info("WindowsHotkeyListener újraindítása...")
        self.stop()
        self.start()
        logger.info("✅ WindowsHotkeyListener újraindult")

    def cleanup(self):
        """Erőforrások felszabadítása"""
        self.stop()
        logger.debug("WindowsHotkeyListener cleanup")


# Teszt funkció
def _test_windows_hotkey():
    """Teszt futtatás"""
    import time

    def on_f8_press():
        print("✅ F8 lenyomva!")

    def on_f8_release():
        print("🔴 F8 felengedve!")

    listener = WindowsHotkeyListener()
    listener.register_hotkey("f8", on_press=on_f8_press, on_release=on_f8_release)

    try:
        listener.start()
        print("🎧 Windows Hotkey listener fut... (F8)")
        print("⚠️  FONTOS: Admin jogok szükségesek!")
        print("   Válts más appba és próbáld ki az F8-at!")
        print("   Ctrl+C = kilépés")

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n🛑 Leállítás...")
        listener.stop()
    except Exception as e:
        print(f"\n❌ HIBA: {e}")
        print("\n🔧 MEGOLDÁS:")
        print("   1. Futtasd adminként a scriptet!")
        print("   2. Jobb klikk → 'Futtatás rendszergazdaként'")


if __name__ == "__main__":
    _test_windows_hotkey()
