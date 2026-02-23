import numpy as np
import sounddevice as sd
import wave
import tempfile
import os
import string
import asyncio
import threading
import edge_tts
import soundfile as sf
from faster_whisper import WhisperModel


class VoiceSystem:
    """Handles speech-to-text (Whisper), text-to-speech (edge-tts), and audio I/O"""

    def __init__(self, config, command_callback=None):
        """
        Args:
            config: Full settings dict
            command_callback: Callable(command: str, speak_response: bool) for processing commands
        """
        self.command_callback = command_callback

        # Audio settings
        audio_config = config["audio"]
        self.sample_rate = audio_config["sample_rate"]
        self.chunk_duration = audio_config["chunk_duration"]
        self.command_duration = audio_config["command_duration"]
        self.energy_threshold = audio_config["energy_threshold"]
        self.variance_threshold = audio_config["variance_threshold"]
        self.noise_words = audio_config["noise_words"]

        # Voice settings
        voice_config = config["voice"]
        self.tts_voice = voice_config["tts_voice"]
        self.wake_words = voice_config["wake_words"]

        # Load Whisper model
        whisper_config = config["whisper"]
        print(f"[Loading] Whisper model ({whisper_config['model_size']}) on GPU...")
        try:
            self.whisper_model = WhisperModel(
                whisper_config["model_size"],
                device=whisper_config["preferred_device"],
                compute_type=whisper_config["preferred_compute_type"]
            )
            print("   Whisper loaded on GPU!")
        except Exception as e:
            print(f"   GPU failed, falling back to CPU: {e}")
            self.whisper_model = WhisperModel(
                whisper_config["model_size"],
                device=whisper_config["fallback_device"],
                compute_type=whisper_config["fallback_compute_type"]
            )
            print("   Whisper loaded on CPU!")

        # TTS stop event — set by stop_speaking() to interrupt playback
        self._stop_event = threading.Event()

        print("   TTS ready!")

    def record_audio(self, duration):
        """Record audio for specified duration"""
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.int16
        )
        sd.wait()
        return audio

    def transcribe_audio(self, audio):
        """Convert audio to text using Whisper"""
        temp_file = tempfile.mktemp(suffix=".wav")
        with wave.open(temp_file, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(audio.tobytes())

        segments, info = self.whisper_model.transcribe(temp_file, language="en")
        text = " ".join([segment.text.strip() for segment in segments])

        os.remove(temp_file)
        return text.lower().translate(str.maketrans('', '', string.punctuation))

    def speak(self, text):
        """Convert text to speech using edge-tts.

        Returns True if speech completed normally, False if interrupted.
        """
        print(f"Talon: {text}")

        if not text or len(text.strip()) == 0:
            return True

        if text.startswith("Error:"):
            return True

        # Edge TTS mispronounces "Talon" as "tah-LONE".
        # "talun" produces the correct short-a "TAL-un" sound (like "talent" - t).
        tts_text = text.replace("Talon", "talun").replace("talon", "talun")

        self._stop_event.clear()
        return asyncio.run(self._async_speak(tts_text))

    def stop_speaking(self):
        """Interrupt any in-progress TTS playback immediately."""
        self._stop_event.set()
        try:
            sd.stop()
        except Exception:
            pass

    async def _async_speak(self, text):
        """Async helper for edge-tts.

        Returns True if completed normally, False if interrupted.
        """
        temp_audio = tempfile.mktemp(suffix=".mp3")

        try:
            communicate = edge_tts.Communicate(text, self.tts_voice)
            await communicate.save(temp_audio)
            await asyncio.sleep(0.2)

            if self._stop_event.is_set():
                return False

            if not os.path.exists(temp_audio):
                return True

            if os.path.getsize(temp_audio) == 0:
                return True

            data, samplerate = sf.read(temp_audio)
            sd.play(data, samplerate)

            # Poll instead of sd.wait() so we can respond to stop_speaking()
            while sd.get_stream() and sd.get_stream().active:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                await asyncio.sleep(0.05)

            return True

        except Exception as e:
            print(f"TTS Error: {e}")
            return True

        finally:
            if os.path.exists(temp_audio):
                try:
                    os.remove(temp_audio)
                except OSError:
                    pass

    def listen_for_wake_word(self):
        """Continuously listen for wake words"""
        print(f"Voice mode active - Listening for: {', '.join(self.wake_words)}")
        print("(Press Ctrl+C to exit)\n")

        while True:
            audio = self.record_audio(self.chunk_duration)

            audio_energy = np.abs(audio).mean()
            audio_variance = np.var(audio)

            if audio_energy < self.energy_threshold or audio_variance < self.variance_threshold:
                continue

            text = self.transcribe_audio(audio)

            if text.strip() in self.noise_words:
                continue

            if text and len(text) > 1:
                print(f"Heard: {text}")

            if any(wake_word in text for wake_word in self.wake_words):
                print("\nWake word detected!\n")
                self.handle_command()

    def handle_command(self):
        """Process voice command after wake word"""
        self.speak("Ready")

        audio = self.record_audio(self.command_duration)
        command = self.transcribe_audio(audio)

        if self.command_callback:
            self.command_callback(command, True)
