"""
Voice Cloning Studio
====================
Fully offline, no internet required.
Records voice samples → builds a voice profile → synthesizes speech.

Dependencies (all installable via pip, no internet after install):
  - pyttsx3       : offline TTS engine (uses Windows SAPI5)
  - sounddevice   : microphone recording
  - numpy         : audio processing
  - scipy         : WAV file I/O + DSP
  - tkinter       : GUI (bundled with Python)
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys
import json
import wave
import struct
import math
import time
import queue
import shutil
import tempfile
from pathlib import Path
from datetime import datetime

# ── optional heavy deps loaded lazily ──────────────────────────────────────
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import sounddevice as sd
    HAS_SD = True
except ImportError:
    HAS_SD = False

try:
    import pyttsx3
    HAS_TTS = True
except ImportError:
    HAS_TTS = False

try:
    from scipy.io import wavfile
    from scipy import signal as sig
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from pydub import AudioSegment
    HAS_PYDUB = True
except ImportError:
    HAS_PYDUB = False

# ── constants ───────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050
CHANNELS      = 1
DTYPE         = 'int16'
APP_DIR       = Path.home() / "VoiceCloningStudio"
PROFILES_DIR  = APP_DIR / "profiles"
RECORDINGS_DIR = APP_DIR / "recordings"
EXPORTS_DIR   = APP_DIR / "exports"

SAMPLE_SENTENCES = [
    "The sun rises early in the morning, casting a warm golden light across the quiet valley.",
    "She walked slowly through the autumn leaves, thinking about the journey that lay ahead.",
    "Technology has changed the way we communicate, work, and experience the world around us.",
    "A warm cup of tea and a good book make for a perfect and peaceful evening at home.",
    "The mountain trail winds through dense pine forests before opening onto a breathtaking view.",
    "Every great achievement begins with the decision to try, no matter how difficult the task.",
    "The old clock on the mantle ticked steadily as the family gathered around the fireplace.",
    "Learning something new every day keeps the mind sharp and the spirit full of curiosity.",
]

PALETTE = {
    "bg":        "#1a1a2e",
    "surface":   "#16213e",
    "card":      "#0f3460",
    "accent":    "#e94560",
    "accent2":   "#533483",
    "green":     "#00b894",
    "amber":     "#fdcb6e",
    "text":      "#eaeaea",
    "muted":     "#8892a4",
    "border":    "#2d3561",
    "white":     "#ffffff",
}

# ── helpers ─────────────────────────────────────────────────────────────────

def ensure_dirs():
    for d in [APP_DIR, PROFILES_DIR, RECORDINGS_DIR, EXPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def samples_to_numpy(frames, dtype=np.int16 if HAS_NUMPY else None):
    if not HAS_NUMPY:
        return None
    return np.frombuffer(b"".join(frames), dtype=np.int16)

def write_wav(path: Path, data, sr=SAMPLE_RATE):
    """Write int16 numpy array or raw bytes to WAV."""
    if HAS_SCIPY and HAS_NUMPY and isinstance(data, np.ndarray):
        wavfile.write(str(path), sr, data.astype(np.int16))
    else:
        with wave.open(str(path), 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            if isinstance(data, (bytes, bytearray)):
                wf.writeframes(data)
            else:
                wf.writeframes(b"".join(data))

def read_wav(path: Path):
    if HAS_SCIPY:
        sr, data = wavfile.read(str(path))
        return sr, data
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        if HAS_NUMPY:
            data = np.frombuffer(raw, dtype=np.int16)
        else:
            data = raw
    return sr, data

def compute_rms(data):
    if HAS_NUMPY and isinstance(data, np.ndarray):
        return float(np.sqrt(np.mean(data.astype(np.float32)**2)))
    if isinstance(data, (bytes, bytearray)):
        n = len(data) // 2
        vals = struct.unpack(f'{n}h', data[:n*2])
        return math.sqrt(sum(v*v for v in vals) / max(n, 1))
    return 0.0

def estimate_pitch(data, sr):
    """Very lightweight autocorrelation pitch estimator (no external deps)."""
    if not (HAS_NUMPY and isinstance(data, np.ndarray)):
        return 150.0
    chunk = data[:sr//4].astype(np.float32)
    if len(chunk) < 512:
        return 150.0
    # normalise
    mx = np.max(np.abs(chunk))
    if mx < 1:
        return 150.0
    chunk /= mx
    # autocorrelation via numpy
    corr = np.correlate(chunk, chunk, mode='full')
    corr = corr[len(corr)//2:]
    # look in 80–400 Hz range
    lo = int(sr / 400)
    hi = int(sr / 80)
    hi = min(hi, len(corr)-1)
    if lo >= hi:
        return 150.0
    peak = np.argmax(corr[lo:hi]) + lo
    return float(sr / peak) if peak > 0 else 150.0

# ── voice profile ────────────────────────────────────────────────────────────

class VoiceProfile:
    def __init__(self, name="My Voice"):
        self.name       = name
        self.created    = datetime.now().isoformat()
        self.pitch_hz   = 150.0    # average fundamental freq
        self.rate_wpm   = 150      # words per minute
        self.energy     = 0.0      # average RMS
        self.pitch_adj  = 0        # user pitch tweak (-10..+10)
        self.speed_adj  = 1.0      # user speed multiplier
        self.volume_adj = 1.0      # user volume
        self.sample_files = []     # list of WAV paths used
        self.voice_id   = None     # pyttsx3 voice id

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        p = cls()
        p.__dict__.update(d)
        return p

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def analyze_samples(self, wav_paths):
        """Analyse recorded WAV files to extract voice characteristics."""
        pitches, energies, durations = [], [], []
        for p in wav_paths:
            try:
                sr, data = read_wav(Path(p))
                pitches.append(estimate_pitch(data, sr))
                energies.append(compute_rms(data))
                if HAS_NUMPY and isinstance(data, np.ndarray):
                    durations.append(len(data) / sr)
            except Exception:
                pass
        if pitches:
            self.pitch_hz = float(sum(pitches) / len(pitches))
        if energies:
            self.energy   = float(sum(energies) / len(energies))
        # rough WPM estimate: average ~3.5 chars/word
        if durations:
            avg_dur = sum(durations) / len(durations)
            chars_per_sec = 8  # rough estimate
            self.rate_wpm = max(80, min(250, int(chars_per_sec * avg_dur * 3)))
        self.sample_files = [str(p) for p in wav_paths]

# ── recorder ────────────────────────────────────────────────────────────────

class Recorder:
    def __init__(self):
        self.is_recording = False
        self._frames = []
        self._stream  = None
        self._q       = queue.Queue()

    def start(self, callback_level=None):
        if not HAS_SD:
            raise RuntimeError("sounddevice not installed")
        self._frames = []
        self.is_recording = True
        self._cb_level = callback_level

        def _cb(indata, frames, time_info, status):
            if self.is_recording:
                chunk = indata.copy()
                self._frames.append(chunk.tobytes())
                if callback_level:
                    rms = compute_rms(chunk.tobytes())
                    callback_level(rms)

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=_cb,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self.is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self._frames

    def save(self, path: Path):
        frames = self._frames
        if not frames:
            return False
        if HAS_NUMPY:
            data = np.frombuffer(b"".join(frames), dtype=np.int16)
            write_wav(path, data)
        else:
            write_wav(path, b"".join(frames))
        return True

# ── TTS engine wrapper ───────────────────────────────────────────────────────

class TTSEngine:
    def __init__(self):
        self._engine = None
        self._voices = []
        self._lock   = threading.Lock()

    def _get_engine(self):
        if self._engine is None:
            if not HAS_TTS:
                raise RuntimeError("pyttsx3 not installed")
            self._engine = pyttsx3.init()
        return self._engine

    def get_voices(self):
        if self._voices:
            return self._voices
        eng = self._get_engine()
        self._voices = eng.getProperty('voices')
        return self._voices

    def apply_profile(self, profile: VoiceProfile):
        eng = self._get_engine()
        # Rate: pyttsx3 default ~200 wpm
        base_rate = max(80, min(300, profile.rate_wpm))
        rate = int(base_rate * profile.speed_adj)
        eng.setProperty('rate', rate)
        # Volume
        eng.setProperty('volume', min(1.0, profile.volume_adj))
        # Voice selection
        voices = self.get_voices()
        if profile.voice_id:
            eng.setProperty('voice', profile.voice_id)
        elif voices:
            eng.setProperty('voice', voices[0].id)

    def speak(self, text: str, profile: VoiceProfile, done_cb=None):
        def _run():
            with self._lock:
                try:
                    eng = self._get_engine()
                    self.apply_profile(profile)
                    eng.say(text)
                    eng.runAndWait()
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    if done_cb:
                        done_cb()
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def save_to_file(self, text: str, out_path: Path, profile: VoiceProfile, done_cb=None):
        def _run():
            with self._lock:
                try:
                    eng = self._get_engine()
                    self.apply_profile(profile)
                    out_path_obj = Path(out_path)
                    want_mp3 = out_path_obj.suffix.lower() == '.mp3'
                    # pyttsx3 always saves WAV; convert afterwards if MP3 wanted
                    wav_path = out_path_obj.with_suffix('.wav') if want_mp3 else out_path_obj
                    eng.save_to_file(text, str(wav_path))
                    eng.runAndWait()
                    if want_mp3:
                        if HAS_PYDUB:
                            AudioSegment.from_wav(str(wav_path)).export(
                                str(out_path_obj), format='mp3')
                            wav_path.unlink(missing_ok=True)
                        else:
                            print("pydub not installed; saved as WAV instead of MP3")
                except Exception as e:
                    print(f"TTS save error: {e}")
                finally:
                    if done_cb:
                        done_cb()
        t = threading.Thread(target=_run, daemon=True)
        t.start()

    def stop(self):
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════════════════
# GUI
# ═══════════════════════════════════════════════════════════════════════════

class WaveformCanvas(tk.Canvas):
    """Animated waveform display."""
    def __init__(self, master, **kw):
        kw.setdefault('bg', PALETTE['card'])
        kw.setdefault('highlightthickness', 0)
        super().__init__(master, **kw)
        self._level   = 0.0
        self._bars    = []
        self._history = [0.0] * 60
        self._running = False
        self.bind('<Configure>', self._on_resize)

    def _on_resize(self, _e=None):
        self._draw()

    def set_level(self, rms: float):
        self._level = min(1.0, rms / 8000.0)
        self._history.append(self._level)
        self._history = self._history[-60:]
        self._draw()

    def set_static(self, wav_path: Path):
        """Draw a static waveform from a saved WAV file."""
        if not HAS_NUMPY or not wav_path.exists():
            return
        try:
            sr, data = read_wav(wav_path)
            if not isinstance(data, np.ndarray):
                return
            self._history = []
            w = self.winfo_width() or 300
            step = max(1, len(data) // w)
            for i in range(0, len(data), step * 4):
                chunk = data[i:i+step]
                val = float(np.max(np.abs(chunk))) / 32768.0
                self._history.append(val)
            self._draw_static()
        except Exception:
            pass

    def _draw_static(self):
        self.delete('all')
        w = self.winfo_width() or 300
        h = self.winfo_height() or 40
        mid = h // 2
        if not self._history:
            return
        step = w / max(len(self._history), 1)
        for i, v in enumerate(self._history):
            x = int(i * step)
            bar_h = max(1, int(v * (h * 0.9)))
            self.create_line(x, mid - bar_h, x, mid + bar_h,
                             fill=PALETTE['green'], width=1)

    def _draw(self):
        self.delete('all')
        w = self.winfo_width() or 300
        h = self.winfo_height() or 40
        mid = h // 2
        hist = self._history[-w:]
        if not hist:
            # idle line
            self.create_line(0, mid, w, mid, fill=PALETTE['border'], width=1)
            return
        step = w / max(len(hist), 1)
        for i, v in enumerate(hist):
            x = int(i * step)
            bar_h = max(1, int(v * (h * 0.85)))
            color = PALETTE['accent'] if self._running else PALETTE['green']
            self.create_line(x, mid - bar_h, x, mid + bar_h,
                             fill=color, width=1)

    def start_anim(self):
        self._running = True

    def stop_anim(self):
        self._running = False


class SampleRow(tk.Frame):
    """One recording row in the Record tab."""
    def __init__(self, master, idx: int, text: str, studio, **kw):
        super().__init__(master, bg=PALETTE['surface'], **kw)
        self.idx     = idx
        self.text    = text
        self.studio  = studio
        self.state   = 'idle'   # idle | recording | done
        self.wav_path = RECORDINGS_DIR / f"sample_{idx:02d}.wav"
        self._build()

    def _build(self):
        P = PALETTE
        # index badge
        self.badge = tk.Label(self, text=str(self.idx+1), width=3,
                              bg=P['card'], fg=P['muted'],
                              font=('Segoe UI', 9, 'bold'))
        self.badge.pack(side='left', padx=(8,4), pady=6)

        # sentence
        self.lbl = tk.Label(self, text=f'"{self.text}"',
                            bg=P['surface'], fg=P['muted'],
                            font=('Segoe UI', 9, 'italic'),
                            wraplength=340, justify='left', anchor='w')
        self.lbl.pack(side='left', padx=4, fill='x', expand=True)

        # waveform
        self.wave = WaveformCanvas(self, width=120, height=34)
        self.wave.pack(side='left', padx=4)

        # buttons
        self.rec_btn = tk.Button(self, text='⏺ Record',
                                 bg=P['accent2'], fg=P['white'],
                                 font=('Segoe UI', 8, 'bold'),
                                 relief='flat', bd=0, padx=8, pady=4,
                                 cursor='hand2',
                                 command=self._toggle_record)
        self.rec_btn.pack(side='left', padx=3)

        self.play_btn = tk.Button(self, text='▶',
                                  bg=P['card'], fg=P['green'],
                                  font=('Segoe UI', 9), relief='flat',
                                  bd=0, padx=6, pady=4, cursor='hand2',
                                  command=self._play,
                                  state='disabled')
        self.play_btn.pack(side='left', padx=2)

        self.del_btn = tk.Button(self, text='✕',
                                 bg=P['card'], fg=P['accent'],
                                 font=('Segoe UI', 9), relief='flat',
                                 bd=0, padx=6, pady=4, cursor='hand2',
                                 command=self._delete,
                                 state='disabled')
        self.del_btn.pack(side='left', padx=(2,8))

        # separator
        sep = tk.Frame(self, height=1, bg=P['border'])
        sep.pack(side='bottom', fill='x')

        if self.wav_path.exists():
            self._set_done()

    def _toggle_record(self):
        if self.state == 'recording':
            self.studio.stop_recording(self.idx)
        else:
            self.studio.start_recording(self.idx)

    def set_recording(self):
        self.state = 'recording'
        self.rec_btn.config(text='⏹ Stop', bg=PALETTE['accent'])
        self.wave.start_anim()
        self.badge.config(bg=PALETTE['accent'], fg=PALETTE['white'])

    def _set_done(self):
        self.state = 'done'
        self.rec_btn.config(text='⟳ Re-record', bg=PALETTE['accent2'])
        self.play_btn.config(state='normal')
        self.del_btn.config(state='normal')
        self.badge.config(bg=PALETTE['green'], fg=PALETTE['surface'],
                          text='✓')
        self.lbl.config(fg=PALETTE['text'])
        self.wave.stop_anim()
        self.wave.set_static(self.wav_path)

    def set_done(self):
        self._set_done()

    def _play(self):
        if not self.wav_path.exists():
            return
        import subprocess
        try:
            # Windows: use built-in playsound via winsound
            import winsound
            winsound.PlaySound(str(self.wav_path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except ImportError:
            # fallback: use sounddevice
            if HAS_SD and HAS_SCIPY:
                sr, data = read_wav(self.wav_path)
                sd.play(data, sr)

    def _delete(self):
        if self.wav_path.exists():
            self.wav_path.unlink()
        self.state = 'idle'
        self.rec_btn.config(text='⏺ Record', bg=PALETTE['accent2'])
        self.play_btn.config(state='disabled')
        self.del_btn.config(state='disabled')
        self.badge.config(bg=PALETTE['card'], fg=PALETTE['muted'],
                          text=str(self.idx+1))
        self.lbl.config(fg=PALETTE['muted'])
        self.wave._history = [0.0] * 60
        self.wave._draw()
        self.studio.update_status()

    def update_level(self, rms):
        self.wave.set_level(rms)


class VoiceStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_dirs()
        self.title("Voice Cloning Studio")
        self.configure(bg=PALETTE['bg'])
        self.geometry("860x680")
        self.minsize(760, 560)
        self.resizable(True, True)

        self.recorder   = Recorder()
        self.tts        = TTSEngine()
        self.profile    = VoiceProfile()
        self.profile_path = PROFILES_DIR / "default.json"
        self._active_rec_row: SampleRow = None
        self._speaking  = False

        self._check_deps()
        self._load_profile()
        self._build_ui()
        self.update_status()

    def _check_deps(self):
        missing = []
        if not HAS_SD:
            missing.append("sounddevice  (pip install sounddevice)")
        if not HAS_TTS:
            missing.append("pyttsx3      (pip install pyttsx3)")
        if not HAS_NUMPY:
            missing.append("numpy        (pip install numpy)")
        if missing:
            msg = ("Some dependencies are not installed.\n"
                   "Recording and synthesis will be limited.\n\n"
                   "Missing:\n" + "\n".join(missing) +
                   "\n\nRun install_deps.bat to fix this.")
            messagebox.showwarning("Missing Dependencies", msg)

    def _load_profile(self):
        if self.profile_path.exists():
            try:
                self.profile = VoiceProfile.load(self.profile_path)
            except Exception:
                pass

    # ── UI BUILD ──────────────────────────────────────────────────────────

    def _build_ui(self):
        P = PALETTE

        # ── title bar ─────────────────────────────────────────────────────
        title_bar = tk.Frame(self, bg=P['card'], height=52)
        title_bar.pack(fill='x', side='top')
        title_bar.pack_propagate(False)

        tk.Label(title_bar, text="🎙 Voice Cloning Studio",
                 bg=P['card'], fg=P['white'],
                 font=('Segoe UI', 14, 'bold')).pack(side='left', padx=20)
        tk.Label(title_bar, text="Offline · No Internet Required",
                 bg=P['card'], fg=P['muted'],
                 font=('Segoe UI', 9)).pack(side='left', padx=4)

        # profile name
        tk.Label(title_bar, text="Profile:", bg=P['card'],
                 fg=P['muted'], font=('Segoe UI', 9)).pack(side='right', padx=(0,4))
        self.profile_name_var = tk.StringVar(value=self.profile.name)
        self.profile_entry = tk.Entry(title_bar, textvariable=self.profile_name_var,
                                      bg=P['surface'], fg=P['white'],
                                      insertbackground=P['white'],
                                      font=('Segoe UI', 9), width=14,
                                      relief='flat', bd=4)
        self.profile_entry.pack(side='right', padx=(0,16), pady=10)

        # ── notebook / tabs ───────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use('clam')
        style.configure('TNotebook',
                         background=P['bg'], borderwidth=0)
        style.configure('TNotebook.Tab',
                         background=P['card'], foreground=P['muted'],
                         padding=[18, 8],
                         font=('Segoe UI', 10, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', P['accent2'])],
                  foreground=[('selected', P['white'])])

        self.nb = ttk.Notebook(self)
        self.nb.pack(fill='both', expand=True, padx=12, pady=(8,0))

        self.tab_record   = tk.Frame(self.nb, bg=P['bg'])
        self.tab_train    = tk.Frame(self.nb, bg=P['bg'])
        self.tab_synth    = tk.Frame(self.nb, bg=P['bg'])
        self.tab_settings = tk.Frame(self.nb, bg=P['bg'])

        self.nb.add(self.tab_record,   text='  1 · Record  ')
        self.nb.add(self.tab_train,    text='  2 · Train   ')
        self.nb.add(self.tab_synth,    text='  3 · Synthesize ')
        self.nb.add(self.tab_settings, text='  ⚙ Settings  ')

        self._build_record_tab()
        self._build_train_tab()
        self._build_synth_tab()
        self._build_settings_tab()

        # ── status bar ────────────────────────────────────────────────────
        self.status_bar = tk.Frame(self, bg=P['card'], height=28)
        self.status_bar.pack(fill='x', side='bottom')
        self.status_bar.pack_propagate(False)
        self.status_lbl = tk.Label(self.status_bar,
                                    text="Ready",
                                    bg=P['card'], fg=P['muted'],
                                    font=('Segoe UI', 8))
        self.status_lbl.pack(side='left', padx=12)
        self.dep_lbl = tk.Label(self.status_bar,
                                 text=self._dep_status(),
                                 bg=P['card'], fg=P['amber'],
                                 font=('Segoe UI', 8))
        self.dep_lbl.pack(side='right', padx=12)

    # ── RECORD TAB ────────────────────────────────────────────────────────

    def _build_record_tab(self):
        P = PALETTE
        tab = self.tab_record

        # info bar
        info = tk.Frame(tab, bg=P['surface'])
        info.pack(fill='x', padx=10, pady=(10,4))
        tk.Label(info,
                 text="📋  Read each sentence naturally, at your normal speaking pace."
                      "  Aim for 5–10 seconds per sample.  Quiet room recommended.",
                 bg=P['surface'], fg=P['amber'],
                 font=('Segoe UI', 9), pady=8, padx=12,
                 wraplength=700, justify='left').pack(anchor='w')

        # scrollable sample list
        frame = tk.Frame(tab, bg=P['bg'])
        frame.pack(fill='both', expand=True, padx=10, pady=4)

        canvas = tk.Canvas(frame, bg=P['bg'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient='vertical',
                                  command=canvas.yview)
        self.samples_frame = tk.Frame(canvas, bg=P['surface'])
        self.samples_frame.bind('<Configure>',
            lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0,0), window=self.samples_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        self.sample_rows: list[SampleRow] = []
        for i, text in enumerate(SAMPLE_SENTENCES):
            row = SampleRow(self.samples_frame, i, text, self)
            row.pack(fill='x', padx=2, pady=2)
            self.sample_rows.append(row)

        # Add custom sentence button
        add_btn = tk.Button(tab, text='+ Add Custom Sentence',
                            bg=P['card'], fg=P['text'],
                            font=('Segoe UI', 9), relief='flat',
                            bd=0, padx=12, pady=6, cursor='hand2',
                            command=self._add_custom)
        add_btn.pack(side='left', padx=12, pady=8)

        # bottom status / next
        self.rec_status_lbl = tk.Label(tab, text='',
                                        bg=P['bg'], fg=P['muted'],
                                        font=('Segoe UI', 9))
        self.rec_status_lbl.pack(side='left', padx=12, pady=8)

        next_btn = tk.Button(tab, text='Next: Train  →',
                             bg=P['green'], fg=P['surface'],
                             font=('Segoe UI', 10, 'bold'),
                             relief='flat', bd=0, padx=16, pady=8,
                             cursor='hand2',
                             command=lambda: self.nb.select(1))
        next_btn.pack(side='right', padx=12, pady=8)

    def _add_custom(self):
        win = tk.Toplevel(self)
        win.title("Add Custom Sentence")
        win.configure(bg=PALETTE['surface'])
        win.geometry("480x160")
        win.resizable(False, False)
        tk.Label(win, text="Enter your sentence (10–25 words is ideal):",
                 bg=PALETTE['surface'], fg=PALETTE['text'],
                 font=('Segoe UI', 10)).pack(padx=20, pady=(16,6), anchor='w')
        var = tk.StringVar()
        entry = tk.Entry(win, textvariable=var,
                         bg=PALETTE['card'], fg=PALETTE['white'],
                         insertbackground=PALETTE['white'],
                         font=('Segoe UI', 10), relief='flat', bd=6)
        entry.pack(fill='x', padx=20)
        entry.focus()
        def ok():
            t = var.get().strip()
            if len(t) < 8:
                return
            idx = len(self.sample_rows)
            SAMPLE_SENTENCES.append(t)
            row = SampleRow(self.samples_frame, idx, t, self)
            row.pack(fill='x', padx=2, pady=2)
            self.sample_rows.append(row)
            win.destroy()
        tk.Button(win, text='Add', bg=PALETTE['accent2'], fg=PALETTE['white'],
                  font=('Segoe UI', 10, 'bold'), relief='flat', bd=0,
                  padx=20, pady=6, cursor='hand2', command=ok).pack(pady=12)
        win.bind('<Return>', lambda _: ok())

    # ── TRAIN TAB ─────────────────────────────────────────────────────────

    def _build_train_tab(self):
        P = PALETTE
        tab = self.tab_train

        # summary card
        sum_frame = tk.Frame(tab, bg=P['card'])
        sum_frame.pack(fill='x', padx=12, pady=(12,6))

        tk.Label(sum_frame, text="Recordings Summary",
                 bg=P['card'], fg=P['white'],
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=14, pady=(10,4))

        self.summary_lbl = tk.Label(sum_frame, text='',
                                     bg=P['card'], fg=P['muted'],
                                     font=('Segoe UI', 9),
                                     justify='left', anchor='w')
        self.summary_lbl.pack(anchor='w', padx=14, pady=(0,8))

        # quality bar
        q_frame = tk.Frame(tab, bg=P['bg'])
        q_frame.pack(fill='x', padx=12, pady=2)
        tk.Label(q_frame, text="Voice profile quality:",
                 bg=P['bg'], fg=P['muted'],
                 font=('Segoe UI', 9)).pack(side='left')
        self.quality_lbl = tk.Label(q_frame, text='0%',
                                     bg=P['bg'], fg=P['green'],
                                     font=('Segoe UI', 9, 'bold'))
        self.quality_lbl.pack(side='right')

        self.quality_bar = ttk.Progressbar(tab, orient='horizontal',
                                            length=200, mode='determinate',
                                            maximum=100)
        style = ttk.Style()
        style.configure("green.Horizontal.TProgressbar",
                         troughcolor=P['card'],
                         background=P['green'])
        self.quality_bar.configure(style="green.Horizontal.TProgressbar")
        self.quality_bar.pack(fill='x', padx=12, pady=4)

        # voice selection
        v_frame = tk.Frame(tab, bg=P['card'])
        v_frame.pack(fill='x', padx=12, pady=6)
        tk.Label(v_frame, text="System Voice (base)",
                 bg=P['card'], fg=P['white'],
                 font=('Segoe UI', 10, 'bold')).pack(anchor='w', padx=14, pady=(10,4))
        tk.Label(v_frame,
                 text="Your voice profile tuning is applied ON TOP of the selected system voice.\n"
                      "Choose the voice closest to yours in gender / accent.",
                 bg=P['card'], fg=P['muted'],
                 font=('Segoe UI', 8), justify='left').pack(anchor='w', padx=14)

        self.voice_var = tk.StringVar()
        self.voice_combo = ttk.Combobox(v_frame, textvariable=self.voice_var,
                                         state='readonly', width=54)
        self.voice_combo.pack(padx=14, pady=8, anchor='w')
        self._populate_voices()

        # build button
        self.build_btn = tk.Button(tab, text='🧠  Build Voice Profile',
                                    bg=P['accent'], fg=P['white'],
                                    font=('Segoe UI', 11, 'bold'),
                                    relief='flat', bd=0, padx=24, pady=10,
                                    cursor='hand2',
                                    command=self._build_profile)
        self.build_btn.pack(pady=12)

        self.train_status = tk.Label(tab, text='',
                                      bg=P['bg'], fg=P['green'],
                                      font=('Segoe UI', 9))
        self.train_status.pack()

        self.train_progress = ttk.Progressbar(tab, orient='horizontal',
                                               length=300, mode='indeterminate')
        self.train_progress.pack(pady=4)

        # next
        tk.Button(tab, text='Next: Synthesize  →',
                  bg=P['green'], fg=P['surface'],
                  font=('Segoe UI', 10, 'bold'),
                  relief='flat', bd=0, padx=16, pady=8,
                  cursor='hand2',
                  command=lambda: self.nb.select(2)).pack(side='right', padx=12, pady=8)

    def _populate_voices(self):
        if not HAS_TTS:
            self.voice_combo['values'] = ['(pyttsx3 not installed)']
            return
        try:
            voices = self.tts.get_voices()
            names = [f"{v.name}  [{v.languages[0] if v.languages else '?'}]"
                     for v in voices]
            self.voice_combo['values'] = names if names else ['(no voices found)']
            # preselect saved
            if self.profile.voice_id and voices:
                for i, v in enumerate(voices):
                    if v.id == self.profile.voice_id:
                        self.voice_combo.current(i)
                        return
            if names:
                self.voice_combo.current(0)
        except Exception as e:
            self.voice_combo['values'] = [f'Error: {e}']

    def _build_profile(self):
        wav_paths = [r.wav_path for r in self.sample_rows if r.wav_path.exists()]
        if len(wav_paths) < 1:
            messagebox.showwarning("No recordings",
                                   "Please record at least 1 sample first.")
            return

        self.build_btn.config(state='disabled')
        self.train_status.config(text="Analysing recordings…", fg=PALETTE['amber'])
        self.train_progress.start(12)

        def _work():
            self.profile.name = self.profile_name_var.get().strip() or "My Voice"
            self.profile.analyze_samples(wav_paths)
            # save selected voice id
            if HAS_TTS:
                voices = self.tts.get_voices()
                sel = self.voice_combo.current()
                if voices and 0 <= sel < len(voices):
                    self.profile.voice_id = voices[sel].id
            self.profile.save(self.profile_path)
            self.after(0, _done)

        def _done():
            self.train_progress.stop()
            self.build_btn.config(state='normal')
            n = len(wav_paths)
            q = min(100, int(n / len(SAMPLE_SENTENCES) * 100))
            self.quality_bar['value'] = q
            self.quality_lbl.config(text=f'{q}%')
            self.train_status.config(
                text=f"✓ Voice profile built from {n} sample(s)  "
                     f"— estimated pitch: {self.profile.pitch_hz:.0f} Hz  "
                     f"— rate: {self.profile.rate_wpm} wpm",
                fg=PALETTE['green'])

        threading.Thread(target=_work, daemon=True).start()

    # ── SYNTH TAB ─────────────────────────────────────────────────────────

    def _build_synth_tab(self):
        P = PALETTE
        tab = self.tab_synth

        # text input
        tk.Label(tab, text="Text to Synthesize",
                 bg=P['bg'], fg=P['white'],
                 font=('Segoe UI', 11, 'bold')).pack(anchor='w', padx=14, pady=(14,4))

        self.synth_text = tk.Text(tab, height=7, wrap='word',
                                   bg=P['card'], fg=P['text'],
                                   insertbackground=P['white'],
                                   font=('Segoe UI', 11),
                                   relief='flat', bd=8,
                                   selectbackground=P['accent2'])
        self.synth_text.pack(fill='x', padx=12)
        self.synth_text.insert('1.0',
            "Type or paste any text here — it will be spoken in your cloned voice.")
        self.synth_text.bind('<FocusIn>', self._clear_placeholder)

        # char/word count
        self.word_count_lbl = tk.Label(tab, text='',
                                        bg=P['bg'], fg=P['muted'],
                                        font=('Segoe UI', 8))
        self.word_count_lbl.pack(anchor='e', padx=16)
        self.synth_text.bind('<KeyRelease>', self._update_word_count)

        # buttons row
        btn_row = tk.Frame(tab, bg=P['bg'])
        btn_row.pack(fill='x', padx=12, pady=8)

        self.speak_btn = tk.Button(btn_row, text='▶  Speak',
                                    bg=P['green'], fg=P['surface'],
                                    font=('Segoe UI', 10, 'bold'),
                                    relief='flat', bd=0, padx=18, pady=8,
                                    cursor='hand2', command=self._speak)
        self.speak_btn.pack(side='left', padx=(0,8))

        self.stop_btn = tk.Button(btn_row, text='⏹  Stop',
                                   bg=P['accent'], fg=P['white'],
                                   font=('Segoe UI', 10, 'bold'),
                                   relief='flat', bd=0, padx=18, pady=8,
                                   cursor='hand2', command=self._stop_speak,
                                   state='disabled')
        self.stop_btn.pack(side='left', padx=(0,8))

        tk.Button(btn_row, text='💾  Save as WAV',
                  bg=P['card'], fg=P['text'],
                  font=('Segoe UI', 9), relief='flat',
                  bd=0, padx=14, pady=8,
                  cursor='hand2', command=self._save_wav).pack(side='left', padx=(0,8))

        tk.Button(btn_row, text='🗑 Clear',
                  bg=P['card'], fg=P['muted'],
                  font=('Segoe UI', 9), relief='flat',
                  bd=0, padx=10, pady=8,
                  cursor='hand2', command=self._clear_text).pack(side='left')

        # quick phrases
        tk.Label(tab, text="Quick phrases:",
                 bg=P['bg'], fg=P['muted'],
                 font=('Segoe UI', 9)).pack(anchor='w', padx=14, pady=(6,2))
        phrases_frame = tk.Frame(tab, bg=P['bg'])
        phrases_frame.pack(fill='x', padx=12)
        for phrase in [
            "Hello, how are you?",
            "The quick brown fox jumps over the lazy dog.",
            "Today is a beautiful day to learn something new.",
            "Voice cloning technology is truly remarkable.",
        ]:
            short = phrase[:32] + '…' if len(phrase) > 32 else phrase
            tk.Button(phrases_frame, text=short,
                      bg=P['surface'], fg=P['muted'],
                      font=('Segoe UI', 8), relief='flat',
                      bd=0, padx=8, pady=4, cursor='hand2',
                      command=lambda p=phrase: self._insert_phrase(p)
                      ).pack(side='left', padx=3, pady=3)

        # output / speaking indicator
        self.synth_status = tk.Label(tab, text='',
                                      bg=P['bg'], fg=P['green'],
                                      font=('Segoe UI', 10))
        self.synth_status.pack(pady=8)

        # profile info
        prof_frame = tk.Frame(tab, bg=P['surface'])
        prof_frame.pack(fill='x', padx=12, pady=6, side='bottom')
        self.prof_info_lbl = tk.Label(prof_frame, text='',
                                       bg=P['surface'], fg=P['muted'],
                                       font=('Segoe UI', 8), pady=6, padx=10)
        self.prof_info_lbl.pack(anchor='w')
        self._refresh_prof_info()

    def _clear_placeholder(self, _e=None):
        txt = self.synth_text.get('1.0', 'end-1c')
        if txt.startswith("Type or paste"):
            self.synth_text.delete('1.0', 'end')

    def _update_word_count(self, _e=None):
        txt = self.synth_text.get('1.0', 'end-1c').strip()
        words = len(txt.split()) if txt else 0
        self.word_count_lbl.config(text=f'{words} words')

    def _insert_phrase(self, phrase):
        self.synth_text.delete('1.0', 'end')
        self.synth_text.insert('1.0', phrase)

    def _clear_text(self):
        self.synth_text.delete('1.0', 'end')
        self.synth_status.config(text='')

    def _speak(self):
        if not HAS_TTS:
            messagebox.showerror("Missing dependency",
                                  "pyttsx3 is not installed.\nRun install_deps.bat")
            return
        text = self.synth_text.get('1.0', 'end-1c').strip()
        if not text or text.startswith("Type or paste"):
            messagebox.showwarning("No text", "Please enter some text to speak.")
            return
        self._speaking = True
        self.speak_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.synth_status.config(text='🔊 Speaking…', fg=PALETTE['green'])

        def _done():
            self._speaking = False
            self.speak_btn.config(state='normal')
            self.stop_btn.config(state='disabled')
            self.synth_status.config(text='✓ Done', fg=PALETTE['muted'])

        self.tts.speak(text, self.profile, done_cb=lambda: self.after(0, _done))

    def _stop_speak(self):
        self.tts.stop()
        self._speaking = False
        self.speak_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.synth_status.config(text='', fg=PALETTE['muted'])

    def _save_wav(self):
        if not HAS_TTS:
            messagebox.showerror("Missing", "pyttsx3 not installed.")
            return
        text = self.synth_text.get('1.0', 'end-1c').strip()
        if not text or text.startswith("Type or paste"):
            messagebox.showwarning("No text", "Please enter text first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension='.wav',
            filetypes=[('WAV audio', '*.wav'), ('MP3 audio', '*.mp3'), ('All files', '*.*')],
            initialdir=str(EXPORTS_DIR),
            title='Save synthesized audio')
        if not path:
            return
        self.synth_status.config(text='Saving…', fg=PALETTE['amber'])
        def _done():
            self.after(0, lambda: self.synth_status.config(
                text=f'Saved to {Path(path).name}', fg=PALETTE['green']))
        self.tts.save_to_file(text, Path(path), self.profile, done_cb=_done)

    def _refresh_prof_info(self):
        if self.profile.pitch_hz:
            self.prof_info_lbl.config(
                text=f"Profile: {self.profile.name}  |  "
                     f"Pitch ≈ {self.profile.pitch_hz:.0f} Hz  |  "
                     f"Rate ≈ {self.profile.rate_wpm} wpm  |  "
                     f"Samples: {len(self.profile.sample_files)}")
        else:
            self.prof_info_lbl.config(
                text="No voice profile built yet — go to Train tab first.")

    # ── SETTINGS TAB ──────────────────────────────────────────────────────

    def _build_settings_tab(self):
        P = PALETTE
        tab = self.tab_settings

        def section(label):
            tk.Label(tab, text=label, bg=P['bg'], fg=P['white'],
                     font=('Segoe UI', 10, 'bold')).pack(
                anchor='w', padx=14, pady=(14,4))
            tk.Frame(tab, height=1, bg=P['border']).pack(fill='x', padx=12)

        def slider_row(parent, label, var, from_, to, res, fmt_fn=None):
            row = tk.Frame(parent, bg=P['surface'])
            row.pack(fill='x', padx=14, pady=3)
            tk.Label(row, text=label, bg=P['surface'], fg=P['text'],
                     font=('Segoe UI', 9), width=18, anchor='w').pack(side='left')
            val_lbl = tk.Label(row, text=str(var.get()),
                                bg=P['surface'], fg=P['green'],
                                font=('Segoe UI', 9, 'bold'), width=6)
            val_lbl.pack(side='right')
            def _upd(v):
                v2 = float(v)
                txt = fmt_fn(v2) if fmt_fn else f'{v2:.1f}'
                val_lbl.config(text=txt)
                self._save_settings()
            s = ttk.Scale(row, from_=from_, to=to, orient='horizontal',
                          variable=var, command=_upd, length=220)
            s.pack(side='left', padx=8)
            return s

        section("Voice Tuning")
        sf = tk.Frame(tab, bg=P['surface'])
        sf.pack(fill='x', padx=12, pady=4)

        self.pitch_var  = tk.DoubleVar(value=self.profile.pitch_adj)
        self.speed_var  = tk.DoubleVar(value=self.profile.speed_adj)
        self.vol_var    = tk.DoubleVar(value=self.profile.volume_adj)

        slider_row(sf, "Pitch adjust", self.pitch_var,
                   -10, 10, 1, lambda v: f'{int(v):+d}')
        slider_row(sf, "Speed (×)",    self.speed_var,
                   0.5, 2.0, 0.05, lambda v: f'{v:.2f}×')
        slider_row(sf, "Volume",       self.vol_var,
                   0.1, 1.0, 0.05, lambda v: f'{v:.0%}')

        section("Directories")
        df = tk.Frame(tab, bg=P['surface'])
        df.pack(fill='x', padx=12, pady=4)
        for label, path in [
            ("Recordings", RECORDINGS_DIR),
            ("Profiles",   PROFILES_DIR),
            ("Exports",    EXPORTS_DIR),
        ]:
            row = tk.Frame(df, bg=P['surface'])
            row.pack(fill='x', padx=14, pady=2)
            tk.Label(row, text=label, bg=P['surface'], fg=P['muted'],
                     font=('Segoe UI', 9), width=12, anchor='w').pack(side='left')
            tk.Label(row, text=str(path), bg=P['surface'], fg=P['text'],
                     font=('Segoe UI', 8)).pack(side='left', padx=8)
            tk.Button(row, text='Open',
                      bg=P['card'], fg=P['text'],
                      font=('Segoe UI', 8), relief='flat',
                      bd=0, padx=8, pady=2, cursor='hand2',
                      command=lambda p=path: os.startfile(str(p))
                      ).pack(side='right')

        section("Dependency Status")
        deps = [
            ("sounddevice (recording)", HAS_SD),
            ("pyttsx3 (TTS synthesis)", HAS_TTS),
            ("numpy (audio analysis)",  HAS_NUMPY),
            ("scipy (DSP / WAV IO)",    HAS_SCIPY),
            ("pydub (MP3 export)",      HAS_PYDUB),
        ]
        dep_frame = tk.Frame(tab, bg=P['surface'])
        dep_frame.pack(fill='x', padx=12, pady=4)
        for name, ok in deps:
            row = tk.Frame(dep_frame, bg=P['surface'])
            row.pack(fill='x', padx=14, pady=2)
            dot = "✓" if ok else "✗"
            color = P['green'] if ok else P['accent']
            tk.Label(row, text=dot, bg=P['surface'], fg=color,
                     font=('Segoe UI', 10, 'bold'), width=2).pack(side='left')
            tk.Label(row, text=name, bg=P['surface'], fg=P['text'],
                     font=('Segoe UI', 9)).pack(side='left', padx=6)

        # save / reset
        btn_row = tk.Frame(tab, bg=P['bg'])
        btn_row.pack(fill='x', padx=12, pady=14, side='bottom')
        tk.Button(btn_row, text='Save Settings',
                  bg=P['green'], fg=P['surface'],
                  font=('Segoe UI', 10, 'bold'),
                  relief='flat', bd=0, padx=16, pady=8,
                  cursor='hand2', command=self._save_settings).pack(side='left', padx=4)
        tk.Button(btn_row, text='Delete All Recordings',
                  bg=P['accent'], fg=P['white'],
                  font=('Segoe UI', 9),
                  relief='flat', bd=0, padx=16, pady=8,
                  cursor='hand2', command=self._delete_all).pack(side='right', padx=4)

    def _save_settings(self):
        self.profile.pitch_adj  = float(self.pitch_var.get())
        self.profile.speed_adj  = float(self.speed_var.get())
        self.profile.volume_adj = float(self.vol_var.get())
        self.profile.save(self.profile_path)
        self._refresh_prof_info()

    def _delete_all(self):
        if messagebox.askyesno("Confirm",
                                "Delete ALL recordings and the voice profile?"):
            shutil.rmtree(str(RECORDINGS_DIR), ignore_errors=True)
            RECORDINGS_DIR.mkdir(exist_ok=True)
            for row in self.sample_rows:
                row._delete()
            self.profile = VoiceProfile()
            self.profile.save(self.profile_path)

    # ── RECORDING CONTROL ─────────────────────────────────────────────────

    def start_recording(self, idx: int):
        if not HAS_SD:
            messagebox.showerror("Missing",
                                  "sounddevice is not installed.\nRun install_deps.bat")
            return
        # stop any active recording
        if self._active_rec_row is not None:
            self.stop_recording(self._active_rec_row.idx)

        row = self.sample_rows[idx]
        self._active_rec_row = row
        row.set_recording()
        self.status_lbl.config(text=f'Recording sample {idx+1}…', fg=PALETTE['accent'])

        def _level_cb(rms):
            self.after(0, lambda: row.update_level(rms))

        try:
            self.recorder.start(callback_level=_level_cb)
        except Exception as e:
            messagebox.showerror("Mic error", str(e))
            row.state = 'idle'
            self._active_rec_row = None

    def stop_recording(self, idx: int):
        row = self.sample_rows[idx]
        frames = self.recorder.stop()
        if frames:
            ok = self.recorder.save(row.wav_path)
            if ok:
                row.set_done()
        else:
            row.state = 'idle'
            row.rec_btn.config(text='⏺ Record', bg=PALETTE['accent2'])
        self._active_rec_row = None
        self.status_lbl.config(text='Ready', fg=PALETTE['muted'])
        self.update_status()

    # ── STATUS ────────────────────────────────────────────────────────────

    def update_status(self):
        n = sum(1 for r in self.sample_rows if r.wav_path.exists())
        total = len(self.sample_rows)
        self.rec_status_lbl.config(
            text=f'{n} of {total} samples recorded  '
                 f'({"✓ ready to train" if n >= 3 else f"record {3-n} more to train"})  ',
            fg=PALETTE['green'] if n >= 3 else PALETTE['muted'])
        # update train summary
        self.summary_lbl.config(
            text=f'{n} WAV file(s) ready  ·  '
                 f'Profile: {self.profile.name}  ·  '
                 f'Stored in: {RECORDINGS_DIR}')
        q = min(100, int(n / max(len(SAMPLE_SENTENCES), 1) * 100))
        self.quality_bar['value'] = q
        self.quality_lbl.config(text=f'{q}%')

    def _dep_status(self):
        ok = sum([HAS_SD, HAS_TTS, HAS_NUMPY, HAS_SCIPY])
        return f"Deps: {ok}/4 installed"


def main():
    ensure_dirs()
    app = VoiceStudioApp()
    app.mainloop()


if __name__ == '__main__':
    main()
