from pathlib import Path
import os
import librosa
import numpy as np
import pandas as pd
import pickle
from scipy.io import savemat
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.interpolate import interp1d
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

ENVELOPE_FRAME_MS = 10   # downsample to one value per frame

def extract_envelope(
    y: np.ndarray,
    sr: int,
    frame_ms: float = ENVELOPE_FRAME_MS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding max |amplitude| (hop = 1 sample), then moving-average smooth."""
    window_size = max(1, int(frame_ms / 1000 * sr))
    abs_y = np.abs(y)
    n = len(abs_y)

    envelope_raw = maximum_filter1d(abs_y, size=window_size, mode="reflect")
    envelope_smooth = uniform_filter1d(envelope_raw, size=window_size, mode="reflect")
    time = np.arange(n) / sr

    return time, envelope_raw, envelope_smooth

def plot_pitch_change_with_cycles(data: dict, window_size_ms: int = 500):
    """Plot audio/envelope/voice probability above, and F0 below."""

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 6), dpi=300, sharex=True
    )

    # row 1: raw audio, envelope, voice probability
    ax1.plot(data["time_audio"], np.abs(data["audio_cleaned"]),
             color="steelblue", linewidth=0.1, label="raw audio")

    ax1.plot(data["time_audio"], data["audio_envelope"],
             color="r", linewidth=0.1, label="envelope")

    # # moving average of voice probability

    # window_size = int(window_size_ms / 1000 * data["sr"])
    # voice_prob_smooth = uniform_filter1d(data["voice_prob_raw"], size=window_size, mode="reflect")
    voice_prob_smooth = data["voice_prob_smooth"]
    
    ax1.plot(data["time_audio"], voice_prob_smooth,
             color="g", linewidth=0.1, label="voice prob")
    
    ax1.axhline(0.06, color="r", linewidth=0.3,)
    ax1.axhline(0.04, color="r", linewidth=0.3,)
    ax1.axhline(0.02, color="r", linewidth=0.3,)
    
    
    ax1.set_ylabel("Amplitude / probability")
    ax1.set_title(data["piece"])
    ax1.legend(loc="upper right")
    # ax1.set_xlim(0, 60)

    # row 2: f0
    # f0_midi = np.log2(data["f0_plot"]/440 + 1e-8) * 12 + 69 
    
    ax2.plot(data["time_audio"], data["f0_midi"],                   # data["f0_midi"]
             color="g", linewidth=0.5, alpha=0.7)

    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("f0 (Hz)")
    ax2.legend(loc="upper right")
    # ax2.set_xlim(24, 44)
    ax2.set_ylim(40,80)

    plt.tight_layout()

    return fig, (ax1, ax2)

trimmed_dir = "data/vocal/vocal_trimmed"
# cycle_dir = "data/virtual_cycles"

HOP_LENGTH = 1024
MIN_F = "F3"
MAX_F = "E5"
FMIN = librosa.note_to_hz(MIN_F) # C4
FMAX = librosa.note_to_hz(MAX_F) # C6

pitch_dir = f"data/vocal/vocal_pitch/{MIN_F}-{MAX_F}"
Path(pitch_dir).mkdir(parents=True, exist_ok=True)

with open("data/selected_piece_list.pkl", "rb") as f:
    piece_list = pickle.load(f)


def find_trimmed_vocal(piece: str) -> Path | None:
    matches = sorted(Path(trimmed_dir).rglob(f"*{piece}*.wav"))
    return matches[0] if matches else None


def extract_pitch(y: np.ndarray, sr: int, hop_length: int = HOP_LENGTH) -> dict:
    
    f0, _, voiced_probs = librosa.pyin(
        y,
        fmin=FMIN,
        fmax=FMAX,
        sr=sr,
        hop_length=hop_length,
    )
    t_audio = np.arange(len(y)) / sr 
    time_f0 = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    
    # interpolate voice probability
    f_interp_voice_prob = interp1d(
        time_f0, voiced_probs,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    voice_prob_audio = f_interp_voice_prob(t_audio)  # same length as raw audio
    
    window_size_ms = 2000
    window_size = int(window_size_ms / 1000 * sr)
    voice_prob_smooth = uniform_filter1d(voice_prob_audio, size=window_size, mode="reflect")
    
    voiced_flag = voice_prob_smooth > 0.02

    
    # interpolate f0
    f_interp_f0 = interp1d(
        time_f0, np.nan_to_num(f0, nan=0.0),  # np.nan_to_num(f0_clean, nan=0.0)
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    f0_audio = f_interp_f0(t_audio)  # same length as raw audio
    
    # f0_plot
    
    f_interp_f0_plot = interp1d(
        time_f0, f0,  # np.nan_to_num(f0_clean, nan=0.0)
        kind="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    f0_plot = f_interp_f0_plot(t_audio)  # same length as raw audio
    f0_midi = np.log2(f0_plot/440 + 1e-8) * 12 + 69 
    
    return {
        "f0": f0_audio,
        "f0_midi": f0_midi,
        "f0_plot": f0_plot,
        # "voiced_flag": voiced_flag,
        # "voice_prob_raw": voice_prob_audio,
        "voice_prob_smooth": voice_prob_smooth,

    }


pitch_data = {}
for piece in piece_list:
    print("Processing piece: ", piece)
    vocal_path = find_trimmed_vocal(piece)
    if vocal_path is None:
        print(f"No trimmed vocal for {piece}")
        continue

    y, sr = librosa.load(str(vocal_path), sr=None, mono=True)
    duration = len(y) / sr
    time_y = np.arange(len(y)) / sr
    
    _, _, envelope_smooth = extract_envelope(y, sr, )
    
    ######## clean audio ########
    win_size = int(1000 / 1000 * sr)
    envelope_smooth1 = uniform_filter1d(envelope_smooth, size=win_size, mode="reflect")
    
    envelope_mask = envelope_smooth1 > 0.04
    audio_cleaned = envelope_mask * y
    ##############################
    
    pitch = extract_pitch(audio_cleaned, sr)
    # f0_smooth = envelope_smooth * pitch["f0"]   
    

    data = {
        "piece": piece,
        "time_audio": time_y,
        # "audio_raw": y,
        "audio_cleaned": audio_cleaned,
        "audio_envelope": envelope_smooth1,
        "sr": sr,
        # "duration": duration,
        # "wav_path": str(vocal_path),
        # "audio_enve*f0": f0_smooth, # env * f0
        **pitch,
    }
    # pitch_data[piece] = data
    print("Saving data...")
    # save data
    out_path = os.path.join(pitch_dir, "pitch_data" ,f"{piece}.mat")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    savemat(out_path, data)
    
    fig, _ = plot_pitch_change_with_cycles(data, window_size_ms=2000)
    plot_save_path = os.path.join(pitch_dir, "plots", f"{piece}.png")
    Path(plot_save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_save_path)
    plt.close(fig)
    
    #############################################
    # compute complex tone
    #############################################
    print("Computing complex tone: ", piece)
    # Instantaneous phase: phi[n] = phi[n-1] + 2*pi*f0[n]/sr
    phase = np.cumsum(2 * np.pi * data["f0"] / data["sr"])
    # Simple tone (pure fundamental)
    tone = np.sin(phase)
    # "Richer" / more vocal-like: additive harmonics
    tone = (
        1.00 * np.sin(1 * phase)
    + 0.50 * np.sin(2 * phase)
    + 0.25 * np.sin(3 * phase)
    )
    
    # voice_mask = data["voice_prob_smooth"] > 0.02
    # voiced_audio = voice_mask * data["audio_cleaned"]
    # audio_envelope_masked = voice_mask * data["audio_envelope"]


    amp = data["audio_envelope"] * data["f0"]  
    synth = amp * tone
    # Normalize
    synth_norm = synth / (np.max(np.abs(synth)) + 1e-8)

    audio_norm = data["audio_cleaned"] / (np.max(np.abs(data["audio_cleaned"])) + 1e-8)
    mix = 0.05 * audio_norm + 0.95 * synth_norm
    
    # save audio
    out_path = os.path.join(pitch_dir, "mixed_audio", f"{piece}_complex_tone.wav")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix, data["sr"])
    
