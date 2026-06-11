import os
import multiprocess as mp
import pandas as pd
import numpy as np
import librosa
import warnings
import torchaudio
import itertools
import re
from scipy.signal import hilbert
#import pesto

warnings.filterwarnings('ignore')  # ignore audio read warning

# ---- Defaults in case ever need them ----
DEFAULT_MIN_F = librosa.note_to_hz('C6')
DEFAULT_MAX_F = librosa.note_to_hz('C4')  # ~2093 Hz

# Path to your Excel with per-file ranges (MIDI units)
RANGE_XLSX = None

def extract_all(range_xlsx=None, no_overwrite=True):
    """
    Runs the full extraction pass using current config and Excel filter.
    - range_xlsx: optional path to the Excel. If provided, reloads RANGE_MAP.
    - no_overwrite=True: skip files whose CSV already exists.
    """
    global RANGE_XLSX, RANGE_MAP
    if range_xlsx is not None:
        RANGE_XLSX = range_xlsx
        RANGE_MAP = load_range_map(RANGE_XLSX)

    input_paths, output_paths, file_names = get_process_files()
    print(f"Starting multiprocessing with {N_CORE} cores for {len(input_paths)} file(s)...")
    if not input_paths:
        print("Nothing to do (Excel empty or all CSVs already exist).")
        return

    with mp.Pool(N_CORE) as pool:
        pool.starmap(
            extract_melody,
            zip(input_paths, output_paths, file_names, itertools.repeat(no_overwrite))
        )


# ---------------------------------------------------------------------
# Helper functions to read Excel and convert per-file MIDI → Hz values
# ---------------------------------------------------------------------
def _safe_midi_to_hz(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(librosa.midi_to_hz(float(x)))
    except Exception:
        return np.nan

def load_range_map(path):
    """
    Reads Excel with columns: filename, MIN_F, MAX_F (MIDI).
    Keeps ONLY rows where BOTH MIN_F and MAX_F are present (not blank),
    and returns { basename: (min_hz, max_hz) }.
    """
    if not path or not os.path.exists(path):
        return {}

    df = pd.read_excel(path)

    # strict capitalization as provided
    required_cols = {"filename", "MIN_F", "MAX_F"}
    if not required_cols.issubset(df.columns):
        print(f"Excel missing columns {required_cols}; no files will be processed.")
        return {}

    # Keep only rows with both MIN_F and MAX_F filled
    df = df[df["MIN_F"].notna() & df["MAX_F"].notna()].copy()
    if df.empty:
        return {}

    # Convert MIDI -> Hz
    df["MIN_HZ"] = df["MIN_F"].apply(_safe_midi_to_hz)
    df["MAX_HZ"] = df["MAX_F"].apply(_safe_midi_to_hz)

    # Also drop any rows that failed conversion
    df = df[df["MIN_HZ"].notna() & df["MAX_HZ"].notna()].copy()

    range_map = {}
    for _, row in df.iterrows():
        fname = str(row["filename"]).strip()
        if not fname or fname.lower() == "nan":
            continue
        base = os.path.splitext(fname)[0]
        min_hz = float(row["MIN_HZ"])
        max_hz = float(row["MAX_HZ"])
        if min_hz >= max_hz:
            continue  # ignore nonsense rows
        range_map[base] = (min_hz, max_hz)

    return range_map

# Load once on startup
RANGE_MAP = load_range_map(RANGE_XLSX)

def get_bounds_for_file(file_name):
    """Return (min_hz, max_hz) if present in Excel; else None."""
    base = os.path.splitext(file_name)[0]
    return RANGE_MAP.get(base, None)



######################
## HYPER PARAMETERS ##
######################
# ---- SAMPLING CONSISTENCY ----
TARGET_SR = 22050         # default of librosa
FRAME_MS  = 46            # analysis window in ms (~46 ms common for pYIN)
HOP_MS    = 10            # hop in ms (10 ms = 100 fps)

def ms_to_samples(ms, sr):
    return int(round(ms * sr / 1000))

FRAME_LENGTH = ms_to_samples(FRAME_MS, TARGET_SR)
HOP_LENGTH   = ms_to_samples(HOP_MS,   TARGET_SR)

METHOD = "pyin"  # or 'pesto'

# where the audio files are located
DIR = "data/vocal_trimmed"  

# where the analysis csv should be saved
OUTDIR = "data/vocal_pitch"

# if SOURCE_SEP = False, we'll allow common audio formats
ALLOWED_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}

# NEW: restrict processing to a list of NHSDiscography numbers
# e.g. for NHSDiscography-003.wav, NHSDiscography-070.wav, ...
ALLOWED_NUMBERS = None
# for faster processing start multiprocessing
N_CORE = 7

# whether source separate was done. If True, it will look for all the folders and then grab vocal file from each folder
SOURCE_SEP = False

# audio file extension only relevant when SOURCE_SEP = True
AUDIO_EXT = 'wav'

# if SOURCE_SEP = False, we'll allow common audio formats
ALLOWED_EXTS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.aac'}


##############
## FUNCTION ##
##############
def get_process_files():
    os.makedirs(OUTDIR, exist_ok=True)

    # Build list of candidate audio files in DIR
    if SOURCE_SEP:
        folder_names = [f for f in os.listdir(DIR) if f != '.DS_Store']
        file_names_all = folder_names
        file_paths_all = [os.path.join(DIR, f, f"vocals.{AUDIO_EXT}") for f in file_names_all]
        output_paths_all = [os.path.join(OUTDIR, f) + '.csv' for f in file_names_all]
    else:
        files = [f for f in os.listdir(DIR)
                 if f != '.DS_Store' and os.path.splitext(f)[1].lower() in ALLOWED_EXTS]
        file_names_all = [os.path.splitext(f)[0] for f in files]
        file_paths_all = [os.path.join(DIR, f) for f in files]
        output_paths_all = [os.path.join(OUTDIR, fn) + '.csv' for fn in file_names_all]

    # Filter to only those present in RANGE_MAP (i.e., Excel has BOTH MIN_F & MAX_F)
    # and that do NOT already have a CSV (incremental behavior)
    input_paths, output_paths, file_names = [], [], []
    for fp, op, fn in zip(file_paths_all, output_paths_all, file_names_all):

        # OPTIONAL: restrict to specific NHSDiscography-XXX files
        # fn is the basename without extension, e.g. "NHSDiscography-003"
        if ALLOWED_NUMBERS is not None:
            m = re.search(r'(\d+)$', fn)  # captures the trailing digits
            if not m:
                continue
            num = int(m.group(1))        # "003" -> 3
            if num not in ALLOWED_NUMBERS:
                continue

        # Existing Excel + no-overwrite filter
        if not os.path.exists(op):
            input_paths.append(fp)
            output_paths.append(op)
            file_names.append(fn)


    print(f"Found {len(file_paths_all)} audio file(s) in {DIR}.")
    print(f"Will process {len(input_paths)} new file(s) (after ALLOWED_NUMBERS filter, skipping existing CSVs).")
    print(f"Processed files will be stored in {OUTDIR}")
    print(f"TARGET_SR: {TARGET_SR}, FRAME_LENGTH: {FRAME_LENGTH}, HOP_LENGTH: {HOP_LENGTH}")
    return input_paths, output_paths, file_names




def extract_melody(file_path, output_path, file_name, no_overwrite=True):
    # if no_overwrite = True and file already exists, skip
    if os.path.exists(output_path) and no_overwrite:
        print(f"Skipping {file_name} as it already exists...")
        return

    try:
        print(f"Processing {file_name} ...")

        if METHOD == 'pyin':
            # ---- DEFAULT BOUNDS since no Excel provided ----
            min_f_hz = DEFAULT_MIN_F
            max_f_hz = DEFAULT_MAX_F

            # ---- LOAD & RESAMPLE (CONSISTENT SR) ----
            y, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

            # ---- F0 EXTRACTION ----

            f0, voice_flag, voice_prob = librosa.pyin(
                y=y,
                sr=TARGET_SR,
                fmin=min_f_hz,
                fmax=max_f_hz,
                frame_length=FRAME_LENGTH,
                hop_length=HOP_LENGTH,
                switch_prob=0.01
            )

            # ---- Build time axes ----
            n_frames = len(f0)
            frame_idx = np.arange(n_frames)
            time_s = librosa.frames_to_time(frame_idx, sr=TARGET_SR, hop_length=HOP_LENGTH)

            # ---- Package output ----
            output = pd.DataFrame({
                "file":        file_name,
                "time_frame":  frame_idx,
                "time_s":      time_s,
                "frequency":   f0,                       # Hz; may contain NaN for unvoiced
                "voice_flag":  voice_flag.astype(bool),
                "confidence":  voice_prob
            })
            output["sr"]     = TARGET_SR
            output["hop_s"]  = HOP_LENGTH / TARGET_SR
            output["method"] = "pyin"

        elif METHOD == 'pesto':
            # ---- LOAD ----
            wave, orig_sr = torchaudio.load(file_path)   # [channels, samples]
            if wave.shape[0] > 1:
                wave = wave.mean(dim=0, keepdim=True)
            if orig_sr != TARGET_SR:
                wave = torchaudio.functional.resample(wave, orig_sr, TARGET_SR)
            sr = TARGET_SR

            # pesto expects seconds step; keep analysis grid consistent (~10 ms)
            step_ms = HOP_MS
            timesteps, pitch, confidence, activations = pesto.predict(
                wave.squeeze(0).numpy(), sr,
                step_size=step_ms,
                convert_to_freq=True
            )

            output = pd.DataFrame({
                "file":       file_name,
                "time_s":     timesteps,   # pesto returns seconds
                "frequency":  pitch,
                "confidence": confidence
            })
            # derive frame index based on hop
            output["time_frame"] = (output["time_s"] / (HOP_LENGTH / TARGET_SR)).round().astype(int)
            output["sr"]     = TARGET_SR
            output["hop_s"]  = HOP_LENGTH / TARGET_SR
            output["method"] = "pesto"

        else:
            raise ValueError("Please define a valid method! Use 'pyin' or 'pesto'.")

        # ---- Write CSV ----
        output.to_csv(output_path, index=False)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"ERROR processing {file_name}: {e}")


######################
## MULTI PROCESSING ##
######################
input_paths, output_paths, file_names = get_process_files()
print(f"Starting multiprocessing with {N_CORE} cores...")
if __name__ == '__main__':
    with mp.Pool(N_CORE) as pool:
        pool.starmap(extract_melody, zip(input_paths, output_paths, file_names))
    pool.close()
