import pickle
from datetime import datetime
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from scipy.io import savemat
from scipy.ndimage import maximum_filter1d, uniform_filter1d

ENVELOPE_FRAME_MS = 10
AUDIO_DIR = "/itf-fi-ml/home/sagardu/djembe_drive/Data/DD_MultiM_Proc/AudioVideo_proc"
OUT_DIR = Path("data/drum_envelope")
BASE_PATH_CYCLES = Path("data/virtual_cycles")
DANCE_MODE_DIR = Path("data/dance_modes_ts")
DRUM_LABELS = ("M-Dun", "M-Jem-1", "M-Jem-2", "P-Dun", "P-Jem-1", "P-Jem-2")
MODES = ("group", "individual", "audience")
PHASE_STEP = 0.01
METADATA_KEYS = (
    "file_name",
    "dmode_name",
    "dmode_seg_idx",
    "dmode_start",
    "dmode_end",
    "cycle_idx",
    "cycle_start",
    "cycle_end",
    "cycle_frame_times",
    "location",
    "ensemble",
    "day",
    "rec_no",
    "piece",
)


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


def sub_dir_for_piece(piece_id: str) -> str:
    """BKO_E1_D1_01_Suku -> BKO_E1_D1"""
    return "_".join(piece_id.split("_")[:3])


def find_drum_wav(piece_id: str, label: str) -> Path | None:
    sub_dir_path = Path(AUDIO_DIR) / sub_dir_for_piece(piece_id)
    wav_path = sub_dir_path / f"{piece_id}_{label}.wav"
    return wav_path if wav_path.exists() else None


def make_phase_grid(step: float = PHASE_STEP) -> np.ndarray:
    n_phase = int(round(1.0 / step)) + 1
    return np.linspace(0.0, 1.0, n_phase)


def parse_piece_metadata(piece_id: str) -> dict[str, str] | None:
    parts = piece_id.split("_")
    if len(parts) < 5:
        print(f"Unexpected piece id format: {piece_id}")
        return None

    return {
        "location": parts[0],
        "ensemble": parts[1],
        "day": parts[2],
        "rec_no": parts[3],
        "piece": "_".join(parts[4:]),
    }


def load_drum_tracks(piece_id: str) -> dict[str, dict] | None:
    drum_tracks: dict[str, dict] = {}

    for label in DRUM_LABELS:
        wav_path = find_drum_wav(piece_id, label)
        if wav_path is None:
            print(f"No drum wav for {piece_id} / {label}")
            return None

        y, sr = librosa.load(str(wav_path), sr=None)
        time, _, envelope_smooth = extract_envelope(y, sr)
        drum_tracks[label] = {
            "time": time,
            "envelope": envelope_smooth,
        }

    return drum_tracks


def load_virtual_onsets(piece_id: str) -> np.ndarray | None:
    cycles_csv = BASE_PATH_CYCLES / f"{piece_id}_C.csv"
    if not cycles_csv.exists():
        print(f"Cycles CSV not found: {cycles_csv}")
        return None

    try:
        return pd.read_csv(cycles_csv)["Virtual Onset"].values.astype(float)
    except Exception as exc:
        print(f"Failed to load cycles CSV for {piece_id}: {exc}")
        return None


def load_dance_mode_segments(piece_id: str, dance_mode: str) -> list[tuple[float, float]]:
    dmode_path = DANCE_MODE_DIR / f"{piece_id}_{dance_mode}.pkl"
    if not dmode_path.exists():
        return []

    try:
        with open(dmode_path, "rb") as fh:
            return [(float(start), float(end)) for start, end in pickle.load(fh)]
    except Exception as exc:
        print(f"Failed to load dance-mode file {dmode_path}: {exc}")
        return []


def resample_envelope_to_phase(
    time: np.ndarray,
    envelope: np.ndarray,
    cycle_start: float,
    cycle_end: float,
    phase_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target_times = cycle_start + phase_grid * (cycle_end - cycle_start)
    envelope_phase = np.interp(target_times, time, envelope)
    return target_times, envelope_phase


def initialize_cycle_data(phase_grid: np.ndarray) -> dict:
    cycle_data = {
        "envelopes": [],
        "phase_grid": phase_grid,
        "drum_labels": np.array(DRUM_LABELS, dtype=object),
    }
    for key in METADATA_KEYS:
        cycle_data[key] = []
    return cycle_data


def finalize_cycle_data(cycle_data: dict, phase_grid: np.ndarray) -> dict:
    if cycle_data["envelopes"]:
        cycle_data["envelopes"] = np.stack(cycle_data["envelopes"], axis=0)
        cycle_data["cycle_frame_times"] = np.stack(cycle_data["cycle_frame_times"], axis=0)
    else:
        n_phase = len(phase_grid)
        cycle_data["envelopes"] = np.empty((0, len(DRUM_LABELS), n_phase), dtype=float)
        cycle_data["cycle_frame_times"] = np.empty((0, n_phase), dtype=float)
    return cycle_data


def process_piece(piece_id: str, cycle_data: dict, phase_grid: np.ndarray) -> int | None:
    piece_metadata = parse_piece_metadata(piece_id)
    if piece_metadata is None:
        return None

    drum_tracks = load_drum_tracks(piece_id)
    if drum_tracks is None:
        return None

    all_onsets = load_virtual_onsets(piece_id)
    if all_onsets is None:
        return None

    n_before = len(cycle_data["file_name"])

    for dance_mode in MODES:
        dmode_ts = load_dance_mode_segments(piece_id, dance_mode)
        if not dmode_ts:
            continue

        for dmode_idx, (dmode_start, dmode_end) in enumerate(dmode_ts):
            mode_mask = (all_onsets >= dmode_start) & (all_onsets <= dmode_end)
            mode_onsets = all_onsets[mode_mask]

            if len(mode_onsets) < 2:
                continue

            cycle_boundaries = [
                (round(mode_onsets[i], 3), round(mode_onsets[i + 1], 3))
                for i in range(len(mode_onsets) - 1)
            ]

            for c_idx, (cycle_start, cycle_end) in enumerate(cycle_boundaries):
                cycle_envelopes = []
                cycle_frame_times = None

                for label in DRUM_LABELS:
                    target_times, envelope_phase = resample_envelope_to_phase(
                        drum_tracks[label]["time"],
                        drum_tracks[label]["envelope"],
                        cycle_start,
                        cycle_end,
                        phase_grid,
                    )
                    cycle_frame_times = target_times
                    cycle_envelopes.append(envelope_phase)

                cycle_data["envelopes"].append(np.stack(cycle_envelopes, axis=0))
                cycle_data["file_name"].append(piece_id)
                cycle_data["dmode_name"].append(dance_mode)
                cycle_data["dmode_seg_idx"].append(dmode_idx + 1)
                cycle_data["dmode_start"].append(dmode_start)
                cycle_data["dmode_end"].append(dmode_end)
                cycle_data["cycle_idx"].append(c_idx + 1)
                cycle_data["cycle_start"].append(cycle_start)
                cycle_data["cycle_end"].append(cycle_end)
                cycle_data["cycle_frame_times"].append(cycle_frame_times)
                cycle_data["location"].append(piece_metadata["location"])
                cycle_data["ensemble"].append(piece_metadata["ensemble"])
                cycle_data["day"].append(piece_metadata["day"])
                cycle_data["rec_no"].append(piece_metadata["rec_no"])
                cycle_data["piece"].append(piece_metadata["piece"])

    return len(cycle_data["file_name"]) - n_before


def process_all_files(piece_list: list[str]) -> dict:
    phase_grid = make_phase_grid()
    cycle_data = initialize_cycle_data(phase_grid)

    processed = 0
    failed = []

    for piece_id in piece_list:
        n_cycles = process_piece(piece_id, cycle_data, phase_grid)
        if n_cycles is None:
            failed.append(piece_id)
            continue

        processed += 1
        print(f"{piece_id}: {n_cycles} cycles")

    print(f"Processed {processed}/{len(piece_list)} pieces, "
          f"{len(cycle_data['file_name'])} total cycles")
    if failed:
        print(f"Skipped {len(failed)} piece(s): {failed}")

    return finalize_cycle_data(cycle_data, phase_grid)


def save_drum_envelopes(
    cycle_data: dict,
    out_dir: Path = OUT_DIR,
    today: str | None = None,
) -> dict[str, Path]:
    if today is None:
        today = datetime.now().strftime("%d%b").lower()

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cluster_drum_envelope_{today}"
    out_pkl = out_dir / f"{stem}.pkl"
    out_mat = out_dir / f"{stem}.mat"

    with open(out_pkl, "wb") as f:
        pickle.dump(cycle_data, f)
    savemat(out_mat, cycle_data)

    print(f"Saved pkl: {out_pkl}")
    print(f"Saved mat: {out_mat}")
    return {"pkl": out_pkl, "mat": out_mat}


def main() -> None:
    with open("data/selected_piece_list.pkl", "rb") as f:
        piece_list = pickle.load(f)

    cycle_data = process_all_files(piece_list)
    save_drum_envelopes(cycle_data)


if __name__ == "__main__":
    main()
