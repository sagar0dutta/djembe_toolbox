Drum Envelope Data Fields
=========================

File: cluster_drum_envelope_<date>.pkl / .mat

All pieces merged into one dataset. Cycle index i is aligned across all
fields below.


Main arrays
-----------

envelopes
  Shape: (n_cycles, 6, 101)
  Dtype: float64
  Smoothed drum envelope amplitude on the phase grid.
  Axis 0: cycle index
  Axis 1: drum label index (see drum_labels)
  Axis 2: phase index (see phase_grid)

phase_grid
  Shape: (101,)
  Values: 0.00, 0.01, ..., 1.00
  Unitless cycle phase from cycle_start to cycle_end.

drum_labels
  Length: 6
  Order:
    0  M-Dun
    1  M-Jem-1
    2  M-Jem-2
    3  P-Dun
    4  P-Jem-1
    5  P-Jem-2


Per-cycle metadata
------------------

All metadata lists have length n_cycles and align with envelopes[i, ...].

file_name
  Piece identifier, e.g. BKO_E1_D1_01_Suku

dmode_name
  Dance mode: group, individual, or audience

dmode_seg_idx
  Segment index within the current dance mode (1-based)

dmode_start, dmode_end
  Start/end time (seconds) of the dance-mode segment

cycle_idx
  Cycle index within the current dance-mode segment (1-based)

cycle_start, cycle_end
  Start/end time (seconds) of the virtual cycle

cycle_frame_times
  Shape: (n_cycles, 101)
  Absolute time in seconds for each phase-grid point of that cycle

location
  Recording location, e.g. BKO

ensemble
  Ensemble id, e.g. E1

day
  Recording day id, e.g. D1

rec_no
  Recording number within day, e.g. 01

piece
  Piece/rhythm name, e.g. Suku
