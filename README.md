# djembe_toolbox

04 June 2026

A2, A3 verified for refactoring--- all good

A4 -- all good except
- feet plots but save dir does not indicate this
- hardcoded paths
- plot_trajectories_downbeat_window() takes wstart wend
- plot_trajectories_by_beat() takes List of (start, end) tuples -- should take wstart wend
- plot_trajectories_by_subdiv() takes List of (start, end) tuples -- should take wstart wend

A5 ---by dance mode
- saving dance mode plts as output_path=f"output_static_plot/{mode_name}" 
- all good till specified in notebook

A6
- maybe discard
- 

A7 are fine

A8 are version 1 -- can ignore for now, i can refactor to make it cleaner


Functions for the toolbox:
plot_trajectories_downbeat_window()
plot_trajectories_by_beat()
plot_trajectories_by_subdiv()

version 1: for left and right feet

function input
path to filename
path to feet onset csv (one csv two columns for left and right)
path to feet vertical trajectory csv
So no hardcoded path inside the functions
rest parameter

version 2: 