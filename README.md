# Recursive CCS - Exploratory work on extending CCS

Preliminary results here:
https://docs.google.com/document/d/1LCjjnUPN51gHl_rmCWEmmtbY-Wu1dixzOif14e-7i-U

Structure of the repository:
- generate many hidden states with `generate_main.py` (it uses Collin's code from the zip files, copied in `utils_generation`)
- find direction using CCS recursively by adding new constraints each time using `generate_ccs_dirs_main.py` (it uses Collin's code from the github repo + bugfixes + custom things I want for recursion & debuging, the core functions are in `utils.py`). You can see what I use it for by looking at `script.sh` (it's a bit messy).
- analyze the directions found by CCS by running `analyze_ccs_dirs_main.py` (looking at a single layer) or `analyze_ccs_dirs_layers_main.py` (looking at multiple layers at a time) as Python notebooks.
- get a sense of what the CCS loss does by running `vizualisation.py` as a Python notebook.
- see what happens when you mix directions by running `mixing_directions.py` as a Python notebook.