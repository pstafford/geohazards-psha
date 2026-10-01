# A complete PSHA you can read in an afternoon

Tutorial material for **Geotechnical Hazards** (Imperial College London), Lectures 10 and 11:
a probabilistic seismic hazard analysis (PSHA) of a toy model, written so that every step can be
read and changed, and checked against **OpenQuake**.

| | No installation | |
|---|---|---|
| **PSHA Explorer** | runs in your browser: change the model, see hazard curves, spectra and disaggregation | [open](https://pstafford.github.io/geohazards-psha/) |
| **1. PSHA from scratch** | the whole calculation in numpy, step by step | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pstafford/geohazards-psha/blob/main/notebooks/01_psha_from_scratch.ipynb) |
| **2. The same PSHA in OpenQuake** | the model in OpenQuake's hazardlib, and the comparison | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pstafford/geohazards-psha/blob/main/notebooks/02_psha_with_openquake.ipynb) |

Colab needs only a Google account: open a notebook and choose *Runtime → Run all*. No Python
experience is needed.

## The model

A site on rock (V<sub>S30</sub> = 760 m/s) and two sources:

* a vertical strike-slip **fault**, 115 km long, 50 km from the site, 0–18 km deep, with a
  characteristic magnitude–frequency distribution (Youngs & Coppersmith, 1985) whose activity comes
  from a 30 mm/yr slip rate by moment balance; ruptures float along and down the fault;
* a circular **area source** of radius 70 km around the site, with a truncated Gutenberg–Richter
  distribution (b = 1, 5 ≤ M ≤ 6.5, 0.5 earthquakes of M ≥ 5 per year) at 9 km depth;

and the Chiou & Youngs (2014) ground-motion model, truncated at ± 3σ. Every modelling choice
follows OpenQuake's conventions, so the two calculations agree (to about 0.1 % on design values).

## Files

* `psha/psha_toy.py`: the model in plain numpy: scenarios, ground motions, hazard curves,
  disaggregation and conditional mean spectra (about 300 lines)
* `psha/psha_oq.py`: the same model built with OpenQuake hazardlib
* `psha/cy14_coefficients.csv`: the coefficients of Chiou & Youngs (2014), from OpenQuake
* `notebooks/`: the two notebooks
* `docs/index.html`: the PSHA Explorer (a single file; it also works offline once downloaded)

## On your own computer

    pip install numpy scipy pandas matplotlib jupyter
    jupyter lab notebooks/01_psha_from_scratch.ipynb

The OpenQuake notebook also needs `openquake.engine==3.26.2` (see its first cell).
