# ✨ Gen TSO ✨
### A general ETC interface for time-series observations with JWST

<!--
[![Tests](https://github.com/pcubillos/gen_tso/actions/workflows/python-package.yml/badge.svg?branch=master)](https://github.com/pcubillos/gen_tso/actions/workflows/python-package.yml)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/gen_tso.svg)](https://anaconda.org/conda-forge/gen_tso)
-->

[![GitHub Actions Workflow Status](https://img.shields.io/github/actions/workflow/status/pcubillos/gen_tso/quarto-publish.yml?label=docs)](https://pcubillos.github.io/gen_tso/)
[![PyPI](https://img.shields.io/pypi/v/gen_tso.svg)](https://pypi.org/project/gen_tso)
[![GitHub License](https://img.shields.io/github/license/pcubillos/gen_tso?style=flat&color=blue)](https://github.com/pcubillos/gen_tso/blob/master/LICENSE)


> Gen TSO is up to date with ``Pandeia`` version **2026.2**


### Install as:

Available for Python 3.9+ from PyPI:
```
pip install gen_tso
```

Then setup the pandeia reference files following these instructions: [https://pcubillos.github.io/gen_tso/install.html](https://pcubillos.github.io/gen_tso/install.html#pandeia-engine-installation)

To launch the application use this prompt command:
```
tso
```


Run with ``-h`` to display the full list of command line options:
```
tso -h
```

<details>
<summary>Click to expand</summary>

```shell
usage: tso [-h] [-v] [-m PATH] [-t FILE] [--debug]

Launch the Gen TSO interactive application

options:
  -h, --help            show this help message and exit
  -v, --version         show Gen TSO version
  -m PATH, --models PATH
                        include custom SED and planet spectra from input path
  -t FILE, --targets FILE
                        include custom targets from input file
  --debug               run reloading the GUI when the source code is updated

Other functionality:
  usage: tso [--update_exo] [--update_programs] [--update_db]
             [--update_custom CSV] [--add_custom CSV]

  --update_exo          update NASA Exoplanet Archive
  --update_programs     update JWST TSO programs
  --update_db           check and update SED/synphot atlases
  --update_custom CSV   update custom targets from csv file
  --add_custom CSV      add new custom targets from csv file

This is Gen TSO version 1.4.0
Copyright (c) 2025-2026 Patricio Cubillos. GPL-2.0 license
Documentation at: https://pcubillos.github.io/gen_tso
```

</details>

### Take the tour (docs):

[https://pcubillos.github.io/gen_tso/get_started.html](https://pcubillos.github.io/gen_tso/get_started.html)

### Cite as:

```bibtex
@ARTICLE{Cubillos2024paspGenTSO,
       author = {{Cubillos}, Patricio E.},
        title = "{Gen TSO: A General JWST Simulator for Exoplanet Times-series Observations}",
      journal = {\pasp},
     keywords = {Exoplanets, Time series analysis, Astronomy databases, 498, 1916, 83, Astrophysics - Earth and Planetary Astrophysics, Astrophysics - Instrumentation and Methods for Astrophysics},
         year = 2024,
        month = dec,
       volume = {136},
       number = {12},
          eid = {124501},
        pages = {124501},
          doi = {10.1088/1538-3873/ad8fd4},
archivePrefix = {arXiv},
       eprint = {2410.04856},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2024PASP..136l4501C},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

**And take a look:**

<img alt="Gen_TSO" src="https://github.com/pcubillos/gen_tso/blob/master/docs/gen_tso_screenshot.png">

