# Bayesian Uncertainty Quantification of the Infinite Nuclear Matter Equation of State

<img align="right" width="140" src="./logos/buqeye_logo_web.png">
This repository contains the data and Jupyter notebooks to produce the figures
in our publications:

* Drischler, Furnstahl, Melendez, and Phillips, _How well do we know the neutron-matter equation of state at the densities
inside neutron stars? A Bayesian approach with correlated uncertainties_, [arXiv:2004.07232](https://arxiv.org/abs/2004.07232).

* Drischler, Melendez, Furnstahl, and Phillips, _Effective
Field Theory Convergence Pattern of Infinite Nuclear Matter_, [arXiv:2004.07805](https://arxiv.org/abs/2004.07805).


## Overview

The directory `analysis` contains all the relevant Jupyter notebooks, including
the main notebooks `derivatives-bands.ipynb` and
`correlated_matter_analysis_refactored.ipynb`, which generate the figures in our
papers. The directories `nuclear_matter` and `other_figures` contain the raw
Python implementation, helper functions, etc. and additional figures not shown
in the papers (e.g., for talks). The raw data for the equation of state of
neutron matter and symmetric nuclear matter can be found in `data` and
`raw_data`. More information can be found in the README files as well as in the
annotated notebooks.


## Requirements and Installations

Installing and running our Jupyter notebooks is straightforward. `Python 3` is
required with the (standard) packages listed in `requirements.txt` installed.
They can be installed by running the command:
``` shell
pip3 install -r requirements.txt
```
In addition, J. Melendez's package `gsum`, which is publicly available
[here](https://github.com/buqeye/gsum) including installation instructions, needs to be installed
separately. Do not use `gsum` as installed by `pip3`.

With these prerequisites, to install this repository simply run (at the top
level):
```shell
pip3 install .
```

## Symmetry Energy and its Slope Parameter

BUQEYE's version of J. Lattimer's well-known `Sv--L plot`, Figure 2 of our
[arXiv:2004.07232](https://arxiv.org/abs/2004.07232), can be produced using the
Jupyter Notebook `analysis/Esym-L/Esym_L_correlation_plot.ipynb`. In addition to
a static `pdf` file, we support the export of an animated `gif`, which shows the
different empirical constraints incrementally. This is, in particular, useful
for scientific talks and teaching.

<p align="center">
  <img width="380" src="analysis/Esym-L/incremental_plots/Esym_L_correlation_animated.gif">
</p>


## Contact

To report any issues please use the issue tracker.


## Citing this Work and Further Reading

* Drischler, Furnstahl, Melendez, and Phillips, _How well do we know the neutron-matter equation of state at the densities
inside neutron stars? A Bayesian approach with correlated uncertainties_, [arXiv:2004.07232](https://arxiv.org/abs/2004.07232).

* Drischler, Melendez, Furnstahl, and Phillips, _Effective
Field Theory Convergence Pattern of Infinite Nuclear Matter_, [arXiv:2004.07805](https://arxiv.org/abs/2004.07805).



[buqeye]:https://buqeye.github.io/ "to the website of the BUQEYE collaboration"
[gsum]:https://github.com/buqeye/gsum "to the gsum's github repository"
[shortPaper]: https://buqeye.github.io/
[longPaper]: https://buqeye.github.io/
