# Raw data

This directory contains the data for the energy per particle in neutron matter
and symmetric nuclear matter as obtained from many-body perturbation theory.
See the following references for details:

* [Drischler _et al._, Phys. Rev. Lett. **122**, 042501 (2019)][DrischlerPRL]
* [Leonhardt _et al._, arXiv:1907.05814][Leonhardt]
* [Drischler _et al._, arXiv:2004.07232][DrischlerShort]

Specifically, `NN-only` contains the results for the NN-only potentials (but the
single-particle spectrum does include 3N contributions). The directory `NN+3N`
contains the data for NN and 3N potentials up to `n = 0.21 fm^{-3}`, which is
extended in `NN+3N_high_density` to `n = 0.34 fm^{-3}`.

## Filenames and column labels

The filenames encode the proton fraction and the label of the Hamiltonian as
follows:
```
EOS_x_["proton fraction"]_["Hamiltonian label"].txt
```
For instance, `EOS_x_0._Ham_2_NLO_EM450new.txt` corresponds to neutron matter
and the second 3N fit (Hamiltonian) below.

The columns in the files are given by:

| label       | unit    | description |
| -----       | ----    | ----------- |
| "kf"        | fm^{-1} | Fermi momentum     |
| "n"         | fm^{-3} | density      |
| "Kin"       | MeV     | kinetic energy      |
| "HF_tot"    | MeV     | Hartree-Fock energy      |
| "Scnd_tot"  | MeV     | Second-order energy       |
| "Trd_tot"   | MeV     | Third-order energy        |
| "Fth_tot"   | MeV     | Fourth-order energy       |
| "total"     | MeV     | sum of the previous energies      |



## Hamiltonians

For the 3N fits to the triton and nuclear matter see Figure 2 in the
Supplemental Material of [Drischler _et al._, Phys. Rev. Lett. **122**, 042501
(2019)][DrischlerPRL]. A machine-readable compilation of all relevant LECs and
cutoff values is provided in `DHS_hamiltonians_2017.par`. For  convenience, here
is a simplified version.

| \# | Chiral Order |	NN Potential |	cD  |  cE
:---:|:-------------|:-------------|-----:|-----:|
| 1  | N2LO         |	EMN 450 MeV  |	2.25|	0.07
| 2  | N2LO         |	EMN 450 MeV  |	2.50|	0.10
| 3  | N2LO         |	EMN 450 MeV  |	2.75|	0.13
| 4  | N2LO         |	EMN 500 MeV  |	-1.75|	-0.64
| 5  | N2LO         |	EMN 500 MeV  |	-1.50|	-0.61
| 6  | N2LO         |	EMN 500 MeV  |	-1.25|	-0.59
| 7  | N3LO         |	EMN 450 MeV  |	0.00|	-1.32
| 8  | N3LO         |	EMN 450 MeV  |	0.25|	-1.28
| 9  | N3LO         |	EMN 450 MeV  |	0.50|	-1.25
| 10 | N3LO         |	EMN 500 MeV  |	-3.00|	-2.22
| 11 | N3LO         |	EMN 500 MeV  |	-2.75|	-2.19
| 12 | N3LO         |	EMN 500 MeV  |	-2.50|	-2.15

[DrischlerPRL]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.122.042501
[Leonhardt]: https://arxiv.org/abs/1907.05814
[DrischlerShort]: https://arxiv.org/abs/2004.07232
