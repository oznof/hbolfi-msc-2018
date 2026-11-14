# High-Dimensional Bayesian Optimization for Likelihood-Free Inference (MSc Thesis Code, 2018)

**The code of the MSc thesis *High-dimensional Bayesian Optimization for Learning in Generative Models*** (Afonso Eduardo, MSc in Artificial Intelligence, University of Edinburgh, 2018, supervised by Michael U. Gutmann), kept as it was submitted in August 2018.

- **Author:** Afonso Eduardo, [ORCID 0009-0003-0127-8223](https://orcid.org/0009-0003-0127-8223).
- **Thesis:** [doi:10.5281/zenodo.24226042](https://doi.org/10.5281/zenodo.24226042) (Zenodo).
- **Status:** research code from 2018, kept as it was, except for bug fixes in `sl.py` (see the table). It is not maintained, and it was not rerun when it was published here.
- **Follow-up:** [Bayesian Optimization with Informative Covariance](https://arxiv.org/abs/2208.02704) (Eduardo and Gutmann, *Transactions on Machine Learning Research*, 2023; code: [BOIC](https://github.com/oznof/BOIC)). The thesis found that, once the hyperparameters were learned, the search could drift to the boundary of the search space, a failure of the prior. Informative covariance functions address it.
- **Later reimplementation:** [HBOLFI](https://github.com/oznof/HBOLFI) (2019), the same methodology rewritten with PyTorch, GPyTorch, and Pyro at the start of the doctoral research.

## Contents

The thesis extends Bayesian optimization for likelihood-free inference (BOLFI; [Gutmann and Corander, 2016](https://jmlr.org/papers/v17/15-017.html)) to higher dimensions by learning an additive structure of the discrepancy with Markov chain Monte Carlo.

| File | Role | Content |
|---|---|---|
| `gp_opt.py` | Library | BOLFI and the Bayesian optimization loop |
| `acquisition.py` | Library | Acquisition functions, including LCBSC and its additive version LCBSCADD |
| `no_decomp.py` | Library | Structure learning of additive Gaussian processes: Metropolis-Hastings and Gibbs samplers |
| `mcmc.py` | Library | MCMC samplers, including the No-U-Turn Sampler (Hoffman and Gelman, 2014, Algorithm 6) |
| `priors.py` | Library | Priors of the Gaussian process hyperparameters |
| `qpoly.py` | Library | Quadratic mean function |
| `utils.py` | Library | Optimization, prediction, performance measures (maximum mean discrepancy, closed-form KL divergence between Laplace approximations), and ABC baselines |
| `rungp.py` | Thesis experiments | Semiparametric model (2x2 and 4x4): a Gaussian process sample plus a quadratic mean, with a known posterior |
| `runquad.py` | Thesis experiment | Parametric model (2x2): a quadratic form with a known posterior |
| `1run_gp.py` | Demonstration | One-dimensional BOLFI run; saves one figure per acquisition, used as the frames of an animation |
| `styb.py` | Draft (July 2018) | Experiment on a rotated Styblinski-Tang function, mostly commented out; not among the thesis experiments |
| `sl.py` | Draft (July 2018) | Toy synthetic likelihood simulator: Gaussian summary statistics with a parameter-dependent covariance, pairs of parameters only; unfinished and not among the thesis experiments. Fixed in 2026: the log-determinant (sum of logs, not log of a sum), the covariance in `noisy_f` (per point, not the whole batch), and the constructor arguments, which were ignored |

The benchmarks are simulators defined in code, so no data files are needed.

## Requirements

Python 3, with GPy, paramz, NumPy, SciPy, pandas, scikit-learn, SymPy, joblib, numdifftools, scipydirect, ghalton, sobol_seq, and Matplotlib. The versions used in 2018 were not recorded. GPy releases from 2018 are the most likely to work.

## License

MIT License. See [LICENSE](LICENSE).
