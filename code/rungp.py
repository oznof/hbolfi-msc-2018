import os
import argparse
import numpy as np
import GPy
from acquisition import LCBSC, LCBSCADD
from no_decomp import MHNO, GibbsNO, GPSampler
from gp_opt import BOLFI
from qpoly import QPoly, QuarticPoly, LogQPoly
import sobol_seq
import ghalton
import scipy.linalg as sl
import scipy.stats as ss
import scipy.integrate as integrate
import time
import matplotlib.pyplot as plt
from utils import minimize, gen_name
from utils import pred_noiseless, pred_grad, min_gp
from utils import normalize, dist, mcint, imp_samp, kl_normal, kl_normal_sym, timer, partition_metrics
from utils import sq_mmd, quasi_kl, bootstrap_test
from utils import pilotABC, rejectionABC, SMCABC, isampler
from sklearn.metrics import adjusted_rand_score
import pickle
import logging
import pandas as pd
import numdifftools as ndt

# ========================== AUXILIARY FUNCTIONS =======================================
def todict(**kw):
  return {k:v for k,v in kw.items()}


class TrueModel():
  def __init__(self, input_dim, fm, fv, gm, gv, lik_noise_var,
               xmin=None, seed=None, threshold=None):
    self.input_dim = input_dim
    self.fm = fm # mean function(x)
    self.fv = fv # var function(x) without observation noise
    self.gm = gm # gradient mean function(x)
    self.gv = gv # gradient var function(x)
    self.lik_nv = lik_noise_var # observation variance
    self.random_state = np.random if seed is None else np.random.RandomState(seed)
    self.xmin = xmin
    self.fmin = fm(xmin) if xmin is not None else None
    self.threshold = self.fmin if threshold is None else threshold

  def predict(self, x):
    return map(np.atleast_2d, [self.fm(x), self.fv(x) + self.lik_nv])

  def predict_noiseless(self, x, **kws):
    return self.fm(x, **kws), self.fv(x, **kws)

  def predictive_gradients(self, x):
    return self.gm(x), self.gv(x)

  def true_f(self, x):
    x = x.squeeze()
    return self.fm(x)

  def noisy_f(self, x):
    x = x.squeeze()
    f = np.atleast_1d(self.fm(x))
    fv = np.atleast_1d(self.fv(x))
    fnoise = f + np.sqrt(fv + self.lik_nv) * self.random_state.randn(*f.shape)
    idx = np.where(fnoise < self.fmin)[0]
    while len(idx) > 0 and np.any(fnoise[idx] < self.fmin):
      fnoise[idx] = f[idx] + np.sqrt(fv[idx] + self.lik_nv) * self.random_state.randn(*fnoise[idx].shape)
      idx = idx[np.where(fnoise[idx] < self.fmin)[0]]
    return fnoise.squeeze()

def todict(**kw):
  return {k:v for k,v in kw.items()}


def rabc(true_model, bounds, npost_samples, random_state, exp_path, exp_name, nn,  xs, gsampler_true, percentile_thre=1):
    path_store = "{}/s_{}_{}.csv".format(exp_path, exp_name, nn)
    postpred_true = true_model.noisy_f(xs)
    n_sample = int(npost_samples / percentile_thre * 100)
    DF = pd.DataFrame(index=np.arange(1), 
                     columns=["kl_n", "sqmmd","ppc_mean", "ppc_var", "ppc_q25", "ppc_median", "ppc_q75",
                              "ppc_kurtosis", "ppc_skewness", "ppc_min", "ppc_max"])
    SAMPLES = pd.DataFrame(index=np.arange(2), columns=np.arange(npost_samples))
    SAMPLES.loc[0, :] = postpred_true
    prior_f = lambda n=1: np.hstack([random_state.uniform(*b, n)[:, None] for b in bounds]).squeeze()
    ys, fys = rejectionABC(simulator=true_model.noisy_f, threshold=None, sampler=prior_f, n_sample=n_sample, verbose=True)[:2]
    threshold_aprx = np.percentile(fys, percentile_thre)
    ys = ys[np.where(fys <= threshold_aprx)[0]][:len(xs)]
    postpred_aprx = true_model.noisy_f(ys)
    SAMPLES.loc[1, :] = postpred_aprx
    SAMPLES.to_csv("{}/s_{}_{}.csv".format(exp_path, exp_name, nn))
    i = 0
    DF.loc[i, "sqmmd"] = sq_mmd(xs, ys)
    DF.loc[i, "ppc_mean"] = np.abs(np.mean(postpred_aprx) - np.mean(postpred_true)) 
    DF.loc[i, "ppc_var"] = np.abs(np.var(postpred_aprx, ddof=1) - np.var(postpred_true, ddof=1))
    DF.loc[i, "ppc_q25"] = np.abs(np.percentile(postpred_aprx, 25) - np.percentile(postpred_true, 25))
    DF.loc[i, "ppc_median"] = np.abs(np.median(postpred_aprx) - np.median(postpred_true))
    DF.loc[i, "ppc_q75"] = np.abs(np.percentile(postpred_aprx, 75) - np.percentile(postpred_true, 75))
    DF.loc[i, "ppc_kurtosis"] = np.abs(ss.kurtosis(postpred_aprx, bias=False) - ss.kurtosis(postpred_true, bias=False))
    DF.loc[i, "ppc_skewness"] = np.abs(ss.skew(postpred_aprx, bias=False) - ss.skew(postpred_true, bias=False))
    DF.loc[i, "ppc_min"] = np.abs(np.min(postpred_aprx) - np.min(postpred_true))
    DF.loc[i, "ppc_max"] = np.abs(np.max(postpred_aprx) - np.max(postpred_true))
    # Try laplace approximation to the kde estimate (might yield completely inaccurate results in high dimensions)
    kde_aprx = ss.gaussian_kde(ys.T)
    nll_kde = lambda x: -np.log(kde_aprx(x) + 1e-20)
    xstar = minimize(fun=nll_kde, bounds=bounds, grad=None, guess_point=true_model.xmin)[0]
    H = ndt.Hessian(nll_kde)(xstar)
    H[np.diag_indices(len(H))] += 1e-8
    try:
        C = sl.cho_solve(sl.cho_factor(H), np.eye(xstar.size))
        DF.loc[i, "kl_n"] = kl_normal(gsampler_true.mean, gsampler_true.cov, xstar, C)
    except:
        pass
    DF.to_csv("{}/r_{}_{}.csv".format(exp_path, exp_name, nn))


# ========================== PARSE ARGUMENTS =======================================

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Discrepancy function as proxy simulation model (2D+)")
parser.add_argument('--learn_type', type=str, nargs='?', default="full", help='learn kernel')
parser.add_argument('--update_k', type=int, nargs='?', default=10, help="update kernel every k iterations")
parser.add_argument('--iters_k', type=int, nargs='?', default=50, help="number of kernel learning iterations")
parser.add_argument('--pairs', type=int, nargs='?', default=None, help="dimension of pairs problem.")
parser.add_argument('--quads', type=int, nargs='?', default=None, help="dimension of quads problem.")
parser.add_argument('--halve', type=int, nargs='?', default=None, help="dimension of halve problem.")
parser.add_argument('--exp_path', type=str, nargs='?', default="new_acq_e1/", help="Results dir.")
parser.add_argument('--max_iters', type=int, nargs='?', default=1000, help="Maximum number of optimization iterations.")
parser.add_argument('--additive', type=int, nargs='?', default=1, help="Maximum number of optimization iterations.")
parser.add_argument('--acq_noise', type=float, nargs='?', default=0.1, help="Maximum number of optimization iterations.")
parser.add_argument('--nn_init', type=int, nargs='?', default=0, help="Initial run.")
parser.add_argument('--nn_end', type=int, nargs='?', default=10, help="Final run.")
parser.add_argument('--corr', type=float, nargs='?', default=0.6, help="Correlation between variables in same block")
parser.add_argument('--mu', type=float, nargs='?', default=0.5, help="Minimum")

args = parser.parse_args()
learn_type = args.learn_type
update_k = args.update_k
iters_k = args.iters_k
pairs = args.pairs
quads = args.quads
halve = args.halve
max_opt_iters = args.max_iters
additive = bool(args.additive)
acq_noise = args.acq_noise
nn_init = args.nn_init
nn_end = args.nn_end
corr = args.corr
mu_min = args.mu

seed = 117
nacqs = 500
n_res_update = 10
n_res_update2 = 50 # must be multiple of n_res_update
lik_noise_var = 1e-2
true_lengthscale = 3
true_var = 10
##true_n_obs = 1200
npost_samples = 5000
n_top_test_pts = 500

num_dims = list(filter(None, [pairs, quads, halve]))[0]
if pairs is not None:
  problem = "pairs"
  group_dims = 2
elif quads is not None:
  group_dims = 4
  problem = "quads"
elif halve is not None:
  problem = "halve"
  group_dims = num_dims // 2
else:
  raise ValueError("Use either pairs, quads, halve.")
num_groups = num_dims // group_dims
true_z = np.array([np.repeat(_, group_dims) for _ in range(num_groups)]).flatten().tolist()
true_z_str = str(true_z)

mins = np.full(num_dims, -3)
maxes = np.full(num_dims, 3)
bounds = np.vstack((mins, maxes)).T
true_kernel = []
true_groups = []
zuniq = np.unique(true_z)
for zz in zuniq:
    pdd = np.where(true_z == zz)[0]
    lenpd = len(pdd)
    true_groups.append(pdd)
    kernel_pdd = GPy.kern.RBF(input_dim=len(pdd), active_dims=pdd, ARD=False, name=gen_name())
    kernel_pdd.lengthscale.constrain_fixed(true_lengthscale)
    kernel_pdd.variance.constrain_fixed(true_var)
    true_kernel.append(kernel_pdd)
true_kernel = GPy.kern.Add(true_kernel)

# ========================== EXPERIMENT DESCRIPTION =======================================
exp_name = "_".join(map(str, [len(true_z), learn_type, update_k, iters_k,
                              max_opt_iters, additive, acq_noise, problem]))
exp_path = os.path.join(args.exp_path, exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)
print("Experiment: ", exp_name)
print("Save path: ", exp_path)
print("Testing the following structure: ", true_z_str)
if learn_type == "rabc":
    kws = {"kernel_update":False}
    msg = "no (Rejection-ABC)"
elif learn_type == "full":
    kws = {"kernel_update":False}
    msg = "no (assume standard)"
elif learn_type == "fullscaled":
    kws = {"kernel_update":False}
    msg = "no (assume standard, scaled)"  
elif learn_type == "oracle":
    kws = {"kernel_update":False}
    msg = "oracle"
elif learn_type == "indep":
    kws = {"kernel_update":False}
    msg = "no (assume full independence)"
elif  learn_type == "indepscaled":
    kws = {"kernel_update":False}
    msg = "no (assume full independence, scaled)"  
elif learn_type == "mhno":
    kws = {"kernel_update":True, "kernel_sampler":MHNO, "ARD":False}
    msg = "yes (Metropolis Hastings). update_k: {}. iters_k: {}".format(update_k, iters_k)
elif learn_type == "gibbsno":
    kws = {"kernel_update":True, "kernel_sampler":GibbsNO,"ARD":False}
    msg = "yes (Gibbs). update_k: {}. iters_k: {}".format(update_k, iters_k)
else:
  raise ValueError("Invalid kernel learning approach.")
print("Learn kernel: {}".format(msg))

# ========================== SETUP INITIAL KERNEL =======================================
if learn_type == "oracle":
  initial_kernel = true_kernel.copy().parts
elif learn_type != "indep" and  learn_type != "indepscaled":
  initial_kernel = [GPy.kern.RBF(input_dim=num_dims, active_dims=range(num_dims), ARD=False)]
  initial_kernel[0].lengthscale.constrain_fixed(true_lengthscale)
  if learn_type != "fullscaled":
    initial_kernel[0].variance.constrain_fixed(true_var)
  else: 
    initial_kernel[0].variance.constrain_fixed(true_var * num_groups)
else:
  initial_kernel = []
  for dd in range(num_dims):
    kk = GPy.kern.RBF(input_dim=1, active_dims=dd, ARD=False, name=gen_name())
    kk.lengthscale.constrain_fixed(true_lengthscale)
    if learn_type != "indepscaled":
      kk.variance.constrain_fixed(true_var)
    else:
      kk.variance.constrain_fixed(true_var / group_dims)
    initial_kernel.append(kk)
initial_kernel = GPy.kern.Add(initial_kernel)

# ========================== DEFINE OBSERVATIONS / MEAN FUNCTION =======================================
Corr = np.full((group_dims, group_dims), corr)
Corr[np.diag_indices(group_dims)] = 1
C1 = np.linalg.inv(Corr)
mean_a_ws = 0.5 * C1[0, 0]

# 0.5 * (num_dims * np.log(2*np.pi) - len(groups) * np.log(1 - 0.6**2))
mean_tt = mu_min * np.ones(num_dims)
def f_obs(X, groups):
  X = np.atleast_2d(X)
  y = np.zeros(len(X))
  for group in groups:
    ngroup = len(group)
    x = X[:, group]
    mean = mean_tt[group]
    y += np.array([0.5 * ((xx - mean).dot(C1).dot(xx - mean) - C1[0, 0] * np.sum(xx**2)) for xx in x])
  return y

mean_function = None
if mean_a_ws is not None:
  mean_function = QPoly(num_dims, 1, seed=seed, set_zero=True)
  mean_function.A.constrain_fixed(mean_a_ws*np.ones(num_dims)[:,None])
  mean_function.B.constrain_fixed(np.zeros(num_dims)[:,None])
  mean_function.C.constrain_fixed(0)


# ======================================================================================================
# ======================================== START RUNS ==================================================
nn_step = -1 if nn_init > nn_end else 1
nruns = np.abs(nn_end - nn_init)
gtimer = timer(nruns * nacqs)
for nn in range(nn_init, nn_end, nn_step):
  print("="*30)
  print("Starting run {} of {}".format(nn + 1, nn_end if nn_step == 1 else nn_init+1))
  seed_nn = seed + nn 
  RandomState = np.random.RandomState(seed_nn)
  np.random.seed(seed_nn)

  # Store info
  DF = pd.DataFrame(index=np.arange(0, nacqs+1, n_res_update), 
                     columns=["regret", "cumregret", "expected_fmin", "nearb", "l1nx", "linfx",
                              "loglik", "rand_idx", "tpr", "tnr", "prec", "f1_score",
                              "ari", "kl_n", "logbf", "logrbf", "kl_n2", "kl_sym_n2",
                              "dist_m2l2", "dist_m2l1n", "dist_c2l2", "dist_c21n", 
                              "ess_ref", "psrf_ref", "sqmmd_ref", "ess", "psrf", "sqmmd",
                              "ppc_mean", "ppc_var", "ppc_q25", "ppc_median", "ppc_q75",
                              "ppc_kurtosis", "ppc_skewness", "ppc_min", "ppc_max"])
  SAMPLES = pd.DataFrame(index=np.arange(0, nacqs+1, n_res_update), columns=np.arange(npost_samples)) # index 0 will be the reference
  # Other meta info to store
  kernel_times = []
  acq_times = []
  partitions = []
  alphas = []
  resmeta = todict(n_res_update=n_res_update,
                   true_z=true_z, learn_type=learn_type, update_k=update_k, iters_k=iters_k,
                   kernel_times=kernel_times, acq_times=acq_times, partitions=partitions,
                   alphas=alphas)
  # ================================================================================================
  # DEFINE TRUE MODEL
  mean_f_copy = mean_function.copy() if mean_function is not None else None
  num_s = 300
  xmin = np.full(num_dims, np.inf)
  i = 0
  while not np.all(np.logical_and(bounds[0][0]+0.2 <= xmin, xmin <= bounds[0][1] - 0.2)):
    print("Sampling function: {}".format(i+1))
    radii = [0.01, 0.1, 0.5, 1] 
    X0 = np.zeros((1 + len(radii) * num_s, num_dims))
    X0[0] = mean_tt
    for ii, rr in enumerate(radii):
      Xaux = np.random.randn(num_s, num_dims)
      Xaux = rr * Xaux / np.linalg.norm(Xaux, axis=1)[:,None] - mean_tt
      X0[1+ii*num_s:1+(ii+1)*num_s] = Xaux
      y0 = f_obs(X0, true_groups)
    X_true = X0
    y_true_no_mean = y0
    y_true = y_true_no_mean
    if mean_function is not None:
      y_true += mean_function.f(X_true).flatten()
    #y_obs = y_true + np.sqrt(lik_noise_var) * RandomState.randn(len(y_true))
    true_f = GPy.models.GPRegression(X_true, y_true[:,None], true_kernel,
                 noise_var=lik_noise_var, mean_function=mean_f_copy)

    st = time.time()
    xmin_full = min_gp(true_f, additive=False, bounds=bounds, seed=seed_nn,
                       guess_point=np.zeros(num_dims))
    time_full = time.time() - st
    st = time.time()
    xmin_add = min_gp(true_f, additive=True, bounds=bounds, seed=seed_nn,
                      guess_point=np.zeros(num_dims))
    time_add = time.time() - st
    st = time.time()
    xmin = xmin_add
    i += 1
  print("Optimization Results:")
  for t, x in zip([time_full, time_add], [xmin_full, xmin_add]):
      print("time: {}".format(t))
      print("xmin: {}".format(x))
      print("fmin: {}".format(pred_noiseless(x, true_f)[0]))
      print("fmin0: {}".format(pred_noiseless(np.zeros(num_dims), true_f)[0]))
      print()
  true_model = TrueModel(num_dims, fm=lambda x, **kw: true_f.predict_noiseless(x.reshape(-1, num_dims), **kw)[0].ravel(),
                           fv=lambda x, **kw: np.zeros(len(x.reshape(-1, num_dims))),
                           gm=lambda x: true_f.predictive_gradients(x.reshape(-1, num_dims))[0],
                           gv=lambda x, **kw: np.zeros((1, x.size)),
                           lik_noise_var=lik_noise_var, xmin=xmin,
                           seed=seed_nn)
  # ================================================================================================
  # ===================================== SETUP BOLFI / REFERENCE METRICS ==========================
  bolfi = BOLFI(f=true_model.noisy_f, kernel_list=initial_kernel.copy().parts, bounds=bounds, acq_f=LCBSCADD, prior_f=None,
                initial_n=20, mean_function=mean_function.copy() if mean_function is not None else None,
                lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=true_lengthscale,
                variance=true_var, optimizer=None, messages=False, max_iters=max_opt_iters,
                logging=False, seed=seed_nn, n_update=update_k,
                acq_f_kws={"n_inits": max(10, int(num_dims)), "noise_var":acq_noise}, xopt=true_model.xmin,
                f_true=true_model.true_f, additive=additive, **kws)

  oracle = BOLFI(f=true_model.noisy_f, kernel_list=true_kernel.copy().parts, bounds=bounds,
                initial_n=1, mean_function=mean_function.copy() if mean_function is not None else None,
                lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=true_lengthscale,
                variance=true_var, kernel_update=False)

  # Set "true" loglikelihood (unnormalized)
  loglik_true_un = lambda x: bolfi.unnormalized_loglikelihood(true_model.threshold, x, true_model, bounds)
  # Laplace approximation of the "true" loglikelihood function
  gsampler_true = bolfi.laplace_aprx(true_model.threshold, true_model, bounds, seed=seed_nn,
                                     guess_point=true_model.xmin)
  # Sampling 1: Use NUTS | MH
  logger.setLevel(level=logging.INFO)
  st_sample = time.time()
  xs, effx, psrfx = bolfi.mcmc(true_model.threshold, n_iter=npost_samples, model=true_model, bounds=bounds,
                              params0=gsampler_true.mean, 
                              seed=seed_nn, get_stats=True, n_chains=1, target_prob=0.7)
  DF.loc[:, "ess_ref"]  = np.mean(effx)
  DF.loc[:, "psrf_ref"] = np.mean(psrfx)
  if learn_type == "rabc":
    rabc(true_model, bounds, npost_samples, RandomState, exp_path, exp_name, nn, xs, gsampler_true)
    continue
  ##st_sample = time.time()
  ##xs1 = bolfi.mcmc(true_model.threshold, n_iter=npost_samples, model=true_model, bounds=bounds,
  ##                 params0=gsampler_true.mean, seed=seed_nn+1, get_stats=False,
  ##                 n_chains=1, target_prob=0.7)
  ##print("time sampling true (2): {0:.3f}".format(time.time() - st_sample))
  logger.setLevel(level=logging.CRITICAL)
  xxs2 = np.unique(xs, axis=0)
  xs_test_locs = xxs2[np.argsort(loglik_true_un(xxs2))[::-1][:n_top_test_pts]]
  mean_true, cov_true = true_f.predict(xs_test_locs, full_cov=True)
  cov_true[np.diag_indices(len(cov_true))] += 1e-8
  cov_true = np.round(cov_true, 5)
  postpred_true = true_model.noisy_f(xs)
  SAMPLES.loc[0, :] = postpred_true
  ##DF.loc[:, "sqmmd_ref"] = sq_mmd(xs, xs1)

  # ==================================== START ACQUISITIONS =================================================
  for i in range(0, nacqs+1):
      print("="*20)
      print("total: {}, eta: {}".format(*gtimer.update()))
      print("{}: {} - {}.\n".format(nn+1, i, bolfi.curr_model.log_likelihood()))
      if i > 0:
        bolfi.update_next_point(n=1, kernel_n_iters=iters_k)
      if i % n_res_update == 0:
        st = time.time()
        # Store log-likelihood
        DF.loc[i, "loglik"] = bolfi.curr_model.log_likelihood()
        # Store BF
        oracle.curr_model.set_XY(bolfi.curr_model.X, bolfi.curr_model.Y)
        DF.loc[i, "logbf"] = oracle.curr_model.log_likelihood() - bolfi.curr_model.log_likelihood()
        DF.loc[i, "logrbf"] = oracle.curr_model.log_likelihood() / bolfi.curr_model.log_likelihood()
        # Store regret
        DF.loc[i, ["regret", "cumregret", "expected_fmin"]] = bolfi.regret, bolfi.cumregret, bolfi.expected_fcurr
        # Store info x
        DF.loc[i, "nearb"] = np.any(np.logical_or(bolfi.xcurr >= bounds[:,1] -0.1, bolfi.xcurr <= bounds[:,0]+0.1))
        DF.loc[i, "l1nx"]  = np.sum(np.abs(true_model.xmin - bolfi.xcurr)) / num_dims
        DF.loc[i, "linfx"] = np.max(np.abs(true_model.xmin - bolfi.xcurr))
        # Store partition information
        z = np.empty(num_dims)
        for v, k in enumerate(bolfi.curr_model.kern.parts):
          z[k.active_dims] = v
        partitions.append(z)
        pmetrics = partition_metrics(true_z, z)
        DF.loc[i, ["rand_idx", "tpr", "tnr", "prec", "f1_score"]] = pmetrics
        DF.loc[i, "ari"] = adjusted_rand_score(true_z, z)
        gsampler_aprx = None
        if not DF.loc[i, "nearb"]:
          threshold_aprx = bolfi.expected_fcurr
          #print("threshold_aprx: ", threshold_aprx)
          # Setup Laplace approximation
          gsampler_aprx = bolfi.laplace_aprx(threshold_aprx, seed=seed_nn,
                                             guess_point=bolfi.xcurr)
          if gsampler_aprx is not None:
            DF.loc[i, "kl_n"] = kl_normal(gsampler_true.mean, gsampler_true.cov,
                                          gsampler_aprx.mean, gsampler_aprx.cov)
            mean_aprx, cov_aprx = bolfi.curr_model.predict(xs_test_locs, full_cov=True)
            cov_aprx[np.diag_indices(len(cov_aprx))] += 1e-8
            cov_aprx = np.round(cov_aprx, 5)
            DF.loc[i, "kl_n2"] = kl_normal(mean_true, cov_true, mean_aprx, cov_aprx)
            DF.loc[i, "kl_sym_n2"] = kl_normal_sym(mean_true, cov_true, mean_aprx, cov_aprx)
            DF.loc[i, "dist_m2l2"] = np.linalg.norm(mean_true - mean_aprx)
            DF.loc[i, "dist_m2l1n"] = np.sum(np.abs(mean_true - mean_aprx)) / mean_true.size
            DF.loc[i, "dist_c2l2"] = np.linalg.norm(cov_true - cov_aprx) # frobenius
            DF.loc[i, "dist_c21n"] = np.sum(np.abs(cov_true - cov_aprx)) / cov_true.size
            if (i % n_res_update2 == 0) and i > 0:
              # Sampling 2: Use NUTS | MH
              logger.setLevel(level=logging.INFO)
              st_sample = time.time()
              ys, effx, psrfx = bolfi.mcmc(threshold_aprx, n_iter=npost_samples, params0=gsampler_aprx.mean, 
                                           seed=seed_nn, get_stats=True, n_chains=1, target_prob=0.7)
              print("time sampling aprx: {0:.3f}".format(time.time() - st_sample))
              logger.setLevel(level=logging.CRITICAL)
              DF.loc[i, ["ess", "psrf", "sqmmd"]] = np.mean(effx), np.mean(psrfx), sq_mmd(xs, ys)
              postpred_aprx = true_model.noisy_f(ys)
              SAMPLES.loc[i, :] = postpred_aprx
              DF.loc[i, "ppc_mean"] = np.abs(np.mean(postpred_aprx) - np.mean(postpred_true)) 
              DF.loc[i, "ppc_var"] = np.abs(np.var(postpred_aprx, ddof=1) - np.var(postpred_true, ddof=1))
              DF.loc[i, "ppc_q25"] = np.abs(np.percentile(postpred_aprx, 25) - np.percentile(postpred_true, 25))
              DF.loc[i, "ppc_median"] = np.abs(np.median(postpred_aprx) - np.median(postpred_true))
              DF.loc[i, "ppc_q75"] = np.abs(np.percentile(postpred_aprx, 75) - np.percentile(postpred_true, 75))
              DF.loc[i, "ppc_kurtosis"] = np.abs(ss.kurtosis(postpred_aprx, bias=False) - ss.kurtosis(postpred_true, bias=False))
              DF.loc[i, "ppc_skewness"] = np.abs(ss.skew(postpred_aprx, bias=False) - ss.skew(postpred_true, bias=False))
              DF.loc[i, "ppc_min"] = np.abs(np.min(postpred_aprx) - np.min(postpred_true))
              DF.loc[i, "ppc_max"] = np.abs(np.max(postpred_aprx) - np.max(postpred_true))
        print("elapsed time: {0:.3f}".format(time.time() - st))
        print("learn_type: ", learn_type)
        print("target partition: {}".format(true_z))
        print("current partition: {}".format(partitions[-1]))
        print("xbolfi: ", bolfi.xcurr)
        print(DF.loc[i])

        if len(bolfi.kernel_times) > 0:
          print("kernel times: {0:.3f}".format(np.mean(bolfi.kernel_times)))
        if len(bolfi.acq_times) > 0:
          print("acq times: {0:.3f}".format(np.mean(bolfi.acq_times)))
      if i == nacqs:
        kernel_times.append(bolfi.kernel_times)
        acq_times.append(bolfi.acq_times)
        if learn_type == "mhno":
          alphas.append(bolfi.kernel_sampler.alphas)
      if  i % n_res_update2 == 0 or i == nacqs:    
        DF.to_csv("{}/r_{}_{}.csv".format(exp_path, exp_name, nn))
        SAMPLES.to_csv("{}/s_{}_{}.csv".format(exp_path, exp_name, nn))
        with open("{}/m_{}_{}.obj".format(exp_path, exp_name, nn), "wb") as fp:
          pickle.dump(resmeta, fp)

#============================================================================================
#============================================================================================
#============================================================================================
#============================================================================================












        # print("{}".format(i))
        # print("logbf: {0:.3f}, logrbf: {1:.3f}".format(
        #       logbf[nn, ii], logrbf[nn, ii]))
        # print("oracle: ", oracle.curr_model.log_likelihood())
        # print("model: ", bolfi.curr_model.log_likelihood())
        # print("=================")
        # continue



          # fig, ax2 = plt.subplots(1, 1, figsize=(12 , 6))
          # x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
          # X = np.vstack((x.flatten(), y.flatten())).T
          # #f1 = bolfi.curr_model.predict_noiseless(X)[0].reshape(x.shape)
          # #cs = ax.contour(x, y, f1, 50,
          # #               linewidths=0.5, colors='k', linestyles="dashed")
          # #ax1.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
          # #ax1.contourf(x, y, f1, 50, cmap="RdBu_r")
          # #for xxx in bolfi.X:
          # #  ax1.plot(xxx[0], xxx[1], "ko")
          # #ax1.plot(bolfi.xcurr[0], bolfi.xcurr[1], "ro")
          # #plt.show()
          # #ff, Z = normalize(lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold_aprx, x)), bounds, return_Z=True)
          # #print(bolfi.curr_model)
          # from mpl_toolkits.mplot3d import Axes3D
          # ax2 = fig.gca(projection='3d')
          # f1 = bolfi.curr_model.predict_noiseless(X)[0]
          # f1 = np.exp(loglik_true_un(X)).reshape(x.shape) #/ 0.08124061634136047
          # ax2.plot_trisurf(X[:,0], X[:,1], f1.ravel())

          # # np.exp(bolfi.unnormalized_loglikelihood(threshold_aprx, X)
          # #cs = ax2.contour(x, y, f1, 50,
          # #                 linewidths=0.5, colors='k', linestyles="dashed")
          # #ax2.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
          # #ax2.contourf(x, y, f1, 50, cmap="RdBu_r")
          # #for xxx in bolfi.X:
          # #  ax2.plot(xxx[0], xxx[1], "ko")
          # #ax2.plot(bolfi.xcurr[0], bolfi.xcurr[1], f1.ravel()[0],  "ro")
          # plt.show()
          # # np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14
          # #asdasd




              # sequencer = ghalton.Halton(num_dims)
    # X1 = np.array(sequencer.get(true_n_obs - len(X0)))
    # X1 = (maxes - mins)[None,:] * X1 + mins[None,:]
    # K_10 = true_kernel.K(X1, X0)
    # K_00 = true_kernel.K(X0)
    # K_00[np.diag_indices(len(K_00))] += 1e-8
    # cho_00 = sl.cho_factor(K_00)
    # mean_cond = K_10.dot(sl.cho_solve(cho_00, y0[:,None])).ravel()
    # cov_cond = true_kernel.K(X1) - K_10.dot(sl.cho_solve(cho_00, K_10.T))
    # y1 = ss.multivariate_normal.rvs(mean=mean_cond, cov=cov_cond,
    #                                 size=1, random_state=RandomState)
    # X_true = np.vstack((X0, X1))
    # y_true_no_mean = np.hstack((y0, y1.ravel()))

  
  # f, ax = plt.subplots()
  # x, y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-3, 3, 100))
  # X = np.vstack((x.flatten(), y.flatten())).T
  # f1 = true_model.fm(X).reshape(x.shape)
  # cs = ax.contour(x, y, f1, np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14,
  #                  linewidths=0.5, colors='k', linestyles="dashed")
  # ax.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
  # ax.contourf(x, y, f1, np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14, cmap="RdBu_r")  
  # plt.show()
