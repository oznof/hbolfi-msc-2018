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
from utils import normalize, dist, mcint, imp_samp, kl_normal, timer, partition_metrics
from utils import sq_mmd, quasi_kl, bootstrap_test
from utils import pilotABC, rejectionABC, SMCABC, isampler
from sklearn.metrics import adjusted_rand_score
import pickle
import logging
import pandas as pd

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Discrepancy function as proxy simulation model (2D+)")
parser.add_argument('--learn_type', type=str, nargs='?', default="full", help='learn kernel')
parser.add_argument('--update_k', type=int, nargs='?', default=10, help="update kernel every k iterations")
parser.add_argument('--iters_k', type=int, nargs='?', default=50, help="number of kernel learning iterations")
parser.add_argument('--pairs', type=int, nargs='?', default=None, help="dimension of pairs problem.")
parser.add_argument('--quads', type=int, nargs='?', default=None, help="dimension of quads problem.")
parser.add_argument('--halve', type=int, nargs='?', default=None, help="dimension of halve problem.")
parser.add_argument('--exp_path', type=str, nargs='?', default="styb/", help="Results dir.")
parser.add_argument('--max_iters', type=int, nargs='?', default=1000, help="Maximum number of optimization iterations.")
parser.add_argument('--additive', type=int, nargs='?', default=1, help="Maximum number of optimization iterations.")
parser.add_argument('--acq_noise', type=float, nargs='?', default=0.01, help="Maximum number of optimization iterations.")
parser.add_argument('--nn_init', type=int, nargs='?', default=0, help="Initial run.")
parser.add_argument('--nn_end', type=int, nargs='?', default=5, help="Final run.")


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

nn_end = 1

seed = 117
nacqs = 500
lik_noise_var = 1e-2
n_res_update = 10
n_res_update2 = 10
npost_samples = 1000


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

if learn_type == "full":
    kws = {"kernel_update":False}
    msg = "no (assume full dependence)"
elif learn_type == "oracle":
    kws = {"kernel_update":False}
    msg = "oracle"
elif learn_type == "indep":
    kws = {"kernel_update":False}
    msg = "no (assume full independence)"
elif learn_type == "mhno":
    kws = {"kernel_update":True, "kernel_sampler":MHNO, "ARD":False}
    msg = "yes (Metropolis Hastings). update_k: {}. iters_k: {}".format(update_k, iters_k)
elif learn_type == "gibbsno":
    kws = {"kernel_update":True, "kernel_sampler":GibbsNO,"ARD":False}
    msg = "yes (Gibbs). update_k: {}. iters_k: {}".format(update_k, iters_k)
else:
  raise ValueError("Invalid kernel learning approach.")

def todict(**kw):
  return {k:v for k,v in kw.items()}

exp_name = "_".join(map(str, [len(true_z), learn_type, update_k, iters_k,
                              max_opt_iters, additive, acq_noise, problem]))
exp_path = os.path.join(args.exp_path, exp_name)
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

print("Experiment: ", exp_name)
print("Save path: ", exp_path)
print("Testing the following structure: ", true_z_str)
print("Learn kernel: {}".format(msg))

class Stybtang():
  def __init__(self, Q):
    self.Q = Q
    self.Qt = Q.T 
    self.gopt = -2.90353403 / 6 * np.ones(Q.shape[1])
    self.xmin = self.Qt.dot(self.gopt)
    self.fmin = 0
    self.fmin = self.rot_f(self.xmin) + 14

  def f(self, x):
    x = np.atleast_2d(x)
    return (0.5 * np.sum((6 * x)**4 - 16 * (6 * x)**2 + 5*(6 * x), axis=1)).squeeze()

  def rot_f(self, x):
    return -self.fmin + self.f(np.einsum("ij,kj->ki", self.Q, np.atleast_2d(x)))

  def grad_rot_f(self, x):
    x = np.einsum("ij,kj->ki", self.Q, np.atleast_2d(x))
    grad_f = 0.5 * (24 * (6 * x) ** 3 - 192 * (6 * x) + 30)
    return np.einsum("ij,kj->ki", self.Qt, grad_f).squeeze()

# True model of log discrepancy
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

  def predict_noiseless(self, x):
    return self.fm(x), self.fv(x)

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

mins = np.full(num_dims, -1.5)
maxes = np.full(num_dims, 1.5)
bounds = np.vstack((mins, maxes)).T
true_kernel = []
true_groups = []
zuniq = np.unique(true_z)
for zz in zuniq:
    pdd = np.where(true_z == zz)[0]
    lenpd = len(pdd)
    true_groups.append(pdd)
    kernel_pdd = GPy.kern.RBF(input_dim=len(pdd), active_dims=pdd, ARD=False, name=gen_name())
    ##kernel_pdd.lengthscale.constrain_fixed(true_lengthscale)
    ##kernel_pdd.variance.constrain_fixed(true_var)
    true_kernel.append(kernel_pdd)
true_kernel = GPy.kern.Add(true_kernel)

for nn in range(nn_init, nn_end):
  seed_nn = seed + nn 
  RandomState = np.random.RandomState(seed_nn)
  np.random.seed(seed_nn)

  # Store info
  DF = pd.DataFrame(index=np.arange(0, nacqs+1, n_res_update), 
                     columns=["regret", "cumregret", "expected_fmin", "nearb", "l1nx", "linfx",
                              "loglik", "rand_idx", "tpr", "tnr", "prec", "f1_score",
                              "ari", "kl_n",
                              "ess_ref", "psrf_ref", "sqmmd_ref", "ess", "psrf", "sqmmd",
                              "ppc_mean", "ppc_var", "ppc_q25", "ppc_median", "ppc_q75",
                              "ppc_kurtosis", "ppc_skewness", "ppc_min", "ppc_max"])
  SAMPLES = pd.DataFrame(index=np.arange(0, nacqs+1, n_res_update), columns=np.arange(npost_samples)) # index 0 will be the reference
  # Other meta info to store
  kernel_times = []
  acq_times = []
  partitions = []
  resmeta = todict(n_res_update=n_res_update,
                   true_z=true_z, learn_type=learn_type, update_k=update_k, iters_k=iters_k,
                   kernel_times=kernel_times, acq_times=acq_times, partitions=partitions)  

  # Q = np.zeros((num_dims,num_dims))
  # for group in true_groups:
  #    d = len(group)
  #    if d > 1:
  #        A = np.random.randn(d,d)
  #        Q_d, _ = np.linalg.qr(A)
  #        Q[np.ix_(group, group)] = Q_d
  #    else:
  #        Q[group[0], group[0]] = 1


  angle = np.pi / 8
  Qi = np.array([[np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]])
  Q = np.zeros((num_dims,num_dims))
  for group in true_groups:
    d = len(group)
    assert(d == 2)
    Q[np.ix_(group, group)] = Qi

  print(Q)

  # Instantiate true model
  stybtang = Stybtang(Q)
  true_model = TrueModel(num_dims, fm=stybtang.rot_f,
                        fv=lambda x: 0, gm=stybtang.grad_rot_f,
                        gv=lambda x: np.zeros((1, x.size)),
                        lik_noise_var=lik_noise_var, xmin=stybtang.xmin,
                        seed=seed_nn)

  # f, ax = plt.subplots()
  # x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
  # X = np.vstack((x.flatten(), y.flatten())).T
  # f1 = true_model.fm(X).reshape(x.shape)
  # cs = ax.contour(x, y, f1, np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14,
  #                  linewidths=0.5, colors='k', linestyles="dashed")
  # ax.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
  # ax.contourf(x, y, f1, np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14, cmap="RdBu_r")  
  # plt.show()

  CHECK_OPTIMUM = True
  if CHECK_OPTIMUM:
    st = time.time()
    xmin_, fmin_ = minimize(true_model.fm, bounds, grad=true_model.gm,
                    n_start_points=10, maxiter=1000, random_state=RandomState,
                    options=None, guess_point=None)
    print("Optimization Results:")
    print("xmin: {}, fmin: {}".format(xmin_, fmin_))
    print("xmin truth: {}, fmin truth: {}".format(true_model.xmin,
          true_model.fmin))
    if true_model.xmin is None:
      true_model.xmin = xmin_
      true_model.fmin = fmin_

  # Definition of mean_function with priors
  mean_function = QPoly(num_dims, 1, seed=seed_nn, set_zero=True)
  mean_function.A = np.ones(num_dims)[:, None]
  [p.set_prior(GPy.priors.LogGaussian(0, 1)) for p in mean_function.A]
  [p.set_prior(GPy.priors.StudentT(0, 1, 3)) for p in mean_function.B]
  mean_function.C.set_prior(GPy.priors.StudentT(0, 1, 3))
  # mean_function = QuarticPoly(num_dims, 1, seed=seed_nn, set_zero=True)
  # mean_function.A = np.ones(num_dims)[:, None]
  # [p.set_prior(GPy.priors.LogGaussian(0, 1)) for p in mean_function.A]
  # [p.set_prior(GPy.priors.StudentT(0, 1, 3)) for p in mean_function.B]
  # [p.set_prior(GPy.priors.StudentT(0, 1, 3)) for p in mean_function.C]
  # mean_function.D.set_prior(GPy.priors.StudentT(0, 1, 3))


  # Definition of the kernels
  if learn_type == "oracle":
    initial_kernel = true_kernel.copy().parts
  elif learn_type != "indep":
    initial_kernel = [GPy.kern.RBF(input_dim=num_dims, active_dims=range(num_dims), ARD=False)]
    ##initial_kernel.lengthscale.constrain_fixed(true_lengthscale)
    ##initial_kernel.variance.constrain_fixed(true_var)
  else:
    initial_kernel = []
    for dd in range(num_dims):
      kk = GPy.kern.RBF(input_dim=1, active_dims=dd, ARD=False, name=gen_name())
      ##kk.lengthscale.constrain_fixed(true_lengthscale)
      ##kk.variance.constrain_fixed(true_var)
      initial_kernel.append(kk)
  initial_kernel = GPy.kern.Add(initial_kernel)

  bolfi = BOLFI(f=true_model.noisy_f, kernel_list=initial_kernel.copy().parts, bounds=bounds, acq_f=LCBSCADD, prior_f=None,
                initial_n=20, mean_function=mean_function.copy() if mean_function is not None else None,
                lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=None,
                variance=None, optimizer=None, messages=False, max_iters=max_opt_iters,
                logging=False, seed=seed_nn, n_update=update_k,
                acq_f_kws={"n_inits": max(10, int(num_dims)), "noise_var":acq_noise}, xopt=true_model.xmin,
                f_true=true_model.true_f, additive=additive, **kws)
  ##oracle = BOLFI(f=true_model.noisy_f, kernel_list=true_kernel.copy().parts, bounds=bounds,
  ##              initial_n=1, mean_function=mean_function.copy() if mean_function is not None else None,
  ##              max_iters=max_opt_iters, lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=None,
  ##              variance=None, kernel_update=False)

  # Set "true" loglikelihood (unnormalized)
  true_threshold = true_model.threshold
  loglik_true_un = lambda x: bolfi.unnormalized_loglikelihood(true_threshold, x, true_model, bounds)
  # Laplace approximation of the "true" loglikelihood function
  gsampler_true = bolfi.laplace_aprx(true_threshold, true_model, bounds, seed=seed_nn, guess_point=true_model.xmin)
  ## Sampling 1: Define importance sampler based on Laplace approximation
  ##def sir_true(size, random_state=RandomState):
  ##  return isampler(gsampler_true, loglik_true_un, prior=None, n_sample=size,
  ##                  random_state=random_state, resample=True, return_ess=True)
  ##xs, ess_ref[nn] = sir_true(size=10000)
  ##xs1 = sir_true(size=10000)[0]
  # Sampling 2: Use NUTS | MH
  #logger.setLevel(level=logging.INFO)
  #st_sample = time.time()
  #xs, effx, psrfx = bolfi.mcmc(true_threshold, n_iter=npost_samples, model=true_model, bounds=bounds,
  #                            params0=gsampler_true.mean, seed=seed_nn, get_stats=True,
  #                            n_chains=1, target_prob=0.7)
  #print("time sampling true (1): {0:.3f}".format(time.time() - st_sample))
  #print(effx)
  #print(psrfx)
  #asdasd
  #DF.loc[:, "ess_ref"]  = np.mean(effx)
  #DF.loc[:, "psrf_ref"] = np.mean(psrfx)
  #st_sample = time.time()
  #xs1 = bolfi.mcmc(true_threshold, n_iter=npost_samples, model=true_model, bounds=bounds,
  #                 params0=gsampler_true.mean, seed=seed_nn+1, get_stats=True,
  #                 n_chains=1, target_prob=0.7)
  #print("time sampling true (2): {0:.3f}".format(time.time() - st_sample))
  #logger.setLevel(level=logging.CRITICAL)
  ##mean_true, cov_true = true_model.predict_noiseless(xs, full_cov=True)
  ##cov_true[np.diag_indices(len(xs))] += 1e-8

  #postpred_true = true_model.noisy_f(xs)
  #DF.loc[:, "sqmmd_ref"] = sq_mmd(xs, xs1)

  ## fx = ss.gaussian_kde(postpred_true)
  ## x = np.linspace(np.min(postpred_true), np.max(postpred_true), 1000)
  ## plt.plot(x, fx(x))
  ## print(np.min(postpred_true))
  ## plt.show()
  for i in range(0, nacqs + 1):
    print("="*20)
    print("{}: {} - {}.\n".format(nn, i, bolfi.curr_model.log_likelihood()))
    if i > 0:
      bolfi.update_next_point(n=1, kernel_n_iters=iters_k)
    if i % n_res_update == 0:
      st = time.time()
      # Define loglikelihood approximation (unnormalized)
      print(bolfi.expected_fcurr)
      threshold_aprx = bolfi.expected_fcurr
      print("threshold_aprx: ", threshold_aprx)
      loglik_aprx_un = lambda x: bolfi.unnormalized_loglikelihood(threshold_aprx,
                                                x, bolfi.curr_model, bounds)
      # Store log-likelihood
      DF.loc[i, "loglik"] = bolfi.curr_model.log_likelihood()
      ## Store BF
      ##oracle.curr_model.set_XY(bolfi.curr_model.X, bolfi.curr_model.Y)
      ##oracle.curr_model.optimize(max_iters=max_opt_iters)
      ##logbf[nn, ii]  = oracle.curr_model.log_likelihood() - bolfi.curr_model.log_likelihood()
      ##logrbf[nn, ii] = oracle.curr_model.log_likelihood() / bolfi.curr_model.log_likelihood()
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
      print(DF.loc[i, "nearb"])
      print(bolfi.xcurr)
      if not DF.loc[i, "nearb"]:
        print(DF.loc[i, ["rand_idx", "tpr", "tnr", "prec", "f1_score", "regret", "expected_fmin"]])
        print(partitions[-1])
        # Store KL from Laplace approximation
        gsampler_aprx = bolfi.laplace_aprx(threshold_aprx, seed=seed_nn,
                                           guess_point=bolfi.xcurr)

        print(gsampler_aprx)
        #fig, ax2 = plt.subplots(1, 1, figsize=(12 , 6))
        #x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
        #X = np.vstack((x.flatten(), y.flatten())).T
        #f1 = bolfi.curr_model.predict_noiseless(X)[0].reshape(x.shape)
        #cs = ax.contour(x, y, f1, 50,
        #                 linewidths=0.5, colors='k', linestyles="dashed")
        #ax1.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
        #ax1.contourf(x, y, f1, 50, cmap="RdBu_r")
        #for xxx in bolfi.X:
        #  ax1.plot(xxx[0], xxx[1], "ko")
        #ax1.plot(bolfi.xcurr[0], bolfi.xcurr[1], "ro")
        #plt.show()
        #ff, Z = normalize(lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold_aprx, x)), bounds, return_Z=True)
        #print(bolfi.curr_model)
        #from mpl_toolkits.mplot3d import Axes3D
        #ax2 = fig.gca(projection='3d')
        #f1 = bolfi.curr_model.predict_noiseless(X)[0]
        #f1 = np.exp(loglik_true_un(X)).reshape(x.shape) / 0.08124061634136047
        #ax2.plot_trisurf(X[:,0], X[:,1], f1.ravel())

        # np.exp(bolfi.unnormalized_loglikelihood(threshold_aprx, X)
        #cs = ax2.contour(x, y, f1, 50,
        #                 linewidths=0.5, colors='k', linestyles="dashed")
        #ax2.clabel(cs, inline=1, fontsize=11, fontweight="bold", fmt="%1.1f")
        #ax2.contourf(x, y, f1, 50, cmap="RdBu_r")
        #for xxx in bolfi.X:
        #  ax2.plot(xxx[0], xxx[1], "ko")
        #ax2.plot(bolfi.xcurr[0], bolfi.xcurr[1], f1.ravel()[0],  "ro")
        #plt.show()
        # np.array([0, 1, 5, 15, 20, 30, 40, 50, 60, 100, 150, 175, 190, 300])-14
        #asdasd



        print("xcurr: ", bolfi.xcurr)
        print(gsampler_aprx is not None)
        if gsampler_aprx is not None:
          DF.loc[i, "kl_n"] = kl_normal(gsampler_true.mean, gsampler_true.cov,
                                        gsampler_aprx.mean, gsampler_aprx.cov)
          print("OK! ", DF.loc[i, "kl_n"])
        ##mean_aprx, cov_aprx = bolfi.curr_model.predict_noiseless(xs, full_cov=True)
        ##cov_aprx[np.diag_indices(len(xs))] += 1e-8
        ##kl_n2[nn, ii] = kl_normal(mean_true, cov_true, mean_aprx, cov_aprx)
        #print(i > 0 and gsampler_aprx is not None)
        #print(i % n_res_update2 == 0 and i > 0 and gsampler_aprx is not None)
        #if((i % n_res_update2 == 0) and i > 0):
        #  print("asjdbajsbdjasd")
        if False: #(i % n_res_update2 == 0) and i > 0 and gsampler_aprx is not None:
          print("OLA")
          asdasd
          ##ii2 = i // n_res_update2 - 1
          ## Sampling 1: Define importance sampler based on Laplace approximation
          ##def sir_aprx(size, random_state=RandomState):
          ##  return isampler(gsampler_aprx, loglik_aprx_un, prior=None, n_sample=size,
          ##                  random_state=random_state, resample=True, return_ess=True)
          ##ys, ess[nn, ii] = sir_aprx(size=10000)
          # Sampling 2: Use NUTS | MH
          logger.setLevel(level=logging.INFO)
          st_sample = time.time()
          ys, effx, psrfx = bolfi.mcmc(threshold_aprx, n_iter=npost_samples, params0=gsampler_aprx.mean, 
                                       seed=seed_nn, get_stats=True, n_chains=1, target_prob=0.7)
          ##ys, effx, psrfx = bolfi.mcmc(threshold_aprx, n_iter=100000, params0=gsampler_aprx.mean,
          ##                             sampler="mh", sigma_proposals=np.minimum(0.1*np.sqrt(np.diag(gsampler_aprx.cov)), 0.1*np.ones(num_dims)),
          ##                             thinning=100, seed=seed_nn, get_stats=True, n_chains=1)
          print("time sampling aprx: {0:.3f}".format(time.time() - st_sample))
          print(effx)
          print(psrfx)
          asdasdasd
          logger.setLevel(level=logging.CRITICAL)
          DF.loc[i, ["ess", "psrf", "sqmmd"]] = np.mean(effx), np.mean(psrfx), sq_mmd(xs, ys)
          postpred_aprx = noisy_func(ys)
          DF.loc[i, "ppc_mean"] = np.abs(np.mean(postpred_aprx) - np.mean(postpred_true)) 
          DF.loc[i, "ppc_var"] = np.abs(np.var(postpred_aprx, ddof=1) - np.var(postpred_true, ddof=1))
          DF.loc[i, "ppc_q25"] = np.abs(np.percentile(postpred_aprx, 25) - np.percentile(postpred_true, 25))
          DF.loc[i, "ppc_median"] = np.abs(np.median(postpred_aprx) - np.median(postpred_true))
          DF.loc[i, "ppc_q75"] = np.abs(np.percentile(postpred_aprx, 75) - np.percentile(postpred_true, 75))
          DF.loc[i, "ppc_kurtosis"] = np.abs(ss.kurtosis(postpred_aprx, bias=False) - ss.kurtosis(postpred_true, bias=False))
          DF.loc[i, "ppc_skewness"] = np.abs(ss.skew(postpred_aprx, bias=False) - ss.skew(postpred_true, bias=False))
          DF.loc[i, "ppc_min"] = np.abs(np.min(postpred_aprx) - np.min(postpred_true))
          DF.loc[i, "ppc_max"] = np.abs(np.max(postpred_aprx) - np.max(postpred_true))
      if False:
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
    #   wmode = "wb" if (nn == 0 and ii == 0) else "ab"
    #   if ii == 0:
    #     with open("{}/{}.csv".format(exp_path, exp_name), wmode) as fp:
    #       np.savetxt(fp, np.hstack((nn, ii, postpred_true))[None, :], delimiter=",")
    # if i == nacqs:
    #   kernel_times.append(bolfi.kernel_times)
    #   acq_times.append(bolfi.acq_times)
    # if i % n_res_update2 == 0 or i == nacqs:
    #   with open("{}/{}.xlsx".format(exp_path, exp_name), wmode) as fp:
    #     pd.DataFrame(postpred_aprx[None, :]).to_excel(fp, sheet_name="y"+str(nn), startrow=nn)
    #     DF.to_excel(fp, sheet_name="r"+str(nn), )
    #   #with open("{}/r_{}.obj".format(exp_path, exp_name), 'wb') as fp:
    #   #  pickle.dump(resdict, fp)
    #   if nn == 0:
    #     with open('{}/m_{}.obj'.format(exp_path, exp_name), 'wb') as fp:
    #       alphas = bolfi.kernel_sampler.alphas if learn_type == "mhno" else None
    #       pickle.dump({"gp": bolfi.curr_model, "mh_alphas": alphas}, fp)





    #with open("{}/{}.csv".format(exp_path, exp_name), "wb") as fp:
    #  np.savetxt(fp, np.vstack((regret[0], rand_idx[0], f1_score[0], nearb[0])).T , delimiter=",")

#alpha = np.ones(M)
#z = RandomState.choice(M, size=dim, p=RandomState.dirichlet(alpha))


# seed = 117
# true_lengthscale = 100# np.sqrt(2) * 3  #3
# true_var = 14 / (num_groups ** 2) # 9 * np.exp(.5) * np.abs(C1[0, 1]) 13.948
# print(true_var)
# lik_noise_var = 1e-8

# nruns = 10
# nacqs = 500
# true_n_obs = 3000
# n_res_update = 1
# n_res_update2 = 2 * n_res_update # must be multiple of n_res_update

# # Store results
# n_store = nacqs // n_res_update
# n_store2 = nacqs // n_res_update2
# kl_n = np.full((nruns, n_store), np.nan)
# kl_n2 = np.full((nruns, n_store), np.nan)
# regret = np.full((nruns, n_store), np.nan)
# cumregret = np.full((nruns, n_store), np.nan)
# rand_idx = np.full((nruns, n_store), np.nan)
# tpr = np.full((nruns, n_store), np.nan)
# tnr = np.full((nruns, n_store), np.nan)
# prec = np.full((nruns, n_store), np.nan)
# f1_score = np.full((nruns, n_store), np.nan)
# ari = np.full((nruns, n_store), np.nan)
# logbf = np.full((nruns, n_store), np.nan)
# logrbf = np.full((nruns, n_store), np.nan)
# nearb = np.full((nruns, n_store), False, dtype=bool)
# l1nx = np.full((nruns, n_store), np.nan)
# linfx = np.full((nruns, n_store), np.nan)
# sqmmd_ref = np.full(nruns, np.nan)
# sqmmd = np.full((nruns, n_store2), np.nan)
# ess_ref = np.full(nruns, np.nan)
# ess     = np.full((nruns, n_store2), np.nan)
# psrf_ref = np.full(nruns, np.nan)
# psrf     = np.full((nruns, n_store2), np.nan) 
# ppc_mean     = np.full((nruns, n_store2), np.nan) 
# ppc_var      = np.full((nruns, n_store2), np.nan) 
# ppc_q25      = np.full((nruns, n_store2), np.nan) 
# ppc_median   = np.full((nruns, n_store2), np.nan)
# ppc_q75      = np.full((nruns, n_store2), np.nan) 
# ppc_kurtosis = np.full((nruns, n_store2), np.nan)
# ppc_skewness = np.full((nruns, n_store2), np.nan)
# ppc_min      = np.full((nruns, n_store2), np.nan)
# ppc_max      = np.full((nruns, n_store2), np.nan)

# kernel_times = []
# acq_times = []
# partitions = []

# def todict(**kw):
#   return {k:v for k,v in kw.items()}
# resdict = todict(kl_n=kl_n, kl_n2=kl_n2, regret=regret, cumregret=cumregret, 
#                  rand_idx=rand_idx, tpr=tpr, tnr=tnr,
#                  prec=prec, f1_score=f1_score, ari=ari, 
#                  logbf=logbf, logrbf=logrbf,
#                  sqmmd=sqmmd, sqmmd_ref=sqmmd_ref,
#                  kernel_times=kernel_times, acq_times=acq_times,
#                  n_res_update=n_res_update,  
#                  true_z=true_z, learn_type=learn_type, update_k=update_k, iters_k=iters_k,
#                  partitions=partitions, ess_ref=ess_ref, ess=ess,
#                  nearb=nearb, l1nx=l1nx, linfx=linfx,
#                  psrf_ref=psrf_ref, psrf=psrf,
#                  ppc_mean=ppc_mean, ppc_var=ppc_var,
#                  ppc_q25=ppc_q25, ppc_median=ppc_median, ppc_q75=ppc_q75, 
#                  ppc_kurtosis=ppc_kurtosis, ppc_skewness=ppc_skewness,
#                  ppc_min=ppc_min, ppc_max=ppc_max)

# mins = np.full(num_dims, -3)
# maxes = np.full(num_dims, 3)
# bounds = np.vstack((mins, maxes)).T
# true_kernel = []
# true_groups = []
# zuniq = np.unique(true_z)
# for zz in zuniq:
#     pd = np.where(true_z == zz)[0]
#     lenpd = len(pd)
#     true_groups.append(pd)
#     kernel_pd = GPy.kern.RBF(input_dim=len(pd), active_dims=pd, ARD=False, name=gen_name())
#     kernel_pd.lengthscale.constrain_fixed(true_lengthscale)
#     kernel_pd.variance.constrain_fixed(true_var)
#     true_kernel.append(kernel_pd)
# true_kernel = GPy.kern.Add(true_kernel)

# if learn_type == "full":
#     kws = {"kernel_update":False}
#     msg = "no (assume full dependence)"
# if learn_type == "oracle":
#     kws = {"kernel_update":False}
#     msg = "oracle"
# elif learn_type == "indep":
#     kws = {"kernel_update":False}
#     msg = "no (assume full independence)"
# elif learn_type == "mhno":
#     kws = {"kernel_update":True, "kernel_sampler":MHNO,
#            "kernel_sampler_kws":{"ARD":False, "copy_kernels":False}}
#     msg = "yes (Metropolis Hastings). update_k: {}. iters_k: {}".format(update_k, iters_k)
# elif learn_type == "gibbsno":
#     kws = {"kernel_update":True, "kernel_sampler":GibbsNO,
#            "kernel_sampler_kws":{"ARD":False, "copy_kernels":False}}
#     msg = "yes (Gibbs). update_k: {}. iters_k: {}".format(update_k, iters_k)
# else:
#   raise ValueError("Invalid kernel learning approach.")
# print("Learn kernel: {}".format(msg))


# # Definition of the observations
# # 0.5 * (num_dims * np.log(2*np.pi) - len(groups) * np.log(1 - 0.6**2))
# mean_tt = np.tile(np.array([2, -2]), num_dims // 2)
# def f_obs(X, groups):
#   X = np.atleast_2d(X)
#   y = np.zeros(len(X))
#   for group in groups:
#     ngroup = len(group)
#     x = X[:, group]
#     mean = mean_tt[group]
#     y += np.array([0.5 * ((xx - mean).dot(C1).dot(xx - mean) - C1[0, 0] * np.sum(xx**2)) for xx in x])
#   return y

# mean_function = None
# if mean_a_ws is not None:
#   # Definition of mean_function
#   mean_function = QPoly(num_dims, 1, seed=seed)
#   mean_function.A.constrain_fixed(mean_a_ws*np.ones(num_dims)[:,None])
#   mean_function.B.constrain_fixed(np.zeros(num_dims)[:,None])
#   mean_function.C.constrain_fixed(0)


# gtimer = timer(nruns * nacqs)
# for nn in range(nruns):
#   print("="*30)
#   print("Starting run {} out of {}".format(nn+1, nruns))
#   seed_nn = seed + nn
#   RandomState = np.random.RandomState(seed_nn)
#   np.random.seed(seed_nn)

#   mean_f_copy = mean_function.copy() if mean_function is not None else None

#   num_s = 300
#   xmin = np.full(num_dims, np.inf)
#   i = 0
#   while not np.all(np.logical_and(bounds[0][0]+0.2 <= xmin, xmin <= bounds[0][1] - 0.2)):
#     print("Sampling function: {}".format(i+1))
#     radii = [0.01, 0.1, 0.5, 1] 
#     X0 = np.zeros((1 + len(radii) * num_s, num_dims))
#     #X0[0] = 0.5 * (num_dims * np.log(2*np.pi) - len(true_groups) * np.log(1 - 0.6**2))
#     X0[0] = mean_tt
#     for ii, rr in enumerate(radii):
#       Xaux = np.random.randn(num_s, num_dims)
#       Xaux = rr * Xaux / np.linalg.norm(Xaux, axis=1)[:,None] - mean_tt
#       X0[1+ii*num_s:1+(ii+1)*num_s] = Xaux
#       y0 = f_obs(X0, true_groups)

#     sequencer = ghalton.Halton(num_dims)
#     X1 = np.array(sequencer.get(true_n_obs - len(X0)))
#     X1 = (maxes - mins)[None,:] * X1 + mins[None,:]
#     K_10 = true_kernel.K(X1, X0)
#     K_00 = true_kernel.K(X0)
#     K_00[np.diag_indices(len(K_00))] += 1e-8
#     cho_00 = sl.cho_factor(K_00)
#     mean_cond = K_10.dot(sl.cho_solve(cho_00, y0[:,None])).ravel()
#     cov_cond = true_kernel.K(X1) - K_10.dot(sl.cho_solve(cho_00, K_10.T))
#     y1 = ss.multivariate_normal.rvs(mean=mean_cond, cov=cov_cond,
#                                     size=1, random_state=RandomState)
#     X_true = np.vstack((X0, X1))
#     y_true_no_mean = np.hstack((y0, y1.ravel()))
#     y_true = y_true_no_mean
#     if mean_function is not None:
#       y_true += mean_function.f(X_true).flatten()

#     #X_true = np.atleast_2d(np.ones(num_dims)*1.2)
#     #y_true = np.array([-5])
#     #y_obs = y_true + np.sqrt(lik_noise_var) * RandomState.randn(len(y_true))
#     true_model = GPy.models.GPRegression(X_true, y_true[:,None], true_kernel,
#                  noise_var=lik_noise_var, mean_function=mean_f_copy)
#     st = time.time()
#     xmin_full = min_gp(true_model, additive=False, bounds=bounds, seed=seed_nn,
#                        guess_point=np.zeros(num_dims))
#     time_full = time.time() - st
#     st = time.time()
#     xmin_add = min_gp(true_model, additive=True, bounds=bounds, seed=seed_nn,
#                       guess_point=np.zeros(num_dims))
#     time_add = time.time() - st
#     st = time.time()
#     xmin = xmin_add
#     print(xmin)
#     i += 1

#   print("OPTIMIZATION RESULTS")
#   for t, x in zip([time_full, time_add], [xmin_full, xmin_add]):
#       print("time: {}".format(t))
#       print("xmin: {}".format(x))
#       print("fmin: {}".format(pred_noiseless(x, true_model)[0]))
#       print("fmin0: {}".format(pred_noiseless(np.zeros(num_dims), true_model)[0]))
#       print()

#   def true_func(x):
#     return pred_noiseless(x, true_model)[0].ravel()
 
#   #fmin_full = true_func(xmin_full)
#   #fmin_add  = true_func(xmin_add)
#   if learn_type == "oracle":
#     initial_kernel = true_kernel.copy()
#   elif learn_type != "indep":
#     initial_kernel = GPy.kern.RBF(input_dim=num_dims, active_dims=range(num_dims), ARD=False)
#     initial_kernel.lengthscale.constrain_fixed(true_lengthscale)
#     initial_kernel.variance.constrain_fixed(true_var)
#   else:
#     initial_kernel = []
#     for dd in range(num_dims):
#       kk = GPy.kern.RBF(input_dim=1, active_dims=dd, ARD=False, name=gen_name())
#       kk.lengthscale.constrain_fixed(true_lengthscale)
#       kk.variance.constrain_fixed(true_var)
#       initial_kernel.append(kk)
#     initial_kernel = GPy.kern.Add(initial_kernel)

#   # Set threshold
#   threshold = true_model.predict_quantiles(np.atleast_2d(xmin_add), quantiles=(50,))[0].ravel()[0]
  
#   def noisy_func(x):
#     f = true_func(x)
#     return f + np.sqrt(lik_noise_var) * RandomState.randn(*f.shape)

#   bolfi = BOLFI(f=noisy_func, kernel_list=[initial_kernel.copy()], bounds=bounds, acq_f=LCBSCADD, prior_f=None,
#                 initial_n=10, mean_function=mean_function.copy() if mean_function is not None else None,
#                 lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=true_lengthscale,
#                 variance=true_var, optimizer=None, messages=False, max_iters=1000,
#                 logging=False, seed=seed_nn, n_update=update_k,
#                 acq_f_kws={"n_inits":10, "noise_var":0.0}, xopt=xmin_add,
#                 f_true=true_func, **kws)

#   oracle = BOLFI(f=noisy_func, kernel_list=[true_kernel.copy()], bounds=bounds,
#                 initial_n=10, mean_function=mean_function.copy() if mean_function is not None else None,
#                 lik_noise_var=lik_noise_var, fix_lik_noise_var=True, lengthscale=true_lengthscale,
#                 variance=true_var, kernel_update=False)

#   # Set "true" loglikelihood (unnormalized)
#   loglik_true_un = lambda x: bolfi.unnormalized_loglikelihood(threshold, x, true_model, bounds)
#   # Laplace approximation of the "true" loglikelihood function
#   gsampler_true = bolfi.laplace_aprx(threshold, true_model, bounds, seed=seed_nn,
#                                      guess_point=xmin_add)
#   ## Sampling 1: Define importance sampler based on Laplace approximation
#   ##def sir_true(size, random_state=RandomState):
#   ##  return isampler(gsampler_true, loglik_true_un, prior=None, n_sample=size,
#   ##                  random_state=random_state, resample=True, return_ess=True)
#   ##xs, ess_ref[nn] = sir_true(size=10000)
#   ##xs1 = sir_true(size=10000)[0]
#   # Sampling 2: Use NUTS | MH
#   #logger.setLevel(level=logging.DEBUG)
#   #st_sample = time.time()
#   #xs, effx, psrfx = bolfi.mcmc(threshold, n_iter=1000, model=true_model, bounds=bounds,
#   #                            params0=gsampler_true.mean, 
#   #                            seed=seed_nn, get_stats=True, n_chains=1, target_prob=0.7)
#   ##xs, effx, psrfx = bolfi.mcmc(threshold, n_iter=100000, model=true_model, bounds=bounds,
#   ##                             params0=gsampler_true.mean, sampler="mh", sigma_proposals=np.minimum(0.1*np.sqrt(np.diag(gsampler_true.cov)), 0.1*np.ones(num_dims)),
#   ##                             thinning=100, seed=seed_nn, get_stats=True, n_chains=1)
#   #print("time sampling true (1): {0:.3f}".format(time.time() - st_sample))
#   #ess_ref[nn] = np.mean(effx)
#   #psrf_ref[nn] = np.mean(psrfx)
#   #st_sample = time.time()
#   #xs1 = bolfi.mcmc(threshold, n_iter=1000, model=true_model, bounds=bounds,
#   #                 params0=gsampler_true.mean, 
#   #                 seed=seed_nn+1, get_stats=False, n_chains=1, target_prob=0.7)
#   ##xs1 = bolfi.mcmc(threshold, n_iter=100000, model=true_model, bounds=bounds,
#   ##                 params0=gsampler_true.mean, sampler="mh", sigma_proposals=np.minimum(0.1*np.sqrt(np.diag(gsampler_true.cov)), 0.1*np.ones(num_dims)),
#   ##                 thinning=100, seed=seed_nn+1, get_stats=False, n_chains=1)
#   #print("time sampling true (2): {0:.3f}".format(time.time() - st_sample))
#   #logger.setLevel(level=logging.WARNING)
#   #mean_true, cov_true = true_model.predict_noiseless(xs, full_cov=True)
#   #cov_true[np.diag_indices(len(xs))] += 1e-8
#   #postpred_true = noisy_func(xs)
#   #sqmmd_ref[nn] = sq_mmd(xs, xs1)
#   for i in range(1, nacqs+1):
#       print("="*20)
#       print("total: {}, eta: {}".format(*gtimer.update()))
#       print("{}: {} - {}.\n".format(nn+1, i, bolfi.curr_model.log_likelihood()))
#       bolfi.update_next_point(n=1, kernel_n_iters=iters_k)
#       if i % n_res_update == 0:
#         ii = i // n_res_update - 1
#         st = time.time()
#         # Define loglikelihood approximation (unnormalized)
#         loglik_aprx_un = lambda x: bolfi.unnormalized_loglikelihood(threshold,
#                                                   x, bolfi.curr_model, bounds)
#         # Store Bayes Factor
#         oracle.curr_model.set_XY(bolfi.curr_model.X, bolfi.curr_model.Y)
#         logbf[nn, ii]  = oracle.curr_model.log_likelihood() - bolfi.curr_model.log_likelihood()
#         logrbf[nn, ii] = oracle.curr_model.log_likelihood() / bolfi.curr_model.log_likelihood()
#         # print("{}".format(i))
#         # print("logbf: {0:.3f}, logrbf: {1:.3f}".format(
#         #       logbf[nn, ii], logrbf[nn, ii]))
#         # print("oracle: ", oracle.curr_model.log_likelihood())
#         # print("model: ", bolfi.curr_model.log_likelihood())
#         # print("=================")
#         # continue
#         # Store partition information
#         z = np.empty(num_dims)
#         for v, k in enumerate(bolfi.curr_model.kern.parts):
#             z[k.active_dims] = v
#         partitions.append(z)
#         pmetrics = partition_metrics(true_z, z)
#         rand_idx[nn, ii], tpr[nn, ii], tnr[nn, ii], prec[nn,ii], f1_score[nn,ii] = pmetrics
#         ari[nn, ii] = adjusted_rand_score(true_z, z)
#         # Store regret
#         regret[nn, ii] = bolfi.regret
#         cumregret[nn, ii] = bolfi.cumregret
#         # Store info x
#         nearb[nn, ii] = np.any(np.logical_or(bolfi.xcurr >= bounds[:,1] -0.1, bolfi.xcurr <= bounds[:,0]+0.1))
#         l1nx[nn, ii] = np.sum(np.abs(xmin_add - bolfi.xcurr)) / num_dims
#         linfx[nn, ii] = np.max(np.abs(xmin_add - bolfi.xcurr))
#         gsampler_aprx = None
#         if False:#not nearb[nn, ii]:
#           # Store KL from Laplace approximation
#           gsampler_aprx = bolfi.laplace_aprx(threshold, seed=seed_nn,
#                                              guess_point=bolfi.xcurr)
#           print("xcurr: ", bolfi.xcurr)
#           if gsampler_aprx is not None:
#             kl_n[nn, ii] = kl_normal(gsampler_true.mean, gsampler_true.cov,
#                                      gsampler_aprx.mean, gsampler_aprx.cov)
#           mean_aprx, cov_aprx = bolfi.curr_model.predict_noiseless(xs, full_cov=True)
#           cov_aprx[np.diag_indices(len(xs))] += 1e-8
#           kl_n2[nn, ii] = kl_normal(mean_true, cov_true, mean_aprx, cov_aprx)
#           if i % n_res_update2 == 0 and gsampler_aprx is not None:
#             ii2 = i // n_res_update2 - 1
#             ## Sampling 1: Define importance sampler based on Laplace approximation
#             ##def sir_aprx(size, random_state=RandomState):
#             ##  return isampler(gsampler_aprx, loglik_aprx_un, prior=None, n_sample=size,
#             ##                  random_state=random_state, resample=True, return_ess=True)
#             ##ys, ess[nn, ii] = sir_aprx(size=10000)
#             # Sampling 2: Use NUTS | MH
#             logger.setLevel(level=logging.DEBUG)
#             st_sample = time.time()
#             ys, effx, psrfx = bolfi.mcmc(threshold, n_iter=1000, params0=gsampler_aprx.mean, 
#                                          seed=seed_nn, get_stats=True, n_chains=1, target_prob=0.7)
#             ##ys, effx, psrfx = bolfi.mcmc(threshold, n_iter=100000, params0=gsampler_aprx.mean,
#             ##                             sampler="mh", sigma_proposals=np.minimum(0.1*np.sqrt(np.diag(gsampler_aprx.cov)), 0.1*np.ones(num_dims)),
#             ##                             thinning=100, seed=seed_nn, get_stats=True, n_chains=1)
#             print("time sampling aprx: {0:.3f}".format(time.time() - st_sample))
#             logger.setLevel(level=logging.WARNING)
#             ess[nn, ii2] = np.mean(effx)
#             psrf[nn, ii2] = np.mean(psrfx)
#             sqmmd[nn, ii2] = sq_mmd(xs, ys)
#             postpred_aprx = noisy_func(ys)
#             ppc_mean[nn, ii2] = np.abs(np.mean(postpred_aprx) - np.mean(postpred_true)) 
#             ppc_var[nn, ii2] = np.abs(np.var(postpred_aprx, ddof=1) - np.var(postpred_true, ddof=1))
#             ppc_q25[nn, ii2] = np.abs(np.percentile(postpred_aprx, 25) - np.percentile(postpred_true, 25))
#             ppc_median[nn, ii2] = np.abs(np.median(postpred_aprx) - np.median(postpred_true))
#             ppc_q75[nn, ii2] = np.abs(np.percentile(postpred_aprx, 75) - np.percentile(postpred_true, 75))
#             ppc_kurtosis[nn, ii2] = np.abs(ss.kurtosis(postpred_aprx, bias=False) - ss.kurtosis(postpred_true, bias=False))
#             ppc_skewness[nn, ii2] = np.abs(ss.skew(postpred_aprx, bias=False) - ss.skew(postpred_true, bias=False))
#             ppc_min[nn, ii2] = np.abs(np.min(postpred_aprx) - np.min(postpred_true))
#             ppc_max[nn, ii2] = np.abs(np.max(postpred_aprx) - np.max(postpred_true))
#         print("xcurr: ", bolfi.xcurr)    
#         print("elapsed time: {0:.3f}".format(time.time() - st))
#         print("learn_type: ", learn_type)
#         print("target partition: {}".format(true_z))
#         print("current partition: {}".format(partitions[-1]))
#         print("rand_idx {0:.3f},  tpr: {1:.3f},  tnr: {2:.3f}, ari: {3:.3f}".format(
#               rand_idx[nn, ii], tpr[nn, ii], tnr[nn, ii], ari[nn, ii]))
#         print("prec: {0:.3f},  f1_score: {1:.3f}".format(prec[nn,ii], f1_score[nn,ii]))
#         print("regret: {0:.3e},  logbf: {1:.3f}, logrbf: {2:.3f}".format(
#               regret[nn, ii], logbf[nn, ii], logrbf[nn, ii]))
#         print("nearb: {0}, l1nx: {1:.3e}, linfx: {2:.3e}".format(nearb[nn, ii], l1nx[nn, ii], linfx[nn, ii]))
#         print("xmin: ", xmin_add)
#         print("xbolfi: ", bolfi.xcurr)
#         if gsampler_aprx is not None:
#           print("KL: {0:.3e},  KL2: {1:.3e}".format(kl_n[nn, ii], kl_n2[nn, ii]))
#           if i % n_res_update2 == 0:
#             print("ESS: ref - {0:.3e}, aprx - {1:.3e}".format(ess_ref[nn], ess[nn, ii2]))
#             print("PSRF: ref - {0:.3e}, aprx - {1:.3e}".format(psrf_ref[nn], psrf[nn, ii2]))
#             print("sqMMD: ref - {0:.3e}, aprx - {1:.3e}".format(sqmmd_ref[nn], sqmmd[nn, ii2]))
#             print("ppc_mean: {0:.3e}, ppc_var: {1:.3e}".format(ppc_mean[nn, ii2], ppc_var[nn, ii2]))
#             print("ppc_q25: {0:.3e}, ppc_q50: {1:.3e}, ppc_q75: {2:.3e}".format(ppc_q25[nn, ii2], ppc_median[nn, ii2], ppc_q75[nn, ii2]))
#             print("ppc_min: {0:.3e}, ppc_max: {1:.3e}".format(ppc_min[nn, ii2], ppc_max[nn, ii2]))
#             print("ppc_skew: {0:.3e}, ppc_kurt: {1:.3e}".format(ppc_skewness[nn, ii2], ppc_kurtosis[nn, ii2]))
#         # if len(bolfi.kernel_times) > 0:
#         #   print("kernel times: {0:.3f}".format(np.mean(bolfi.kernel_times)))
#         # if len(bolfi.acq_times) > 0:
#         #   print("acq times: {0:.3f}".format(np.mean(bolfi.acq_times)))
#       #   wmode = "wb" if (nn == 0 and ii == 0) else "ab"
#       #   if ii == 0:
#       #     with open("{}/x_{}.csv".format(exp_path, exp_name), wmode) as fp:
#       #       np.savetxt(fp, np.hstack((nn, ii, postpred_true))[None, :], delimiter=",")
#       # if i % n_res_update2 == 0 or i == nacqs:
#       #   with open("{}/y_{}.csv".format(exp_path, exp_name), wmode) as fp:
#       #     np.savetxt(fp, np.hstack((nn, i, postpred_aprx))[None, :], delimiter=",")
#       #   with open("{}/r_{}.obj".format(exp_path, exp_name), 'wb') as fp:
#       #     pickle.dump(resdict, fp)
#       #   if nn == 0:
#       #     with open('{}/m_{}.obj'.format(exp_path, exp_name), 'wb') as fp:
#       #       alphas = bolfi.kernel_sampler.alphas if learn_type == "mhno" else None
#       #       pickle.dump({"gp": bolfi.curr_model, "mh_alphas": alphas}, fp)
#       # if i == nacqs:
#       #   kernel_times.append(bolfi.kernel_times)
#       #   acq_times.append(bolfi.acq_times)

    





#

#
#            print("="*20)
#            print("total: {}, eta: {}".format(*gtimer.update()))
#            print("{}, {}: {} - {}".format(nn+1, ii, bolfi.curr_model.log_likelihood(),
#                  [kk.active_dims for kk in bolfi.curr_model.kern.parts]))
#
#            # Define aprx. posterior function (prior: uniform)
#            post_aprx_un = lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold, x))
#
#
#
#
#            xs = true_laplacesampler.rvs(size=10000)
#            xs2 = true_laplacesampler.rvs(size=10000)
#            ys = aprx_laplacesampler.rvs(size=10000)
#            st = time.time()
#            print("MMD  ", sq_mmd(xs, ys))
#            print("MMD  ", sq_mmd(xs, xs2))
#            print(time.time() - st)
#            st = time.time()
#            print("MMD test  ", bootstrap_test(xs, ys, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs))
#            print(time.time() - st)
#            print("MMD test  ", bootstrap_test(xs, xs2, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs))
#            asdasdasd
#            print(rnd_score[nn, ii])
#            print(tpr[nn, ii])
#            print(tnr[nn, ii])
#            st = time.time()
#            print(quasi_kl(xs, ys))
#            print(time.time() - st)
#            print()
#            print(kl)
#    #        print()
#    #        seed_nn = 12
#    #        #st = time.time
#            st = time.time()
#            samples, eff_sizes, psrf = bolfi.mcmc(threshold, n_iter=2500, model=true_model,
#                                                  bounds=bounds,
#                                                  params0=xmin_add, seed=seed_nn, get_stats=True, n_chains=4)
#            samples_true = samples.reshape((-1, samples.shape[-1]))
#            print(time.time() - st)
#            print(samples_true.shape)
#            print(eff_sizes)
#            print(psrf)
#            st = time.time()
#            samples, eff_sizes, psrf = bolfi.mcmc(threshold, n_iter=2500, model=bolfi.curr_model,
#                                                  bounds=bounds,
#                                                  params0=xmin_add, seed=seed_nn, get_stats=True, n_chains=4)
#            samples_aprx = samples.reshape((-1, samples.shape[-1]))
#            print(time.time() - st)
#            print("QKL: ", quasi_kl(samples_true, samples_aprx))
#            print("MMD: ", sq_mmd(samples_true, samples_aprx))
#            print("MMD TT: ", sq_mmd(samples_true, samples_true))
#            asdasdasd


#        xs0 = true_laplacesampler.rvs(size=100)
#        plt.figure()
#        x,y = np.meshgrid(np.linspace(mins[0], maxes[0], 100), np.linspace(mins[1], maxes[1], 100))
#        X = np.vstack((x.flatten(), y.flatten())).T
#        print(X.shape)
#        f1 = np.exp(loglik_true_un(X)).reshape(x.shape)
#        cs = plt.contour(x, y, f1, 14, linewidths=0.5, colors='k')
#        plt.clabel(cs, inline=1, fontsize=10)
#        cntr1 = plt.contourf(x, y, f1, 14, cmap="RdBu_r")
#        plt.show()
#        print(xs0[:3])
#        plt.scatter(x=xs0[:, 0], y=xs0[:, 1], color='r', marker="o")
#        xsg = true_gsampler.rvs(size=100)
#        plt.scatter(x=xsg[:, 0], y=xsg[:, 1], color='g', marker="o")
#        yy2 = isampler(true_laplacesampler, loglik_true_un, n_sample=100, resample=True, seed=seed_nn)
#        plt.scatter(x=yy2[:,0], y=yy2[:,1], color='y', marker="o")
#        plt.show()
#        asdasd
#        yy = isampler(true_laplacesampler, loglik_true_un, n_sample=100)[0]
#        plt.scatter(yy[:,0], yy[:,1], 'go')
#        yy2 = isampler(true_laplacesampler, loglik_true_un, n_sample=100, resample=True, seed=seed_nn)
#        plt.plot(yy2[:,0], yy2[:,1], 'yo')
#        plt.show()
#        asdasd
#        x = np.linspace(-3, 3, 1000)
#        plt.plot(x, np.exp(loglik_true_un(x)), "k--")
#        plt.show()
#        asdasd
#        print(xs0[:2, :5])
#        xs1 = true_laplacesampler.rvs(size=10000)
#        print()
#        print(xs1[:2, :5])
#        asdasda
#        xs2, ws2  = isampler(true_laplacesampler, loglik_true_un, n_sample=10000)
#        xs3, ws3 = isampler(true_laplacesampler, loglik_true_un, n_sample=10000)
#        print("sum to 1:  ", np.sum(ws2))
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs0, xs0, measure="sqmmd", seed=seed_nn, n_jobs=1)
#        print("1st LapLap: ", sqmmd, "  ", std, "    ", time.time()-st)
#        asddasd
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs2, xs3, ws2, ws3, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs)
#        print("2nd ISIS: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs2, xs3, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs)
#        print("2ndn ISIS no w: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs3, xs0, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs) # wxs=ws3
#        print("3rd ISLap: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs2, xs1, measure="sqmmd", seed=seed_nn, n_jobs=n_jobs) # wxs=ws3
#        print("4th ISLap: ", sqmmd, "  ", std, "    ", time.time()-st)
#        print("============ KL ======================")
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs0, xs1, measure="qkl", seed=seed_nn, n_jobs=n_jobs)
#        print("1st LapLap: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs2, xs3, ws2, ws3, measure="qkl", seed=seed_nn, n_jobs=n_jobs)
#        print("2nd ISIS: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs3, xs0, wxs=ws3, measure="qkl", seed=seed_nn, n_jobs=n_jobs)
#        print("3rd ISLap: ", sqmmd, "  ", std, "    ", time.time()-st)
#        st = time.time()
#        sqmmd, std = bootstrap_test(xs0, xs3, wys=ws3, measure="qkl", seed=seed_nn, n_jobs=n_jobs)
#        print("3rd LapIS: ", sqmmd, "  ", std, "    ", time.time()-st)
#        asdasdasdasd
#






    # x,y = np.meshgrid(np.linspace(mins[0], maxes[0], 100), np.linspace(mins[1], maxes[1], 100))
    # X = np.vstack((x.flatten(), y.flatten())).T
    # print(X.shape)
    # f1 = true_func(X).reshape(x.shape)
    # cs = plt.contour(x, y, f1, 14, linewidths=0.5, colors='k')
    # plt.clabel(cs, inline=1, fontsize=10)
    # cntr1 = plt.contourf(x, y, f1, 14, cmap="RdBu_r")
    # plt.show()





    #        #samplesABC,acpt = rejectionABC(noisy_func, threshold, bounds=bounds, seed=seed_nn)
    #        threshold_init = pilotABC(noisy_func, quantile=0.9, bounds=bounds, seed=seed_nn)
    #        print("threhold init: ", threshold_init)
    #        samplesABC, iters  = SMCABC(noisy_func, threshold_init, threshold,
    #                                    bounds=bounds, seed=seed_nn, verbose=True)
    #        print(time.time())
    #        print(iters)
    #        print("ABC  ", sq_mmd(xs, samplesABC))
    #        print("ABC  ", sq_mmd(samplesABC, xs))
    #        print("ABC aprx  ", sq_mmd(samplesABC, ys))
    #        asdasdasd
            #kl = kl_normal(true_laplacesampler.mean, true_laplacesampler.cov,
            #               true_laplacesampler.mean, true_laplacesampler.cov)

        #print(kl)

        #asdasdasd
        #post_aprx, Z_aprx = normalize_1d(post_aprx_un, bounds=bounds[0], return_Z=True)
        #qlik = lambda x, q: np.exp(bolfi.unnormalized_loglikelihood_quantiles(threshold, x, quantiles=(q,))[0])
        #qpost_lb = lambda x: qlik(x, 0.025) / Z_aprx
        #qpost_ub = lambda x: qlik(x, 0.975) / Z_aprx
        #kl_dists[nn, ii] = dist_1d(post_true, post_aprx, bounds=bounds[0], type="kl")
        #tv_dists[nn, ii] = dist_1d(post_true, post_aprx, bounds=bounds[0], type="tv")


#        x,y = np.meshgrid(np.linspace(mins[0], maxes[0], 100), np.linspace(mins[1], maxes[1], 100))
#        print(x)
#        X = np.vstack((x.flatten(), y.flatten())).T
#        # print(X.shape)
#        f1 = post_true_un(X).reshape(x.shape)
#        cs = plt.contour(x, y, f1, linewidths=0.5, colors='k')
#        plt.clabel(cs, inline=1, fontsize=10)
#        f2 = sampler.pdf(X).reshape(x.shape)
#        cs = plt.contour(x, y, f1, linewidths=0.5, colors='r', linestyles="dashed")
#        plt.plot(xmin_add[0], xmin_add[1], "bo")
#        plt.clabel(cs, inline=1, fontsize=5)
#        # cntr1 = plt.contourf(x, y, f1, 14, cmap="RdBu_r")
#        plt.show()
#        asdasdasd


#        Z_3, std3 = imp_samp(post_true_un, samplern, n=100000)
#        Z_true1 = mcint(post_true_un, bounds=bounds, n=100000, quasi=True)
#        Z_true2, std2 = mcint(post_true_un, bounds=bounds, n=100000, quasi=False)
#        from utils import sir
#        Z_4 = sir(post_true_un, samplern, n=100000)
#        #print("quad: {}".format(Z_true0))
#        print("qmc: {}".format(Z_true1))
#        print("mc: {} {}".format(Z_true2, std2))
#        print("is: {} {}".format(Z_3, std3))
#        print("sir: {}".format(Z_4))
#        samplert = bolfi.sampler_aprx(threshold, xmin_add, true_model, bounds, seed=seed, df=2)
#        Z_10, std10 = imp_samp(post_true_un, samplert, n=100000)
#        print("is t: {} {}".format(Z_10, std10))
#        Z_last = sir(post_true_un, samplert, n=100000)
#        print("sir t: {}".format(Z_last))
#        #post_true = lambda x: post_true_un(x) / Z_true0
#        print("tv n : {}".format(dist(post_true, samplern.pdf, bounds=bounds, measure="tv")))
#        print("tv t : {}".format(dist(post_true, samplert.pdf, bounds=bounds, measure="tv")))

        #Z_true1, std = mcint(post_true_un, dims=2, n=100000, bounds=bounds, quasi=False)
        #print(Z_true0)
        #print(Z_true1)
        #print(Z_true0 - Z_true1)
        #print(std)
# 200000
#qmc: -0.15407410888072343
#mc: -0.11015108914077892 -0.019068645756197836
#is: 0.14750934393795487
#sir: 0.14440938833489164
#is t: 0.14743888698945723
#sir t: 0.14532463365398274


#        if nn == 0:
#            # Make figures
#            fig = plt.figure(figsize=(12,8))
#            ax1 = plt.subplot(211)
#            ax1.set_title("Iteration {}".format(ii), fontsize=16)
#            ax1.plot(x, pred_noiseless(x, true_model)[0].ravel(), "r-", linewidth=2,
#                    label="True Discrepancy")
#            ax1.plot(x, pred_noiseless(x, bolfi.curr_model)[0].ravel(), "b--",
#                    linewidth=2, label="Aprx. Discrepancy Mean")
#            ax1.plot(bolfi.X, bolfi.y, "+", color="navy", markersize=10,
#                    label="Acquisitions")
#            d_quantiles = bolfi.curr_model.predict_quantiles(x[:,None],
#                                                             quantiles=(2.5, 97.5))
#            ax1.fill_between(x, d_quantiles[0].ravel(), d_quantiles[1].ravel(),
#                            alpha=0.4, label="Aprx. Discrepancy 95% CI")
#            ax1.axhline(y=threshold, color='orangered', linestyle='--',
#                       label="Threshold ({0:.3f})".format(threshold))
#            # Plot posteriors (scaled)
#            ax1.plot(x, scale * post_true(x), "--", color="brown", linewidth=1.5,
#                    label="True Posterior (rescaled)")
#            ax1.plot(x, scale * post_aprx(x), "k--", linewidth=1.5,
#                    label="Aprx. Posterior Mean (rescaled)")
#            ax1.fill_between(x, scale * qpost_lb(x), scale * qpost_ub(x), color="black",
#                            alpha=0.2, label="Aprx. Posterior 95% CI")
#            # Set plot properties
#            ax1.grid()
#            ax1.set_xlim((-3, 3))
#            ax1.set_ylim((-8, 25))
#            ax1.set_xlabel(r"Parameter  $\theta$", size=14)
#            plt.xticks(size=12)
#            ax1.set_ylabel(r"Discrepancy  $\Delta_{\theta}$", size=14)
#            plt.yticks(size=12)
#            ax1.legend()
#            ax2 = plt.subplot(223)
#            ax2.set_xlim((0, nacqs))
#            ax2.set_ylim((0, 0.05))
#            ax2.set_yticks(np.arange(0, 0.1+0.01, 0.01))
#            ax2.set_ylabel("KL Divergence", size=14)
#            ax2.set_xlabel("Iteration", size=14)
#            ax2.plot(kl_dists[nn, :ii+1], "ro--", markersize=3,
#                     label="KL Divergence")
#            ax2.grid()
#            ax3 = plt.subplot(224)
#            ax3.set_xlim((0, nacqs))
#            ax3.set_ylim((0, 0.15))
#            ax3.set_yticks(np.arange(0, 0.15+0.015, 0.015))
#            ax3.set_ylabel("TV Distance", size=14)
#            ax3.set_xlabel("Iteration", size=14)
#            ax3.plot(tv_dists[nn, :ii+1], "go--", markersize=3,
#                     label="TV Distance")
#            ax3.grid()
#            plt.savefig("simple1D/figures_nonoise/{0:03d}.png".format(ii))
#            plt.close()
#
#    np.savetxt("simple1D/kl_dists_nonoise.csv", kl_dists, delimiter=",")
#    np.savetxt("simple1D/tv_dists_nonoise.csv", tv_dists, delimiter=",")
#



#with open('model.obj', 'wb') as fp:
#     pickle.dump(bolfi, fp)
#
#with open('model.obj', 'rb') as fp:
#     bolfi = pickle.load(fp)
#
#print(bounds)





#num_dims = 2
#num_groups = 2
#seed = 117
#lik_noise_var = num_dims #0.5
#
#RandomState = np.random.RandomState(seed)
#global_opt = np.zeros(num_dims)#-2.903534 * np.ones(num_dims)# np.zeros(num_dims)
#
#alpha = np.ones(num_groups)
#true_z = [0, 1] #RandomState.choice(num_groups, size=num_dims, p=RandomState.dirichlet(alpha))
#z_uniq = np.unique(true_z)
#true_groups = [np.where(true_z == v)[0] for v in z_uniq]
#Q = np.zeros((num_dims,num_dims))
#for group in true_groups:
#    d = len(group)
#    if d > 1:
#        A = RandomState.randn(d,d)
#        Q_d, _ = np.linalg.qr(A)
#        Q[np.ix_(group,group)] = Q_d
#    else:
#        Q[group[0], group[0]] = 1
#
#v = RandomState.uniform(1, 3, size=num_dims)
#T = np.diag(v).dot(Q)
#func = lambda x: sphere(T.dot(x))
#funci = lambda x, group: sphere(T[np.ix_(group, group)].dot(x[group]))
#inverse_opt = np.linalg.inv(T).dot(global_opt) # for sphere this will be zeros still
#mins = inverse_opt - 3
#maxes = inverse_opt + 3
#bounds = [_ for _ in zip(*[mins, maxes])]
#
#kernel_list = [GPy.kern.RBF(input_dim=num_dims,active_dims=range(num_dims),
#                           ARD=False)]
#mean_function = QPoly(num_dims, 1, seed=seed)
#
#noisy_func = lambda x: func(x) + RandomState.normal(scale=np.sqrt(lik_noise_var))
#
#import matplotlib.pyplot as plt
#print(v)
#x,y = np.meshgrid(np.linspace(mins[0], maxes[0], 100), np.linspace(mins[1], maxes[1], 100))
#X = np.vstack((x.flatten(), y.flatten())).T
#f1 = np.array([func(xxx) for xxx in X]).reshape(x.shape)
#plt.contour(x,y,f1,colors='b')
#plt.show()
#asdasd
##plt.plot(xx, np.array([func(xxx) for xxx in xx]), 'ro')
##plt.show()
##asdasd
#
#
#
#
#bolfi = BOLFI(f=noisy_func, kernel_list=kernel_list, bounds=bounds, acq_f=LCBSC, prior_f=None,
#              initial_n=20, mean_function=mean_function, lik_noise_var=lik_noise_var,
#              fix_lik_noise_var=True,
#              optimizer=None, messages=False, max_iters=1000,
#              logging=False, seed=117, n_update=30,
#              kernel_update=True, kernel_sampler=MHNO,
#              kernel_sampler_kws={"ARD":False, "copy_kernels":True},
#              acq_f_kws={"n_inits":10, "noise_var":0.0})
#import os
#path = os.path.abspath(os.path.join(os.getcwd(), "../test_images2/"))
#jjj = 0
#for ii in range(500):
#    bolfi.update_next_point(n=1, kernel_n_iters=10)
#    print("{}: {} - {}".format(ii, bolfi.curr_model.log_likelihood(),
#          [kk.active_dims for kk in bolfi.curr_model.kern.parts]))
#    if not ii % 30 and ii > 0:
#        print(bolfi.curr_model)
#        print("===============")
#        xx = np.linspace(mins[0], maxes[0], 100)
#        #plt.plot(xx, np.array([func(xxx) for xxx in xx]), 'bo')
#        #plt.plot(xx, np.array([noisy_func(xxx) for xxx in xx]), 'ro')
#        #print(bolfi.curr_model.predict_noiseless(xx[:,None])[0].squeeze())
#        #plt.plot(xx, bolfi.curr_model.predict_noiseless(xx[:,None])[0].squeeze(), 'go')
#        #plt.show()
#
##        x,y = np.meshgrid(np.linspace(mins[0], maxes[0], 100), np.linspace(mins[1], maxes[1], 100))
##        X = np.vstack((x.flatten(), y.flatten())).T
##        f1 = np.array([func(xxx) for xxx in X]).reshape(x.shape)
##        f2 = bolfi.curr_model.predict_noiseless(X)[0].squeeze().reshape(x.shape)
##        plt.figure()
##        plt.contour(x,y,f1,colors='b')
##        plt.contour(x,y,f2,colors='g')
##        plt.plot(bolfi.curr_model.X[:,0], bolfi.curr_model.X[:,1], "ko")
##        plt.savefig(os.path.abspath(os.path.join(path,"{0:03d}.png".format(jjj))))
##        plt.close()
##        jjj+=1
#
#
#os.system("""ffmpeg -r 2 -i "{}\%03d.png" -f mp4 -q:v 0 -vcodec mpeg4 -r 2 {}\movie.mp4""".format(path, path))
#
#

#ffmpeg -r 2 -i "%03d.png" -f mp4 -q:v 0 -vcodec mpeg4 -r 2 simple1d.mp4
#
#def sphere(x):
#    return np.sum(x**2)
#
## Number of iterations of Bayesopt to run
#num_iters = 200
#
## Number of random runs to do
#num_runs = 1
#
## Dimensionality of function to optimize.
#num_dims = 10
#grid_size = 10000
#all_dim_groups = []
#all_X = [[] for i in range(4)]
#all_y = [[] for i in range(4)]
#global_opt = -2.903534 * np.ones(num_dims)
#
## TEST HYPERS WHEN PUTTING ADD KERNEL
#
#
#for i in range(num_runs):
#    #partition_kernel = GPy.kern.Add([GPy.kern.RBF(input_dim=num_dims,
#    #                                              active_dims=range(num_dims))])
#    #prop = KShiftDimProposal(1000)
#    #partition_kernel = prop.propose([partition_kernel])
#
#    alpha = np.ones(M)
#    z = RandomState.choice(M, size=dim, p=RandomState.dirichlet(alpha))
#    dim_groups = [k.active_dims for k in partition_kernel[0]]
#    Q = np.zeros((num_dims,num_dims))
#    for group in dim_groups:
#        d = len(group)
#        if d > 1:
#            A = np.random.randn(d,d)
#            Q_d, _ = np.linalg.qr(A)
#            Q[np.ix_(group,group)] = Q_d
#        else:
#            Q[group[0],group[0]] = 1
#    func = lambda x: stybtang(Q.dot(x))
#    inverse_opt = np.transpose(Q).dot(global_opt)
#
#    # mins and maxes are a numpy array of, for each dimension, the min and max value bayesopt will search over.
#    # For this experiment, we take the true optimum and search over a hypercube of +/- 3 in each dimension centered
#    # at the optimum.
#    mins = inverse_opt - 3
#    maxes = inverse_opt + 3
#    all_dim_groups.append(dim_groups)
#    print(dim_groups)
#
#    initial_kernel = GPy.kern.RBF(input_dim=num_dims,
#                                  active_dims=range(num_dims),
#                                  ARD=True) # fully-dependent
#    initial_kernel = [initial_kernel]
