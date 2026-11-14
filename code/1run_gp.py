import os
import sys
import argparse
import numpy as np
import GPy
from acquisition import LCBSC, LCBSCADD
from no_decomp import MHNO, GibbsNO, GPSampler
from gp_opt import BOLFI
from qpoly import QPoly, LogQPoly
import ghalton
import sobol_seq
import scipy.integrate as integrate
import time
import matplotlib.pyplot as plt
from utils import minimize
from utils import pred_noiseless, pred_grad, min_gp
from utils import normalize, dist, timer
import json
import pickle

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
    #idx = np.where(fnoise < self.fmin)[0]
    #while len(idx) > 0 and np.any(fnoise[idx] < self.fmin):
    #  fnoise[idx] = f[idx] + np.sqrt(fv[idx] + self.lik_nv) * self.random_state.randn(*fnoise[idx].shape)
    #  idx = idx[np.where(fnoise[idx] < self.fmin)[0]]
    return fnoise.squeeze()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Discrepancy function as proxy simulation model (1D)")
    parser.add_argument('--acq_noise', type=float, nargs='?', default=0.1, help='acquisition noise')
    args = parser.parse_args()
    acq_noise = args.acq_noise
    respath = os.path.abspath("./simple1D")
    if not os.path.exists(respath):
        os.makedirs(respath)
    figpath = os.path.join(respath, "acqnoise_{}".format(acq_noise))
    if not os.path.exists(figpath):
        os.makedirs(figpath)    

    nruns = 1
    nacqs = 50
    seed = 117
    true_lengthscale = 3
    true_var = 10
    lik_noise_var = 0.1
    true_n_obs = 100
    
    kl_dists = np.full((nruns, nacqs+1), np.nan)
    tv_dists = np.full((nruns, nacqs+1), np.nan)

    run_full_dependence = True

    z = [0]
    num_dims = len(z)
    mins = np.full(num_dims, -3)
    maxes = np.full(num_dims, 3)
    bounds = np.vstack((mins, maxes)).T
    zuniq = np.unique(z)
    true_groups = []
    true_kernel = []
    for zz in zuniq:
        pd = np.where(z == zz)[0]
        lenpd = len(pd)
        true_groups.append(pd)
        kernel_pd = GPy.kern.RBF(input_dim=len(pd), active_dims=pd, ARD=False)
        kernel_pd.lengthscale.constrain_fixed(true_lengthscale)
        kernel_pd.variance.constrain_fixed(true_var)
        true_kernel.append(kernel_pd)
    true_kernel = GPy.kern.Add(true_kernel)

    x = np.linspace(bounds[0][0], bounds[0][1], 1000)
    scale = 6
    gtimer = timer(nruns*(nacqs+1))
    for nn in range(nruns):
        seed_nn = seed + nn
        RandomState = np.random.RandomState(seed_nn)
        ## definition of the observations
        #X_true = np.ones((5, num_dims))
        #X_true[0] *= 0
        #X_true[1] *= 0.5
        #X_true[2] *= -0.5
        #X_true[3] *= 1
        #X_true[4] *= -1
        #f_true_no_mean = np.zeros(5)
        xmin = np.full(num_dims, np.inf)
        while not np.all(np.logical_and(-0.5 <= xmin, xmin <= 0.5)):
            X_true = sobol_seq.i4_sobol_generate(num_dims, true_n_obs)
            #sequencer = ghalton.Halton(num_dims)
            #X_true = np.array(sequencer.get(true_n_obs))
            X_true = (maxes - mins)[None,:] * X_true + mins[None,:]
            K_true = true_kernel.K(X_true) + 1e-8 * np.eye(true_n_obs)
            L_true = np.linalg.cholesky(K_true)
            f_true_no_mean = np.dot(L_true, RandomState.randn(true_n_obs))
            ##noise = np.sqrt(lik_noise_var) * RandomState.randn(true_n_obs)
            # definition of mean_function
            mean_function = QPoly(num_dims, 1, seed=seed_nn)#QPoly(num_dims, 1, seed=seed)
            mean_function.A.constrain_fixed(np.ones(num_dims)[:, None])
            mean_function.B.constrain_fixed(np.zeros(num_dims)[:, None])
            mean_function.C.constrain_fixed(0)
            f_true = f_true_no_mean + mean_function.f(X_true).flatten()
            ##Y_obs  = f_true + noise

            true_f = GPy.models.GPRegression(X_true, f_true[:,None], true_kernel,
                                                 noise_var=lik_noise_var,
                                                 mean_function=mean_function.copy())

            st = time.time()
            xmin_full = min_gp(true_f, bounds=bounds, additive=False, seed=seed_nn)
            time_full = time.time() - st
            st = time.time()
            xmin_add = min_gp(true_f, bounds=bounds, additive=True, seed=seed_nn)
            time_add = time.time() - st
            #st = time.time()
            #xmin_adddir = min_gp(true_model, bounds=bounds, additive=True, seed=seed, minimizer="direct")
            #time_adddir = time.time() - st
            xmin = xmin_add
            print("OPTIMIZATION RESULTS")
            for t, mmm in zip([time_full, time_add], [xmin_full, xmin_add]):
                print("time: {}".format(t))
                print("xmin: {}".format(mmm))
                print("fmin: {}".format(pred_noiseless(mmm, true_f)[0]))
                print("")

        true_model = TrueModel(num_dims, fm=lambda x, **kw: true_f.predict_noiseless(x.reshape(-1, num_dims), **kw)[0].ravel(),
                       fv=lambda x, **kw: np.zeros(len(x.reshape(-1, num_dims))),
                       gm=lambda x: true_f.predictive_gradients(x.reshape(-1, num_dims))[0],
                       gv=lambda x, **kw: np.zeros((1, x.size)),
                       lik_noise_var=lik_noise_var, xmin=xmin,
                       seed=seed_nn)

        initial_kernel = GPy.kern.RBF(input_dim=num_dims, active_dims=range(num_dims),
                                      ARD=False)
        initial_kernel.lengthscale.constrain_fixed(true_lengthscale)
        initial_kernel.variance.constrain_fixed(true_var)

        if run_full_dependence:
            kws = {"kernel_update":False}
        else:
            kws = {"kernel_update":True, "kernel_sampler":MHNO,
                   "kernel_sampler_kws":{"ARD":False, "copy_kernels":False}}

        #x = np.linspace(bounds[0][0], bounds[0][1], 1000)
        #plt.plot(x, true_model.true_f(x))
        #plt.show()
        #continue
        # asdasd

        np.random.seed(seed_nn)
        bolfi = BOLFI(f=true_model.noisy_f, kernel_list=[initial_kernel], bounds=bounds, acq_f=LCBSCADD, prior_f=None,
                      initial_n=5, mean_function=mean_function.copy(), lik_noise_var=lik_noise_var,
                      fix_lik_noise_var=True, lengthscale=true_lengthscale, variance=true_var,
                      optimizer=None, messages=False, max_iters=1000,
                      logging=False, seed=seed_nn, n_update=1,
                      acq_f_kws={"n_inits":10, "noise_var":acq_noise}, 
                      **kws) # compute regret xopt=true_model.xmin, f_true=true_model.true_f

        # Define true posterior function (prior: uniform)
        threshold = true_model.threshold
        post_true_un = lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold, x, true_model, bounds))
        post_true, Z_true = normalize(post_true_un, bounds=bounds[0], return_Z=True)
        print(threshold)

        for ii in range(nacqs+1):
            if ii > 0:
                bolfi.update_next_point(n=1, kernel_n_iters=100)
            print("="*20)
            print("total: {}, eta: {}".format(*gtimer.update()))
            print("{}: {} - {}".format(nn+1, ii, bolfi.curr_model.log_likelihood(),
                  [kk.active_dims for kk in bolfi.curr_model.kern.parts]))
            # Define aprx. posterior function (prior: uniform)
            ##bolfi.curr_model.predictive_gradients(np.atleast_2d(xmin_add))
            threshold_aprx = threshold#bolfi.expected_fcurr
            post_aprx_un = lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold_aprx, x))
            post_aprx, Z_aprx = normalize(post_aprx_un, bounds=bounds[0], return_Z=True)
            qlik = lambda x, q: np.exp(bolfi.unnormalized_loglikelihood_quantiles(threshold_aprx, x, quantiles=(q,))[0])
            qpost_lb = lambda x: qlik(x, 0.025) / Z_aprx
            qpost_ub = lambda x: qlik(x, 0.975) / Z_aprx
            kl_dists[nn, ii] = dist(post_true, post_aprx, bounds=bounds[0], measure="kl")
            tv_dists[nn, ii] = dist(post_true, post_aprx, bounds=bounds[0], measure="tv")
            if nn == 0:
                # Make figures
                fig = plt.figure(figsize=(12,8))
                ax1 = plt.subplot(211)
                ax1.set_title("Iteration {}".format(ii), fontsize=16)
                ax1.plot(x, pred_noiseless(x[:,None], true_model)[0].ravel(), "r-", linewidth=2, label="True Discrepancy")
                ax1.plot(x, pred_noiseless(x[:,None], bolfi.curr_model)[0].ravel(), "b--", linewidth=2, label="Aprx. Discrepancy Mean")
                ax1.plot(bolfi.X, bolfi.y, "+", color="navy", markersize=10, label="Acquisitions")
                d_quantiles = bolfi.curr_model.predict_quantiles(x[:,None], quantiles=(2.5, 97.5))
                ax1.fill_between(x, d_quantiles[0].ravel(), d_quantiles[1].ravel(), alpha=0.4, label="Aprx. Discrepancy 95% CI")
                #ax1.axhline(y=threshold, color='orangered', linestyle='--', label="Threshold ({0:.3f})".format(threshold))
                # Plot posteriors (scaled)
                ax1.plot(x, scale * post_true(x), "--", color="brown", linewidth=1.5,  label="True Posterior (rescaled)")
                ax1.plot(x, scale * post_aprx(x), "k--", linewidth=1.5, label="Aprx. Posterior Mean (rescaled)")
                ax1.fill_between(x, scale * qpost_lb(x), scale * qpost_ub(x), color="black", alpha=0.2, label="Aprx. Posterior 95% CI")
                # Set plot properties
                ax1.grid()
                ax1.set_xlim((bounds[0][0], bounds[0][1]))
                ax1.set_ylim((-8, 25))
                ax1.set_xlabel(r"$\theta$", size=14)
                plt.xticks(size=12)
                #ax1.set_ylabel(r"Discrepancy  $\Delta_{\theta}$", size=14)
                plt.yticks(size=12)
                ax1.legend()
                ax2 = plt.subplot(223)
                ax2.set_xlim((0, nacqs))
                #ax2.set_yscale("log")
                #ax2.set_ylim((1e-4, 1e0))
                ax2.set_ylim((0, 0.05))
                ax2.set_yticks(np.arange(0, 0.1+0.01, 0.01))
                ax2.set_ylabel("KL Divergence", size=14)
                ax2.set_xlabel("Iteration", size=14)
                ax2.plot(kl_dists[nn, :ii+1], "ro--", markersize=3, label="KL Divergence")
                ax2.grid()
                ax3 = plt.subplot(224)
                ax3.set_xlim((0, nacqs))
                #ax3.set_yscale("log")
                #ax3.set_ylim((1e-4, 1e0))
                ax3.set_ylim((0, 0.15))
                ax3.set_yticks(np.arange(0, 0.15+0.015, 0.015))
                ax3.set_ylabel("TV Distance", size=14)
                ax3.set_xlabel("Iteration", size=14)
                ax3.plot(tv_dists[nn, :ii+1], "go--", markersize=3, label="TV Distance")
                ax3.grid()
                #plt.show()
                plt.savefig(os.path.join(figpath, "{0:03d}_{1:02d}.pdf".format(ii, nn)))
                plt.close()

        for k, v in zip(["kl", "tv", "times"], [kl_dists, tv_dists, np.array(gtimer.times)]):
            np.savetxt(os.path.join(respath, "{}_acqnoise_{}.csv".format(k, acq_noise)), v, delimiter=",")












#===========================================================================================================
#
#
#    loglik_hat = np.exp(bolfi.unnormalized_loglikelihood(threshold, x[:,None]))
#    Z_hat = integrate.quad(lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold, x)), -3, 3)[0]
#    #Z_hat = 1
#    loglik_hat_lb = np.exp(bolfi.unnormalized_loglikelihood_quantiles(threshold, x[:,None], quantiles=(0.025,))[0])
#    loglik_hat_ub = np.exp(bolfi.unnormalized_loglikelihood_quantiles(threshold, x[:,None], quantiles=(0.975,))[0])
#    ax.plot(x, scale * loglik_hat / Z_hat, "k--", linewidth=1.5, label="Aprx. Posterior Mean (rescaled)")
#    ax.fill_between(x, scale * loglik_hat_lb / Z_hat, scale * loglik_hat_ub / Z_hat, color="black", alpha=0.2, label="Aprx. Posterior 95% CI")
#
#    loglik_true = np.exp(bolfi.unnormalized_loglikelihood(threshold, x[:,None], true_model, bounds))
#    Z_true = integrate.quad(lambda x: np.exp(bolfi.unnormalized_loglikelihood(threshold, x, true_model, bounds)), -3, 3)[0]
#    #Z_true = Z_hat
#    ax.plot(x, scale * loglik_true / Z_true, "--", color="brown", linewidth=1.5, label="True Posterior (rescaled)")
#    plt.grid()
#    ax.set_xlim((-3, 3))
#    ax.set_ylim((-8, 25))
#    ax.set_xlabel(r"Parameter  $\theta$", size=14)
#    plt.xticks(size=12)
#    ax.set_ylabel(r"Discrepancy  $\Delta_{\theta}$", size=14)
#    plt.yticks(size=12)
#    plt.legend()
#    plt.savefig("simple1D/{0:03d}.png".format(ii))
#    plt.close()




#with open('model.obj', 'wb') as fp:
#     pickle.dump(bolfi, fp)
#
#with open('model.obj', 'rb') as fp:
#     bolfi = pickle.load(fp)
#
#print(bounds)



#x = np.linspace(-3, 3, 1000)
#plt.figure()
#
#plt.plot(x, true_model.predict_noiseless(x[:,None])[0].ravel(), "b-")
#plt.plot(xmin_full, true_model.predict_noiseless(np.atleast_2d(xmin_full))[0].ravel(), "ro")
#qs = true_model.predict_quantiles(x[:,None], quantiles=(1, 99))
#plt.fill_between(x, qs[0].ravel(), qs[1].ravel(), alpha=0.4)
#mean =  true_model.predict_noiseless(x[:,None])[0].ravel()
#plt.plot(x, mean-3*np.sqrt(lik_noise_var), "r--")
#plt.plot(x, mean+3*np.sqrt(lik_noise_var), "k--")
#plt.plot(xmin_full, threshold, "go")
#
#import scipy.stats as ss
#arr = (threshold - mean) / np.sqrt(lik_noise_var)
#print(arr)
#print("OK!")
#print(threshold)
#plt.plot(x, ss.norm.cdf(arr), "--", color="brown")
#plt.show()
#
#asdasd


#print(true_model.predict_quantiles(np.atleast_2d(xmin_full), quantiles=(0.005,0.995)))

#xmin_hat = eval_gp(bolfi.curr_model, bounds=bounds)
#print("xmin hat: {}".format(xmin_hat))
#print(threshold)



#bolfi.curr_model

#bolfi.curr_model.to_di
#bolfi.

#=============================================================================
# Test performance of optimizers with default settings



#res = eval_gp(true_model, additive=True, X=x_point)
#val = 0
#for k, v in res.items():
#    val += v[0][0]
#print(val)
#res1 = eval_gp(true_model, additive=False, X=x_point)
#for k, v in res1.items():
#    print(v[0][0])

#res = eval_gp(true_model, additive=True, X=x_point)
#val = 0
##val = np.zeros(true_model.input_dim)
#for k, v in res.items():
#    #val[np.array(json.loads(k))] = v[1][0][0]
#    val += v[0][0]
#print(val)
#print("adsasd")
#res1 = eval_gp2(true_model, additive=True, X=x_point)
##val1 = np.zeros(true_model.input_dim)
#cc = 0
#for k, v in res1.items():
#    #val1[np.array(json.loads(k))] = v[1][0][0]
#    cc += v[0][0]
#print(cc)
#print("adsasd")


#print(aa)
#print(pred_noiseless_old(x_point))
#print(pred_noiseless(x_point, true_model)[0])
#res = eval_gp(true_model, additive=False, X=x_point)
#for k, v in res.items():
#    print(v[0][0])




    #c = minimize(lambda x: pred_noiseless_d(x),
    #                      np.array(bounds)[active_dims],
    #                      lambda x: pred_grad_d(x), None, 50, 10000,
    #                      random_state=np.random.RandomState(seed))[0] # l
    #x_min5[active_dims] = c

#print(time.time()-start)
#print("asdasdad")
#print(x_min5)
#print(pred_noiseless(x_min5))
#print(x_min)
#print(pred_noiseless(x_min))
#print(x_min2)
#print(pred_noiseless(x_min2))
#print(x_min3)
#print(pred_noiseless(x_min3))

#print(time.time()-start)
#print("=======")
#print(x_min5)
#print(pred_noiseless(x_min5))


#print(x_min)
#print(x_min2)


#print(np.min(pred_noiseless(X_true)))
#print(pred_noiseless(x_min))


#print(x_min)


#true_model.predictive_gradients(x)
#print(bounds)
# xhat, _ = minimize(
#            obj,
#            self.model.bounds,
#            grad_obj,
#            self.prior,
#            self.n_inits,
#            self.max_opt_iters,
#            random_state=self.random_state)

#print(X_true)

#plt.plot(X_true, f_true, 'ro', markersize=1)
#plt.plot(X_true, Y_obs, "go", markersize=1)
##plt.plot(X_true, Y_hat, "bo", markersize=1)
#plt.show()


#plt.show()
#true_model = GPy.models.GPRegression(X_true, np.atleast_2d(f_true), true_kernel,
#                                     noise_var=1.0)

#func = lambda X: true_model.predict(np.atleast_2d(X))
#noisy_func = lambda X: true_model.predict(np.atleast_2d(X)) + \
#                       np.sqrt(lik_noise_var) * RandomState.randn(len(np.atleast_2d(X)))

#print(func(X_true[0]))
#print(f_true[0])

#model = GPy.models.GPRegression(self.X, self.y,
#                                        GPy.kern.Add(kernel_list),
#                                        noise_var=self.lik_nv,
#                                        mean_function=mean_function)


#true_kernel = GPy.kern.Add(true_kernel)
#print(true_kernel)


#print(true_groups)

#func = lambda x: x.dot(A).dot(x)
#global_opt = np.zeros(num_dims)
#noisy_func = lambda x: func(x) + RandomState.normal(scale=np.sqrt(lik_noise_var))
#mins = np.full(num_dims, -3)
#maxes = np.full(num_dims, 3)
#bounds = [_ for _ in zip(*[mins, maxes])]
#
#initial_kernel = GPy.kern.RBF(input_dim=num_dims, active_dims=range(num_dims),
#                              ARD=False)
#initial_kernel.lengthscale.constrain_fixed(lengthscale)
#initial_kernel.variance.constrain_fixed(variance)
#
#run_full_dependence = False
#if run_full_dependence:
#    kws = {"kernel_update":False}
#else:
#    kws = {"kernel_update":True, "kernel_sampler":MHNO,
#           "kernel_sampler_kws":{"ARD":False, "copy_kernels":False}}
#
#bolfi = BOLFI(f=noisy_func, kernel_list=[initial_kernel], bounds=bounds, acq_f=LCBSC, prior_f=None,
#              initial_n=20, mean_function=mean_function, lik_noise_var=lik_noise_var,
#              fix_lik_noise_var=True, lengthscale=lengthscale, variance=variance,
#              optimizer=None, messages=False, max_iters=1000,
#              logging=False, seed=seed, n_update=20,
#              acq_f_kws={"n_inits":10, "noise_var":0.0}, **kws)
#
#for ii in range(500):
#    bolfi.update_next_point(n=1, kernel_n_iters=1000)
#    print("{}: {} - {}".format(ii, bolfi.curr_model.log_likelihood(),
#          [kk.active_dims for kk in bolfi.curr_model.kern.parts]))


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
