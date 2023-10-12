# Function for CSGD hybrid
# 03/18/2023


import scipy.stats as st
import numpy as np
from scipy.special import beta
import math
import scipy as sp


def fit_regression_v2(obs_prcp, covars, p_clim, initguess=False, constrain=False, mu_fix=False, sigma_fix=False):
    # Fitting function for CSGD regression
    # 03/18/2023

    # Initial guess of regression parameters: when satellite obs. equals zero;
    # zeroind = np.where(np.abs(large_scale_prcp) < 0.01) # return an array of true and false, true if outarr elemnt is zero
    zeromean = np.nanmean(obs_prcp)  # select climatological data where the outarr element is zero and compute the mean
    zerosig = np.nanstd(
        obs_prcp)  # select climatological data where the outarr element is zero and compute the standard deviation

    # want to know alpha1 and alpha2
    al1al2bnds = ((0.0001, 1.0), (0.0001, 10.0))  # set up bounds for alpha1 and alpha2
    al1al2 = [0.01, zeromean / p_clim[0]]  # set up initial guess for alpha1 and alpha2

    # define an objective function, which is to minimize the difference between
    def estimal1al2(params, zeromean, climmean):
        return np.power(zeromean - climmean / params[0] * np.log(1. + params[1] * (np.exp(params[0]) - 1.)), 2)

    # run the optimization code
    optimal1al2 = sp.optimize.minimize(estimal1al2, al1al2, args=(zeromean, p_clim[0]), method='L-BFGS-B',
                                       bounds=al1al2bnds)
    alpha1 = optimal1al2.x[0]  # retrieve optimized alpha1
    alpha2 = optimal1al2.x[1]  # retrieve optimized alpha2
    # Compute alpha4 using zerosig, and the distribution mean
    alpha3 = zerosig / p_clim[1] / np.sqrt(np.log(1. + alpha2 * (np.exp(alpha1) - 1.)) / alpha1)

    # Set up initial guess
    if initguess == False:
        pst = np.array([alpha1, alpha2, alpha3])  # set the first 3 coefficients

        if covars.shape[1] != 0:
            # set up initial guess for covariates coefficients
            cov_coeff_guess = np.zeros(covars.shape[1])
            # combine them together
            pst = np.hstack([pst, cov_coeff_guess])
        # pst=[0.001, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0]
    else:
        pst = initguess

    # Set up contrains
    if constrain == False:
        # set constrains for alpha1, alpha2, alpha3
        bnds = [(0.0001, 10), (0.0001, 10.0), (0.0001, 10.0)]

        # if there are more covariates
        if covars.shape[1] != 0:
            # set up constrains
            cov_bnds = [(0.0, 5.0)] * covars.shape[1]
            # combine them together
            bnds += cov_bnds

    else:
        bnds = constrain

    if mu_fix == True:
        # set initial guess related to mu to 0
        pst[0:2] = 0
        pst[3:] = 0

        # set contrain related to mu to (0, 0)
        bnds[0:2] = [(0.0, 0.0)] * 2

        if covars.shape[1] != 0:
            bnds[3:] = [(0.0, 0.0)] * covars.shape[1]

    if sigma_fix == True:
        pst[2] = 0
        bnds[2] = (0.0, 0.0)

    # print(bnds)
    # run the optimization code
    output = sp.optimize.minimize(crps_regression_v2, pst,
                                  args=(covars, obs_prcp, p_clim, mu_fix, sigma_fix),
                                  method='L-BFGS-B', options={'disp': False}, bounds=bnds, )

    return output


def crps_regression_v2(par, covars, obs_prcp, p_clim, mu_fix=False, sigma_fix=False):
    # Objective function for CRPS regression
    # 03/18/2023

    # retrieve climatological distribution parameters
    mu_clim = p_clim[0]
    sigma_clim = p_clim[1]
    delta_clim = p_clim[2]

    # coefficients order in par: alpha1, alpha2, alpha3, alpha4, alpha5, alpha6...
    # corresponds to p[0], p[1], p[2], p[3]....

    # alpha1, alpha2: for mu
    # alpha3: for sigma
    # alpha4+: for remaining covariates in mu

    # if mu is not fixed:
    if mu_fix == False:
        # check if it has covariates
        if (covars.shape[1]) != 0:
            # compute the covariates mean
            covmean = np.mean(covars, axis=0)

            # logarg = par[1] + np.sum(par[3:] * (covars / covmean), axis=1)
            logarg = par[1] + np.sum(par[3:] * covars, axis=1) # not using climate mean as base

        else:
            # Compute with no-covariate coefficient only
            logarg = par[1]

        # Compute the mu
        mu = mu_clim / par[0] * np.log1p(np.expm1(par[0]) * logarg)

    else:
        mu = mu_clim

    # if sigma is not fixed
    if sigma_fix == False:
        sigma = par[2] * sigma_clim * np.sqrt(mu / mu_clim)

    else:
        sigma = sigma_clim

    # CRPS-based Minimization Function
    delta = delta_clim  # use climate delta as CSGD detla
    k = np.power(mu / sigma, 2)  # compute the shape parameter
    theta = np.power(sigma, 2) / mu  # compute the scale parameter
    betaf = beta(0.5, 0.5 + k)
    ysq = (obs_prcp - delta) / theta  # use stageIV to rescale
    csq = -delta / theta  # when y = 0
    Fysq = sp.stats.gamma.cdf(ysq, k, scale=1)
    Fcsq = sp.stats.gamma.cdf(csq, k, scale=1)
    FysqkP1 = sp.stats.gamma.cdf(ysq, k + 1, scale=1)
    FcsqkP1 = sp.stats.gamma.cdf(csq, k + 1, scale=1)
    Fcsq2k = sp.stats.gamma.cdf(2 * csq, 2 * k, scale=1)

    # calculate the crps metrics
    crps = ysq * (2. * Fysq - 1.) - csq * np.power(Fcsq, 2) + k * (
            1. + 2. * Fcsq * FcsqkP1 - np.power(Fcsq, 2) - 2. * FysqkP1) - k / math.pi * betaf * (1. - Fcsq2k)

    return 10000. * np.nanmean(theta * crps)