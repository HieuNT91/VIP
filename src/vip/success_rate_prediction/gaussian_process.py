import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_triangular
import json
import os
from datetime import datetime
import argparse 

def kernel_rbf(d, length_scale=1):
    return np.exp(-0.5 * (d / length_scale) ** 2)

def kernel_rbf_median(d):
    length_scale = np.median(d[np.triu_indices_from(d,k=1)])
    return np.exp(-0.5 * (d / length_scale) ** 2)

def kernel_self_tuning(d, k=7, eps=1e-12, square=False):
    """
    Zelnik-Manor & Perona (2004) self-tuning kernel:
      K_ij = exp(-dist_ij^p / (sigma_i * sigma_j)), p in {1,2}
    sigma_i is distance to the k-th nearest neighbor of i (excluding self, ignoring zeros).
    """
    n = d.shape[0]
    sigmas = np.empty(n, dtype=float)
    for i in range(n):
        row = d[i]
        nz = np.sort(row[row > 0])  # exclude zeros (self and duplicates)
        if nz.size == 0:
            sigmas[i] = 1.0
        else:
            idx = min(k - 1, nz.size - 1)
            sigmas[i] = nz[idx]
    sigmas = np.maximum(sigmas, eps)
    denom = sigmas[:, None] * sigmas[None, :] + eps
    if square:
        K = np.exp(-(d ** 2) / denom)
    else:
        K = np.exp(-(d) / denom)
    np.fill_diagonal(K, 1.0)
    return K

class GPR:
    def __init__(self, 
                distance_matrix, 
                reuse_covariance=False,
                reuse_mean=False,
                return_std=False, 
                qid_to_idx=None, 
                prior_value=-1,
                length_scale=0.27
                 ):
        self.distance_matrix = distance_matrix
        # self.covariance_matrix = kernel_rbf_median(distance_matrix)
        self.covariance_matrix = kernel_rbf(distance_matrix, length_scale)
        # self.covariance_matrix = kernel_rbf(distance_matrix, 0.27)
        self.mean = np.zeros(self.covariance_matrix.shape[0]) + prior_value
        self.reuse_covariance = reuse_covariance
        self.reuse_mean = reuse_mean
        self.return_std = return_std
        self.qid_to_idx = qid_to_idx
        self.prior_value = prior_value
        
    def _logit(self, p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))
    
    def _sigmoid(self, f):
        return 1.0 / (1.0 + np.exp(-f))
    
    def fit(self, train_indices, observations):
        if not self.reuse_mean:
            self.mean = np.zeros(self.covariance_matrix.shape[0]) + self.prior_value
        g_t = self._logit(np.clip(observations, a_max=1-1e-6, a_min=1e-6))
        K_in_in = self.covariance_matrix[np.ix_(train_indices, train_indices)]

        L = cholesky(K_in_in + 1e-4 * np.eye(K_in_in.shape[0]), lower=True, check_finite=False)
        alpha = cho_solve((L, True), g_t - self.mean[train_indices], check_finite=False)

        
        all_indices = list(range(self.covariance_matrix.shape[0]))
        K_in_new = self.covariance_matrix[np.ix_(train_indices, all_indices)]
        K_new_in = K_in_new.T
        
        # ALGO: update posterior 
        self.mean[all_indices] = self.mean[all_indices] + K_new_in @ alpha
        self.mean[train_indices] = g_t
        if self.return_std or self.reuse_covariance:
            K_in_in_inv = np.linalg.inv(K_in_in + 1e-4 * np.eye(len(train_indices)))
            V = K_in_in_inv @ K_in_new
            self.std = np.sqrt(np.diag(self.covariance_matrix[np.ix_(all_indices, all_indices)] - K_new_in @ V))
        
        if self.reuse_covariance:
            self.covariance_matrix[np.ix_(all_indices, all_indices)] -= K_new_in @ V

    def fit_qids(self, qids, observations):
        if self.qid_to_idx is None:
            raise ValueError("qid_to_idx mapping is not provided.")
        train_indices = [self.qid_to_idx[qid] for qid in qids if qid in self.qid_to_idx]
        self.fit(train_indices, observations)
        
    def predict(self, indices):
        mean_pred = self._sigmoid(self.mean[indices])
        if self.return_std:
            return mean_pred, self.std[indices]
        else:
            return mean_pred, None
    
    def predict_qids(self, qids):
        if self.qid_to_idx is None:
            raise ValueError("qid_to_idx mapping is not provided.")
        indices = [self.qid_to_idx[qid] for qid in qids if qid in self.qid_to_idx]
        return self.predict(indices)

def acc_to_var(acc):
    return acc * (1 - acc)
