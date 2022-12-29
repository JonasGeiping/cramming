"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.

"""

import enum
import logging
from typing import Tuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class NewtonConvergenceFlag(enum.Enum):
    REACHED_MAX_ITERS = 0
    CONVERGED = 1


class RootInvMethod(enum.Enum):
    EIGEN = 0
    NEWTON = 1


def matrix_inverse_root(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    root_inv_method: RootInvMethod = RootInvMethod.EIGEN,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tensor:
    """Computes matrix root inverse.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        root_inv_method (RootInvMethod): Specifies method to use to compute root inverse. (Default: RootInvMethod.EIGEN)
        max_iterations (int): Maximum number of iterations for coupled Newton iteration. (Default: 1000)

    Returns:
        X (Tensor): Inverse root of matrix A.

    """

    if root_inv_method == RootInvMethod.EIGEN:
        X, _, _ = _matrix_root_eigen(A=A, root=root, epsilon=epsilon)
    elif root_inv_method == RootInvMethod.NEWTON:
        X, _, termination_flag, _, _ = _matrix_inverse_root_newton(
            A=A,
            root=root,
            epsilon=epsilon,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        if termination_flag == NewtonConvergenceFlag.REACHED_MAX_ITERS:
            logging.warning("Newton did not converge and reached maximum number of iterations!")
    else:
        raise NotImplementedError("Root inverse method is not implemented! Specified root inverse method is " + str(root_inv_method) + ".")

    return X


def _matrix_root_eigen(
    A: Tensor,
    root: int,
    epsilon: float = 0.0,
    inverse: bool = True,
    perturb: bool = True,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Compute matrix (inverse) root using eigendecomposition of symmetric positive (semi-)definite matrix.

            A = Q L Q^T => A^{1/r} = Q L^{1/r} Q^T OR A^{-1/r} = Q L^{-1/r} Q^T

    Assumes matrix A is symmetric.

    Args:
        A (Tensor): Square matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        inverse (bool): Returns inverse root matrix. (Default: True)
        perturb (bool): Perturbs matrix eigenvalues to ensure it is (practically) positive semi-definite. (Default: True)

    Returns:
        X (Tensor): (Inverse) root of matrix. Same dimensions as A.
        L (Tensor): Eigenvalues of A.
        Q (Tensor): Orthogonal matrix consisting of eigenvectors of A.

    """

    # check if root is positive integer
    if root <= 0:
        raise ValueError(f"Root {root} should be positive!")

    # compute matrix power
    alpha = 1 / root
    if inverse:
        alpha = -alpha

    # check if matrix is scalar
    if len(A.shape) == 0 or (len(A.shape) == 1 and A.shape[0] == 1):
        return A**alpha, A, torch.tensor(1.0)

    # check matrix shape
    if len(A.shape) != 2:
        raise ValueError("Matrix is not 2-dimensional!")
    elif A.shape[0] != A.shape[1]:
        raise ValueError("Matrix is not square!")

    # compute eigendecomposition and compute minimum eigenvalue
    L, Q = torch.linalg.eigh(A)
    lambda_min = torch.min(L)

    # perturb eigenvalues (if necessary)
    if perturb:
        L += -torch.minimum(lambda_min, torch.tensor(0.0))

    # add epsilon
    L += epsilon

    # compute inverse preconditioner
    X = Q * L.pow(alpha).unsqueeze(0) @ Q.T

    return X, L, Q


def _matrix_inverse_root_newton(
    A,
    root: int,
    epsilon: float = 0.0,
    max_iterations: int = 1000,
    tolerance: float = 1e-6,
) -> Tuple[Tensor, Tensor, NewtonConvergenceFlag, int, Tensor]:
    """Compute matrix square root using coupled inverse Newton iteration.

        alpha <- -1 / p
        X <- 1/c * I
        M <- 1/c^p * A
        repeat until convergence
            M' <- (1 - alpha) * I + alpha * M
            X <- X * M'
            M <- M'^p * M

    where c = (2 |A|_F / (p + 1))^{1/p}. This ensures that |A|_2 <= |A|_F < (p + 1) c^p, which guarantees convergence.
    We will instead use z = (p + 1) / (2 * |A|_F).

    Args:
        A (Tensor): Matrix of interest.
        root (int): Root of interest. Any natural number.
        epsilon (float): Adds epsilon * I to matrix before taking matrix root. (Default: 0.0)
        max_iterations (int): Maximum number of iterations. (Default: 1000)
        tolerance (float): Tolerance. (Default: 1e-6)

    Returns:
        A_root (Tensor): Inverse square root of matrix.
        M (Tensor): Coupled matrix.
        termination_flag (NewtonConvergenceFlag): Specifies convergence.
        iteration (int): Number of iterations.
        error (Tensor): Final error between M and I.

    """

    # initialize iteration, dimension, and alpha
    iteration = 0
    dim = A.shape[0]
    alpha = -1 / root
    identity = torch.eye(dim, dtype=A.dtype, device=A.device)

    # add regularization
    A.add_(identity, alpha=epsilon)

    # initialize matrices
    A_nrm = torch.linalg.norm(A)
    z = (root + 1) / (2 * A_nrm)
    X = z ** (-alpha) * identity
    M = z * A
    error = torch.dist(M, identity, p=torch.inf)

    # main for loop
    while error > tolerance and iteration < max_iterations:
        iteration += 1
        M_p = M.mul(alpha).add_(identity, alpha=(1 - alpha))
        X = X @ M_p
        M = torch.linalg.matrix_power(M_p, root) @ M
        error = torch.dist(M, identity, p=torch.inf)

    # determine convergence flag
    if error <= tolerance:
        termination_flag = NewtonConvergenceFlag.CONVERGED
    else:
        termination_flag = NewtonConvergenceFlag.REACHED_MAX_ITERS

    return X, M, termination_flag, iteration, error
