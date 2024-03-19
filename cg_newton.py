# python3.11.6
"""Implementation of the Conjugate Gradient Optimiser."""

import torch
import math
from collections.abc import Iterable, Callable
from backpack.extensions.backprop_extension import BackpropExtension


class CGN(torch.optim.Optimizer):
    """Conjugate Gradient Torch Optimiser."""

    def __init__(
        self,
        parameters: Iterable,
        bp_extension: BackpropExtension,
        lr: float = 0.1,
        damping: float = 1e-2,
        maxiter: int = 100,
        tol: float = 1e-1,
        atol: float = 1e-8,
    ) -> None:
        super().__init__(
            parameters,
            {
                "lr": lr,
                "damping": damping,
                "maxiter": maxiter,
                "tol": tol,
                "atol": atol,
                "savefield": bp_extension.savefield,
            },
        )
        self.bp_extension = bp_extension

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                damped_curvature = self.damped_matvec(
                    p, group["damping"], group["savefield"]
                )
                direction, _ = self.cg(
                    damped_curvature,
                    -p.grad.data,
                    maxiter=group["maxiter"],
                    tol=group["tol"],
                    atol=group["atol"],
                )

                p.data.add_(direction, alpha=group["lr"])

    def damped_matvec(
        self, param: Iterable, damping: float, savefield: str
    ) -> torch.Tensor:
        """Get damped matvec."""
        curvprod_fn = getattr(param, savefield)

        def matvec(v: torch.Tensor) -> torch.Tensor:
            v = v.unsqueeze(0)
            result = damping * v + curvprod_fn(v)
            return result.squeeze(0)

        return matvec

    @staticmethod
    def cg(
        a: Callable,
        b: torch.Tensor,
        x0: torch.Tensor = None,
        maxiter: int = -1,
        tol: float = 1e-5,
        atol: float = 1e-8,
    ) -> tuple[torch.Tensor, int]:
        r"""Solve :math:`ax = b` for :math:`x` using conjugate gradient.

        The interface is similar to CG provided by :code:`scipy.sparse.linalg.cg`.

        The main iteration loop follows the pseudo code from Wikipedia:
            https://en.wikipedia.org/w/index.php?title=Conjugate_gradient_method&oldid=855450922

        Parameters
        ----------
        a : function
            Function implementing matrix-vector multiplication by `A`.
        b : torch.Tensor
            Right-hand side of the linear system.
        x0 : torch.Tensor
            Initialization estimate.
        atol: float
            Absolute tolerance to accept convergence. Stop if
            :math:`|| A x - b || <` `atol`
        tol: float
            Relative tolerance to accept convergence. Stop if
            :math:`|| A x - b || / || b || <` `tol`.
        maxiter: int
            Maximum number of iterations.

        Returns
        -------
        x (torch.Tensor): Approximate solution :math:`x` of the linear system
        info (int): Provides convergence information, if CG converges info
                    corresponds to numiter, otherwise info is set to zero.
        """
        maxiter = b.numel() if maxiter == -1 else min(maxiter, b.numel())
        x = torch.zeros_like(b) if x0 is None else x0

        # initialize parameters
        r = (b - a(x)).detach()
        p = r.clone()
        rs_old = (r**2).sum().item()

        # stopping criterion
        norm_bound = max([tol * torch.norm(b).item(), atol])

        def converged(rs: float, numiter: int) -> tuple[bool, int]:
            """Check whether CG stops (convergence or steps exceeded)."""
            norm_converged = norm_bound > math.sqrt(rs)
            info = numiter if norm_converged else 0
            iters_exceeded = numiter > maxiter
            return (norm_converged or iters_exceeded), info

        # iterate
        iterations = 0
        while True:
            ap = a(p).detach()

            alpha = rs_old / (p * ap).sum().item()
            x.add_(p, alpha=alpha)
            r.sub_(ap, alpha=alpha)
            rs_new = (r**2).sum().item()
            iterations += 1

            stop, info = converged(rs_new, iterations)
            if stop:
                return x, info

            p.mul_(rs_new / rs_old)
            p.add_(r)
            rs_old = rs_new
