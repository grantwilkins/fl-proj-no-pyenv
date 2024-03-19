# python3.11.6
"""Implementation of the Gauss Newton Optimiser."""

import torch
from collections.abc import Iterable
import numpy as np


class DGN(torch.optim.Optimizer):
    """Diagonal Gauss Newton Torch Optimiser."""

    def __init__(
        self, parameters: Iterable, step_size: float, damping: float, mc: bool
    ) -> None:
        super().__init__(parameters, {"step_size": step_size, "damping": damping})
        if mc:
            self.ggn_field = "diag_ggn_mc"
        else:
            self.ggn_field = "diag_ggn_exact"

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                ggn = getattr(p, self.ggn_field)
                step_direction = p.grad / (ggn + group["damping"])
                p.data.add_(step_direction, alpha=-group["step_size"])
                # print(p.diag_ggn_mc.shape)
                # print(p.grad.shape)
                # print(step_direction.shape)
                # if len(p.diag_ggn_mc.shape) == 1:
                #     plt.imshow(
                #         np.diag(p.diag_ggn_mc.cpu().detach().numpy()), cmap="gray"
                #     )
                #     plt.axis("off")
                #     plt.savefig("diag.pdf", bbox_inches="tight")
                #     plt.clf()
                #     plt.imshow(
                #         np.diag(p.grad.cpu().detach().numpy()),
                #         cmap="gray",
                #     )
                #     plt.axis("off")
                #     plt.savefig("gradient.pdf", bbox_inches="tight")
                #     plt.clf()
                #     plt.imshow(
                #         np.diag(step_direction.cpu().detach().numpy()), cmap="gray"
                #     )
                #     plt.axis("off")
                #     plt.savefig("step_dir.pdf", bbox_inches="tight")
                #     plt.show()


class BDGN(torch.optim.Optimizer):
    """Block Diagonal Gauss Newton Torch Optimiser."""

    def __init__(
        self, parameters: Iterable, step_size: float, damping: float, mc: bool
    ) -> None:
        if mc:
            self.ggn_field = "kfac"
        else:
            self.ggn_field = "kflr"
        super().__init__(parameters, {"step_size": step_size, "damping": damping})

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            for p in group["params"]:
                factors = getattr(p, self.ggn_field)
                if len(factors) == 2:  # noqa:PLR2004
                    q, g = factors
                    k = group["damping"]
                    iq = torch.eye(q.shape[0], device=q.device)
                    ig = torch.eye(g.shape[0], device=g.device)
                    # w = torch.sqrt(
                    #     torch.norm(q.cpu(), p="nuc") / torch.norm(g.cpu(), p="nuc")
                    # ).to(device="mps")
                    # print(w)
                    w = 0.1
                    orig_shape = p.grad.shape
                    left = torch.inverse(q + (w * np.sqrt(k) + iq))
                    right = torch.inverse((g + (w**-1 * np.sqrt(k) * ig)).cpu()).to(
                        device="mps"
                    )
                    if len(p.grad.shape) == 4:  # noqa: PLR2004
                        step_direction = left @ p.grad.flatten(1) @ right
                    elif len(p.grad.shape) in {1, 2}:
                        step_direction = left @ p.grad @ right
                    else:
                        raise ValueError()
                    p.data.add_(
                        torch.reshape(step_direction, orig_shape),
                        alpha=-group["step_size"],
                    )
                elif len(factors) == 1:
                    c = factors[0]
                    k = group["damping"]
                    step_direction = (
                        torch.inverse(c + (k * torch.eye(c.shape[0], device=c.device)))
                        @ p.grad
                    )
                    p.data.add_(
                        step_direction,
                        alpha=-group["step_size"],
                    )
                else:
                    raise ValueError()
