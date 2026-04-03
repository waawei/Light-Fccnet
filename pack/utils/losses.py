"""Losses for Light-FCCNet reproduction."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .density_scale import scale_density_target


def _resolve_target_count(
    pred_density: torch.Tensor,
    gt_density: torch.Tensor,
    gt_count: torch.Tensor | None,
    density_scale: float,
) -> torch.Tensor:
    if gt_count is None:
        return gt_density.sum(dim=(1, 2, 3))
    gt_count = gt_count.to(device=pred_density.device, dtype=pred_density.dtype).reshape(-1)
    return scale_density_target(gt_count, density_scale)


class BaselineCountingLoss(nn.Module):
    """Baseline counting loss: density MSE plus count MSE."""

    def __init__(self, density_scale: float = 1.0):
        super().__init__()
        self.density_scale = float(density_scale)
        self.mse = nn.MSELoss()

    def _count_loss(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_count = pred_density.sum(dim=(1, 2, 3))
        gt_count = _resolve_target_count(pred_density, gt_density, gt_count, self.density_scale)
        return torch.mean((pred_count - gt_count) ** 2)

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ):
        gt_density = scale_density_target(gt_density, self.density_scale)
        l2 = self.mse(pred_density, gt_density)
        count = self._count_loss(pred_density, gt_density, gt_count=gt_count)
        total = l2 + count
        return total, {"loss": total, "l2": l2, "count": count}


class LightFCCLoss(nn.Module):
    """
    L_FCC = (1 - alpha) * (L2 + Lc) + alpha * Ls
    where:
      L2 : pixel-wise MSE
      Lc : count MSE between density sums
      Ls : 1 - SSIM
    """

    def __init__(
        self,
        alpha: float = 0.1,
        density_scale: float = 1.0,
        ssim_window: int = 11,
        ssim_c: float = 1e-4,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.density_scale = float(density_scale)
        self.ssim_window = int(ssim_window)
        self.ssim_c = float(ssim_c)
        self.mse = nn.MSELoss()

    def _count_loss(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pred_count = pred_density.sum(dim=(1, 2, 3))
        gt_count = _resolve_target_count(pred_density, gt_density, gt_count, self.density_scale)
        return torch.mean((pred_count - gt_count) ** 2)

    def _ssim_loss(self, pred_density: torch.Tensor, gt_density: torch.Tensor) -> torch.Tensor:
        window = max(3, self.ssim_window)
        if window % 2 == 0:
            window += 1
        mu_x = F.avg_pool2d(pred_density, kernel_size=window, stride=1, padding=window // 2)
        mu_y = F.avg_pool2d(gt_density, kernel_size=window, stride=1, padding=window // 2)
        sigma_x = F.avg_pool2d(pred_density * pred_density, kernel_size=window, stride=1, padding=window // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(gt_density * gt_density, kernel_size=window, stride=1, padding=window // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred_density * gt_density, kernel_size=window, stride=1, padding=window // 2) - mu_x * mu_y

        ssim_num = (2 * mu_x * mu_y + self.ssim_c) * (2 * sigma_xy + self.ssim_c)
        ssim_den = (mu_x.pow(2) + mu_y.pow(2) + self.ssim_c) * (sigma_x + sigma_y + self.ssim_c)
        ssim_map = ssim_num / (ssim_den + 1e-8)
        return 1.0 - ssim_map.mean()

    def forward(
        self,
        pred_density: torch.Tensor,
        gt_density: torch.Tensor,
        gt_count: torch.Tensor | None = None,
    ):
        gt_density = scale_density_target(gt_density, self.density_scale)
        l2 = self.mse(pred_density, gt_density)
        count = self._count_loss(pred_density, gt_density, gt_count=gt_count)
        ssim = self._ssim_loss(pred_density, gt_density)
        total = (1.0 - self.alpha) * (l2 + count) + self.alpha * ssim
        return total, {"loss": total, "l2": l2, "count": count, "ssim": ssim}
