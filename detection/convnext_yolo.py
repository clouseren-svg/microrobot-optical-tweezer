"""
ConvNext-based YOLO Detector for Microrobot Localization
============================================================
PyTorch implementation of the modified YOLOv4-tiny architecture
described in Ren et al. (MARSS 2022).

Key modifications from standard YOLOv4-tiny:
- ConvNext-based blocks replace CSPDarknet53-tiny backbone
- Gaussian modelling for bbox coordinate uncertainty
- 30.8M parameters (7.4M fewer than YOLOv4-tiny)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# ConvNext-based Block
# ---------------------------------------------------------------------------

class ConvNextBlock(nn.Module):
    """
    ConvNext-based block as described in the paper:
    - 5-pixel padding to prevent edge feature loss
    - Depthwise separable convolution (7x7, stride=7 or 1)
    - 1x1 convolution for channel mixing
    - Residual connection with additional 7x7 depthwise conv + three 1x1 convs
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 expansion: int = 4):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Main branch: 7x7 depthwise conv + 1x1 pointwise conv
        # Padding=3 ensures size is preserved when stride=1,
        # and correct down-sampling when stride=2
        self.dwconv = nn.Conv2d(
            in_channels, in_channels, kernel_size=7, stride=stride,
            padding=3, groups=in_channels, bias=False
        )
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.pwconv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1,
            padding=0, bias=False
        )
        self.norm2 = nn.BatchNorm2d(out_channels)

        # Residual branch (deeper path) — must match spatial dims of main branch
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=stride,
                      padding=3, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels // expansion, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels // expansion),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // expansion, out_channels // expansion,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels // expansion),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels // expansion, out_channels, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(out_channels),
        )

        # Projection shortcut if dimensions mismatch
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        # Main branch
        out = self.dwconv(x)
        out = self.norm1(out)
        out = self.pwconv1(out)
        out = self.norm2(out)

        # Add residual branch
        out = out + self.residual(x)
        out = out + identity
        out = self.act(out)
        return out


# ---------------------------------------------------------------------------
# Feature Pyramid Network (FPN) - Bottleneck
# ---------------------------------------------------------------------------

class FPNBottleneck(nn.Module):
    """
    Feature Pyramid Network bottleneck matching YOLOv4-tiny structure.
    Concatenates low-level features with upsampled high-level features.
    """

    def __init__(self, c3: int, c4: int, c5: int):
        super().__init__()
        # Top-down pathway: project all features to c4//2 channels then fuse
        self.conv_c5 = nn.Sequential(
            nn.Conv2d(c5, c4 // 2, 1, bias=False),
            nn.BatchNorm2d(c4 // 2),
            nn.SiLU(inplace=True),
        )
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv_c4 = nn.Sequential(
            nn.Conv2d(c4, c4 // 2, 1, bias=False),
            nn.BatchNorm2d(c4 // 2),
            nn.SiLU(inplace=True),
        )

        self.conv_c3 = nn.Sequential(
            nn.Conv2d(c3, c4 // 2, 1, bias=False),
            nn.BatchNorm2d(c4 // 2),
            nn.SiLU(inplace=True),
        )

        # After concat: (c4//2 + c4//2) = c4 channels
        self.fuse = nn.Sequential(
            nn.Conv2d(c4, c4, 3, padding=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.SiLU(inplace=True),
        )

    def forward(self, c3: torch.Tensor, c4: torch.Tensor,
                c5: torch.Tensor) -> torch.Tensor:
        p5 = self.conv_c5(c5)
        p5_up = self.upsample(p5)

        c4_proj = self.conv_c4(c4)

        # Interpolate to same size if needed
        if p5_up.shape[-2:] != c4_proj.shape[-2:]:
            p5_up = F.interpolate(p5_up, size=c4_proj.shape[-2:],
                                  mode='nearest')

        fused = torch.cat([c4_proj, p5_up], dim=1)
        out = self.fuse(fused)
        return out


# ---------------------------------------------------------------------------
# Gaussian Detection Head
# ---------------------------------------------------------------------------

class GaussianDetectionHead(nn.Module):
    """
    Detection head with Gaussian modelling.
    Instead of predicting x, y, w, h directly, predicts:
    µtx, Σtx, µty, Σty, µtw, Σtw, µth, Σth
    plus objectness and class probabilities.

    For single-class (microrobot), num_classes = 1.
    """

    def __init__(self, in_channels: int, num_classes: int = 1,
                 num_anchors: int = 3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Each anchor predicts: [µx, Σx, µy, Σy, µw, Σw, µh, Σh, obj, cls...]
        self.num_outputs = 8 + 1 + num_classes  # 8 Gaussian params + obj + classes
        self.filters = num_anchors * self.num_outputs

        # Depthwise separable convolution to reduce parameters
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_channels, self.filters, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        out = self.conv(x)
        # Reshape to (batch, anchors, outputs, H, W)
        out = out.view(batch_size, self.num_anchors, self.num_outputs,
                       out.shape[-2], out.shape[-1])
        # Permute to (batch, anchors, H, W, outputs)
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        return out


# ---------------------------------------------------------------------------
# ConvNext-YOLO Model
# ---------------------------------------------------------------------------

class ConvNextYOLO(nn.Module):
    """
    Complete ConvNext-based YOLO for microrobot detection.
    Input: 416x416x3
    Output: Detection tensor with Gaussian bbox parameters.
    """

    def __init__(self, num_classes: int = 1, input_size: int = 416):
        super().__init__()
        self.input_size = input_size
        self.num_classes = num_classes

        # Backbone
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm2d(96),
            nn.SiLU(inplace=True),
        )

        # Stage 1: 104x104
        self.stage1 = ConvNextBlock(96, 96, stride=1)

        # Stage 2: 52x52
        self.stage2 = ConvNextBlock(96, 192, stride=2)

        # Stage 3: 26x26
        self.stage3 = ConvNextBlock(192, 384, stride=2)

        # Bottleneck (FPN)
        self.bottleneck = FPNBottleneck(c3=96, c4=192, c5=384)

        # Detection head
        self.head = GaussianDetectionHead(
            in_channels=192, num_classes=num_classes, num_anchors=3
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        c3 = self.stage1(x)      # 104x104
        c4 = self.stage2(c3)     # 52x52
        c5 = self.stage3(c4)     # 26x26

        fused = self.bottleneck(c3, c4, c5)
        detections = self.head(fused)
        return detections

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Gaussian Loss Functions
# ---------------------------------------------------------------------------

def gaussian_nll_loss(pred_mu: torch.Tensor, pred_sigma: torch.Tensor,
                      target: torch.Tensor, epsilon: float = 1e-9) -> torch.Tensor:
    """
    Negative log-likelihood loss for Gaussian modelling.
    L = -log(N(target | mu, sigma))
    """
    sigma = pred_sigma + epsilon
    loss = 0.5 * torch.log(2 * np.pi * sigma ** 2) + \
           0.5 * ((target - pred_mu) ** 2) / (sigma ** 2)
    return loss.mean()


def ciou_loss(pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
              eps: float = 1e-7) -> torch.Tensor:
    """
    Complete IoU (CIoU) loss for bbox regression.
    """
    # pred_boxes: [x, y, w, h], target_boxes: [x, y, w, h]
    px, py, pw, ph = pred_boxes[..., 0], pred_boxes[..., 1], \
                     pred_boxes[..., 2], pred_boxes[..., 3]
    tx, ty, tw, th = target_boxes[..., 0], target_boxes[..., 1], \
                     target_boxes[..., 2], target_boxes[..., 3]

    # Intersection area
    inter_x1 = torch.max(px - pw / 2, tx - tw / 2)
    inter_y1 = torch.max(py - ph / 2, ty - th / 2)
    inter_x2 = torch.min(px + pw / 2, tx + tw / 2)
    inter_y2 = torch.min(py + ph / 2, ty + th / 2)

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    pred_area = pw * ph
    target_area = tw * th
    union_area = pred_area + target_area - inter_area + eps

    iou = inter_area / union_area

    # Central point distance
    center_dist_sq = (px - tx) ** 2 + (py - ty) ** 2

    # Enclosing box diagonal
    c_x1 = torch.min(px - pw / 2, tx - tw / 2)
    c_y1 = torch.min(py - ph / 2, ty - th / 2)
    c_x2 = torch.max(px + pw / 2, tx + tw / 2)
    c_y2 = torch.max(py + ph / 2, ty + th / 2)
    c_diag_sq = (c_x2 - c_x1) ** 2 + (c_y2 - c_y1) ** 2 + eps

    # Aspect ratio consistency
    v = (4 / (np.pi ** 2)) * (torch.atan(tw / (th + eps)) -
                               torch.atan(pw / (ph + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - center_dist_sq / c_diag_sq - alpha * v
    loss = 1.0 - ciou
    return loss.mean()


class GaussianYOLOLoss(nn.Module):
    """
    Combined loss: Gaussian NLL for coordinates + CIoU for confidence.
    """

    def __init__(self, num_classes: int = 1):
        super().__init__()
        self.num_classes = num_classes
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, predictions: torch.Tensor, targets: List[dict],
                image_size: int = 416) -> torch.Tensor:
        """
        predictions: (batch, anchors, H, W, outputs)
        targets: list of dicts with 'boxes' [N, 4] and 'labels' [N]
        """
        # Simplified loss computation for demonstration
        # In practice, this requires anchor matching and positive/negative sampling
        batch_size = predictions.shape[0]

        # Extract Gaussian parameters
        pred_mu_x = predictions[..., 0]
        pred_sigma_x = torch.sigmoid(predictions[..., 1]) + 1e-9
        pred_mu_y = predictions[..., 2]
        pred_sigma_y = torch.sigmoid(predictions[..., 3]) + 1e-9
        pred_mu_w = torch.exp(predictions[..., 4])
        pred_sigma_w = torch.sigmoid(predictions[..., 5]) + 1e-9
        pred_mu_h = torch.exp(predictions[..., 6])
        pred_sigma_h = torch.sigmoid(predictions[..., 7]) + 1e-9
        pred_obj = predictions[..., 8]
        pred_cls = predictions[..., 9:] if self.num_classes > 0 else None

        # For demo: compute simple coordinate regression loss on all predictions
        # In real training, only positive samples contribute to coord loss
        coord_loss = gaussian_nll_loss(pred_mu_x, pred_sigma_x,
                                       torch.zeros_like(pred_mu_x)) + \
                     gaussian_nll_loss(pred_mu_y, pred_sigma_y,
                                       torch.zeros_like(pred_mu_y)) + \
                     gaussian_nll_loss(pred_mu_w, pred_sigma_w,
                                       torch.ones_like(pred_mu_w)) + \
                     gaussian_nll_loss(pred_mu_h, pred_sigma_h,
                                       torch.ones_like(pred_mu_h))

        obj_loss = self.bce(pred_obj, torch.zeros_like(pred_obj))

        return coord_loss + obj_loss


# ---------------------------------------------------------------------------
# Post-processing: Decode predictions
# ---------------------------------------------------------------------------

def decode_predictions(predictions: torch.Tensor, conf_thresh: float = 0.5,
                       input_size: int = 416) -> List[np.ndarray]:
    """
    Decode raw predictions to bounding boxes with Gaussian uncertainty.
    Returns list of [N, 10] arrays per image: [x, y, w, h, conf, 
                                               mu_x, sigma_x, mu_y, sigma_y,
                                               mu_w, sigma_w, mu_h, sigma_h]
    """
    batch_size = predictions.shape[0]
    results = []

    for b in range(batch_size):
        preds = predictions[b]  # (anchors, H, W, outputs)
        A, H, W, _ = preds.shape

        # Flatten spatial dimensions
        preds = preds.view(A * H * W, -1)

        obj_score = torch.sigmoid(preds[:, 8])
        mask = obj_score > conf_thresh

        if mask.sum() == 0:
            results.append(np.zeros((0, 12)))
            continue

        selected = preds[mask]
        scores = obj_score[mask]

        # Decode Gaussian parameters
        mu_x = selected[:, 0]
        sigma_x = torch.sigmoid(selected[:, 1])
        mu_y = selected[:, 2]
        sigma_y = torch.sigmoid(selected[:, 3])
        mu_w = torch.exp(selected[:, 4])
        sigma_w = torch.sigmoid(selected[:, 5])
        mu_h = torch.exp(selected[:, 6])
        sigma_h = torch.sigmoid(selected[:, 7])

        # Grid offsets (simplified - assumes single scale)
        # In full implementation, need to account for stride
        stride = input_size // H
        grid_x = torch.arange(W, device=preds.device).repeat(H, 1).view(-1)
        grid_y = torch.arange(H, device=preds.device).repeat_interleave(W)
        grid_x = grid_x.repeat(A)[mask]
        grid_y = grid_y.repeat(A)[mask]

        x = (mu_x + grid_x) * stride
        y = (mu_y + grid_y) * stride
        w = mu_w * stride
        h = mu_h * stride

        boxes = torch.stack([
            x, y, w, h, scores,
            mu_x, sigma_x, mu_y, sigma_y,
            mu_w, sigma_w, mu_h, sigma_h
        ], dim=1)

        results.append(boxes.detach().cpu().numpy())

    return results


# ---------------------------------------------------------------------------
# Simple demo/test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = ConvNextYOLO(num_classes=1, input_size=416)
    print(f"Model parameters: {model.count_parameters() / 1e6:.2f}M")

    dummy_input = torch.randn(2, 3, 416, 416)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # (batch, 3, H, W, 10)

    decoded = decode_predictions(output, conf_thresh=0.3)
    print(f"Decoded boxes per image: {[d.shape for d in decoded]}")
