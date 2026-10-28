"""
Feature Extraction Module for Rice Crop Disease Detection.
Implements the multi-level handcrafted feature extraction technique from Research.pdf:
- Multi-Channel Local Binary Patterns (MCLBP) on RGB channels (Eq. 11-12)
- Multi-Level (ML) spatial block decomposition (Eq. 10)
- Color Correlogram (CC) spatial color distribution (Eq. 9)
- Feature Fusion combining color and texture representations (Eq. 13)
"""

import cv2
import numpy as np
from typing import List, Tuple
from src.config import (
    ML_LEVELS,
    LBP_POINTS,
    LBP_RADIUS,
    CORRELOGRAM_DISTANCES,
    COLOR_BINS_PER_CHANNEL
)


def compute_lbp_channel(channel: np.ndarray) -> np.ndarray:
    """
    Computes Local Binary Patterns (LBP) for a single 2D image channel.
    Mathematical formulation: Research.pdf Eq. (11)
        LBP(x_c, y_c) = sum_{p=0}^{N-1} 2^p * g(B_p - B(x_c, y_c))
    where:
        - B(x_c, y_c) is the intensity of the center pixel
        - B_p is the intensity of neighbor pixel p
        - g(z) = 1 if z >= 0 else 0
        - N = 8 neighbors at radius R = 1

    Implemented via fast vectorized array slicing and bitwise operations.

    Args:
        channel: 2D numpy array of shape (H, W) with uint8 intensity values.

    Returns:
        2D numpy array of shape (H-2, W-2) containing integer LBP codes in [0, 255].
    """
    center = channel[1:-1, 1:-1]
    lbp = np.zeros(center.shape, dtype=np.uint8)

    # 8 neighbor offsets (clockwise from top-left):
    # (row_slice, col_slice, bit_weight)
    shifts = [
        (slice(0, -2), slice(0, -2), 7),  # Top-Left
        (slice(0, -2), slice(1, -1), 6),  # Top
        (slice(0, -2), slice(2, None), 5), # Top-Right
        (slice(1, -1), slice(2, None), 4), # Right
        (slice(2, None), slice(2, None), 3), # Bottom-Right
        (slice(2, None), slice(1, -1), 2), # Bottom
        (slice(2, None), slice(0, -2), 1), # Bottom-Left
        (slice(1, -1), slice(0, -2), 0)   # Left
    ]

    for r_slice, c_slice, bit in shifts:
        neighbor = channel[r_slice, c_slice]
        condition = (neighbor >= center).astype(np.uint8)
        np.bitwise_or(lbp, (condition << bit), out=lbp)

    return lbp


def compute_mclbp_block(block_rgb: np.ndarray, num_bins: int = 256) -> np.ndarray:
    """
    Extracts Multi-Channel Local Binary Patterns (MCLBP) for an image block.
    Mathematical formulation: Research.pdf Eq. (12)
        LBP(RGB) = [LBP(Red), LBP(Green), LBP(Blue)]
    where each channel's LBP map is converted into a normalized histogram.

    Args:
        block_rgb: 3D numpy array of shape (H, W, 3) representing RGB block.
        num_bins: Number of histogram bins (default 256 for 8-bit LBP).

    Returns:
        1D numpy array of length num_bins * 3 (768 features), normalized to sum to 1.
    """
    histograms = []
    for c in range(3):
        lbp_map = compute_lbp_channel(block_rgb[:, :, c])
        hist = np.bincount(lbp_map.ravel(), minlength=num_bins).astype(np.float32)
        total_pixels = hist.sum()
        if total_pixels > 0:
            hist /= total_pixels
        histograms.append(hist)

    return np.concatenate(histograms)


def extract_ml_mclbp(image_rgb: np.ndarray, level: int = ML_LEVELS) -> np.ndarray:
    """
    Extracts Multi-Level Multi-Channel Local Binary Patterns (ML-MCLBP).
    Mathematical formulation: Research.pdf Eq. (10) & Fig. 5
        N = L^2
        DtLog = Union_{n=1}^{N} B_n
    Divides the image into an L x L grid of blocks and computes the MCLBP
    histogram for each block, capturing both global and fine local texture patterns.

    Args:
        image_rgb: RGB image array of shape (H, W, 3).
        level: Multi-level decomposition level L (e.g. 1, 2, 3, 4, or 5).

    Returns:
        1D numpy array of shape (L^2 * 768,).
    """
    h, w, _ = image_rgb.shape
    r_step = h // level
    c_step = w // level

    block_features = []
    for r in range(level):
        for c in range(level):
            r_start = r * r_step
            r_end = (r + 1) * r_step if r < level - 1 else h
            c_start = c * c_step
            c_end = (c + 1) * c_step if c < level - 1 else w

            block = image_rgb[r_start:r_end, c_start:c_end]
            mclbp_feat = compute_mclbp_block(block)
            block_features.append(mclbp_feat)

    return np.concatenate(block_features)


def quantize_colors(image_rgb: np.ndarray, bins_per_channel: int = COLOR_BINS_PER_CHANNEL) -> np.ndarray:
    """
    Quantizes 24-bit RGB colors into a compact palette of discrete color indices.
    For bins_per_channel = 4:
        Red (4 bins), Green (4 bins), Blue (4 bins) -> 4 * 4 * 4 = 64 discrete colors.

    Args:
        image_rgb: RGB image array (H, W, 3) in [0, 255].
        bins_per_channel: Number of quantization bins per color channel.

    Returns:
        2D numpy array of shape (H, W) with integer color indices in [0, num_colors - 1].
    """
    factor = 256 // bins_per_channel
    q = np.clip(image_rgb // factor, 0, bins_per_channel - 1)
    quantized = (
        q[:, :, 0] * (bins_per_channel ** 2) +
        q[:, :, 1] * bins_per_channel +
        q[:, :, 2]
    )
    return quantized.astype(np.int32)


def extract_color_correlogram(
    image_rgb: np.ndarray,
    distances: List[int] = CORRELOGRAM_DISTANCES,
    bins_per_channel: int = COLOR_BINS_PER_CHANNEL
) -> np.ndarray:
    """
    Computes the Color Correlogram (CC) feature descriptor.
    Mathematical formulation: Research.pdf Eq. (9)
        gamma^{(D)}_{c_i, c_j} = Pr_{p1 in I_{c_i}, p2 in I} [ p2 in I_{c_j} | |p1 - p2| = D ]
    
    Measures how the spatial correlation between pixels changes with distance D.
    We implement the spatial auto-correlogram (c_i = c_j) across 8 directions for each
    distance D in `distances`.

    Args:
        image_rgb: RGB image array (H, W, 3).
        distances: List of pixel distance radii D (default [1, 3, 5, 7]).
        bins_per_channel: Number of bins per channel for color quantization.

    Returns:
        1D numpy array of shape (len(distances) * num_colors,).
        For 4 distances and 64 colors = 256 color features.
    """
    quantized = quantize_colors(image_rgb, bins_per_channel=bins_per_channel)
    num_colors = bins_per_channel ** 3
    h, w = quantized.shape

    # 8 directional shift unit vectors
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
    correlogram_features = []

    for d in distances:
        match_counts = np.zeros(num_colors, dtype=np.float64)
        total_pixels = np.zeros(num_colors, dtype=np.float64)

        for sdr, sdc in directions:
            dr, dc = sdr * d, sdc * d
            r_start = max(0, -dr)
            r_end = min(h, h - dr)
            c_start = max(0, -dc)
            c_end = min(w, w - dc)

            p1 = quantized[r_start:r_end, c_start:c_end]
            p2 = quantized[r_start+dr:r_end+dr, c_start+dc:c_end+dc]

            # Accumulate total color frequency in valid window
            total_pixels += np.bincount(p1.ravel(), minlength=num_colors)

            # Accumulate matching pairs (same color at distance d)
            match_mask = (p1 == p2)
            match_counts += np.bincount(p1[match_mask], minlength=num_colors)

        # Compute empirical conditional probability gamma^{(D)}_{c_i, c_i}
        prob = np.zeros(num_colors, dtype=np.float32)
        valid = total_pixels > 0
        prob[valid] = match_counts[valid] / total_pixels[valid]
        correlogram_features.append(prob)

    return np.concatenate(correlogram_features)


def extract_block_color_moments(image_rgb: np.ndarray, level: int = ML_LEVELS) -> np.ndarray:
    """
    Extracts spatial block-level multi-channel chromatic moments and histograms across RGB, HSV, and LAB spaces.
    Mathematical formulation: Extends Research.pdf Eq. (10, 13) by capturing localized lesion discoloration
    (chlorosis, necrosis, yellow/gray halo margins) in conjunction with ML-MCLBP spatial blocks.

    Args:
        image_rgb: Preprocessed RGB image (224x224x3).
        level: Spatial grid level L (default ML_LEVELS = 3).

    Returns:
        1D numpy array of spatial chromatic features across all L x L blocks.
    """
    h, w, _ = image_rgb.shape
    r_step = h // level
    c_step = w // level
    img_hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    block_colors = []
    for r in range(level):
        for c in range(level):
            r_s = r * r_step
            r_e = (r + 1) * r_step if r < level - 1 else h
            c_s = c * c_step
            c_e = (c + 1) * c_step if c < level - 1 else w

            blk_rgb = image_rgb[r_s:r_e, c_s:c_e]
            blk_hsv = img_hsv[r_s:r_e, c_s:c_e]
            blk_lab = img_lab[r_s:r_e, c_s:c_e]

            # 1. First- and second-order color moments across RGB, HSV, and LAB
            for ch in range(3):
                block_colors.extend([
                    float(blk_rgb[:, :, ch].mean()) / 255.0,
                    float(blk_rgb[:, :, ch].std()) / 255.0,
                    float(blk_hsv[:, :, ch].mean()) / 255.0,
                    float(blk_hsv[:, :, ch].std()) / 255.0,
                    float(blk_lab[:, :, ch].mean()) / 255.0,
                    float(blk_lab[:, :, ch].std()) / 255.0
                ])

            # 2. Localized Hue (lesion tone) and Saturation (purity) histograms
            h_hist = np.bincount(blk_hsv[:, :, 0].ravel(), minlength=180)[:24].astype(np.float32)
            h_sum = h_hist.sum()
            if h_sum > 0:
                h_hist /= h_sum

            s_hist = np.bincount(blk_hsv[:, :, 1].ravel(), minlength=256)[:16].astype(np.float32)
            s_sum = s_hist.sum()
            if s_sum > 0:
                s_hist /= s_sum

            block_colors.extend(h_hist)
            block_colors.extend(s_hist)

    return np.array(block_colors, dtype=np.float32)


def extract_fused_features(
    image_rgb: np.ndarray,
    level: int = ML_LEVELS,
    distances: List[int] = CORRELOGRAM_DISTANCES,
    bins_per_channel: int = COLOR_BINS_PER_CHANNEL
) -> np.ndarray:
    """
    Fuses color features and color texture features via concatenation.
    Mathematical formulation: Research.pdf Eq. (13)
        FE = [ gamma^{(D)}_{c_i, c_j}, LBP(RGB), Spatial_Color ]
    
    Combines:
    1. Color Correlogram (CC): captures global spatial color correlation (Eq. 9)
    2. Multi-Level MCLBP: captures multi-scale chromatic and structural leaf textures (Eq. 10-12)
    3. Spatial Chromatic Lesion Distribution: captures fine localized discoloration across blocks

    Args:
        image_rgb: Preprocessed RGB image (224x224x3).
        level: Spatial grid level for ML-MCLBP (L).
        distances: List of distances D for Color Correlogram.
        bins_per_channel: Bins per RGB channel for color quantization.

    Returns:
        1D numpy feature vector.
    """
    # 1. Texture Features via Multi-Level Multi-Channel LBP (Eq. 10-12)
    texture_features = extract_ml_mclbp(image_rgb, level=level)

    # 2. Color Features via Spatial Color Correlogram (Eq. 9)
    color_features = extract_color_correlogram(
        image_rgb, distances=distances, bins_per_channel=bins_per_channel
    )

    # 3. Spatial Chromatic Lesion Distribution (Multi-Channel Block Moments & Histograms)
    spatial_color_moments = extract_block_color_moments(image_rgb, level=level)

    # 4. Concatenation Feature Fusion (Eq. 13)
    fused_features = np.concatenate([texture_features, color_features, spatial_color_moments])
    return fused_features.astype(np.float32)

