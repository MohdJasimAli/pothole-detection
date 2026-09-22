"""
Image Preprocessing Module
============================
Handles image enhancement, normalization, and preparation
for deep learning models.
"""

import cv2
import numpy as np
from typing import Tuple, Optional

import config


def enhance_contrast(image: np.ndarray,
                     clip_limit: float = None,
                     tile_grid_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Enhance image contrast using CLAHE.

    Args:
        image: Input image (BGR format)
        clip_limit: CLAHE clip limit
        tile_grid_size: Tile grid size for CLAHE

    Returns:
        Contrast-enhanced image
    """
    if clip_limit is None:
        clip_limit = config.CLAHE_CLIP_LIMIT
    if tile_grid_size is None:
        tile_grid_size = config.CLAHE_TILE_GRID_SIZE

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_channel = clahe.apply(l_channel)

    limg = cv2.merge((l_channel, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return enhanced


def reduce_noise(image: np.ndarray,
                 blur_kernel: Tuple[int, int] = None,
                 bilateral_iterations: int = 1) -> np.ndarray:
    """Reduce image noise using Gaussian blur and bilateral filtering."""
    if blur_kernel is None:
        blur_kernel = config.GAUSSIAN_BLUR_KERNEL

    denoised = cv2.GaussianBlur(image, blur_kernel, 0)

    for _ in range(bilateral_iterations):
        denoised = cv2.bilateralFilter(denoised, 9, 75, 75)

    return denoised


def normalize_image(image: np.ndarray,
                    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
                    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Normalize image pixel values using ImageNet statistics.

    Args:
        image: Input image (uint8)
        mean: Per-channel mean values
        std: Per-channel standard deviation values

    Returns:
        Normalized image (float32)
    """
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_normalized = image_rgb.astype(np.float32) / 255.0

    for i in range(3):
        image_normalized[:, :, i] = (image_normalized[:, :, i] - mean[i]) / std[i]

    return image_normalized


def correct_lighting(image: np.ndarray) -> np.ndarray:
    """Correct non-uniform lighting using background estimation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    bg = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    bg = cv2.GaussianBlur(bg, (101, 101), 0)
    bg = np.maximum(bg, 1)

    result = image.copy().astype(np.float32)
    bg_3ch = np.stack([bg, bg, bg], axis=2)
    result = result / bg_3ch * 128.0
    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def remove_shadow(image: np.ndarray) -> np.ndarray:
    """Reduce shadow effects in the image."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v = clahe.apply(v)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def preprocess_image(image: np.ndarray,
                     normalize: bool = True,
                     target_size: Tuple[int, int] = None,
                     enhance: bool = True,
                     denoise: bool = True,
                     correct_light: bool = True) -> np.ndarray:
    """
    Full preprocessing pipeline for road images.

    Args:
        image: Input image (BGR format)
        normalize: Whether to normalize pixel values
        target_size: Optional (width, height) to resize to
        enhance: Apply CLAHE contrast enhancement
        denoise: Apply noise reduction
        correct_light: Apply lighting correction

    Returns:
        Preprocessed image
    """
    img = image.copy()

    if correct_light:
        img = correct_lighting(img)

    if enhance:
        img = enhance_contrast(img)

    if denoise:
        img = reduce_noise(img, bilateral_iterations=1)

    img = remove_shadow(img)

    if target_size is not None:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    if normalize:
        img = normalize_image(img)

    return img


def extract_pothole_region(image: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           padding: int = 10) -> np.ndarray:
    """
    Extract and preprocess a pothole region from the image.

    Args:
        image: Full road image
        bbox: (x1, y1, x2, y2) bounding box coordinates
        padding: Extra pixels around the bounding box

    Returns:
        Cropped and preprocessed pothole region
    """
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    region = image[y1:y2, x1:x2]
    region = enhance_contrast(region)
    region = reduce_noise(region, bilateral_iterations=1)

    return region