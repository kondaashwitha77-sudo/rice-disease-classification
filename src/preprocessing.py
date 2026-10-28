"""
Image Preprocessing and Augmentation Module.
Implements the exact preprocessing pipeline described in Research.pdf:
- Resizing images to 224x224 (Eq. 2)
- Data Augmentation: translation, rotation, flipping (Eq. 3-6)
- Logarithmic Transformation image enhancement (Eq. 7-8)
"""

import cv2
import numpy as np
import random
from typing import Union, Tuple, Optional
from src.config import IMAGE_SIZE, LOG_C, AUGMENT_TRANSLATION_RANGE, AUGMENT_ROTATION_RANGE


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """
    Resizes the input rice image to unify dimensions across all datasets.
    Mathematical formulation: Research.pdf Eq. (2)
        D_resized(x, y) = D_t(x/s, y/t)
    where s, t are scaling factors.

    Args:
        image: Input image (BGR or RGB) as numpy array.
        target_size: Target (width, height) tuple, defaults to (224, 224).

    Returns:
        Resized image of shape (target_size[1], target_size[0], channels).
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def augment_image(
    image: np.ndarray,
    translation_range: Tuple[int, int] = AUGMENT_TRANSLATION_RANGE,
    rotation_range: Tuple[int, int] = AUGMENT_ROTATION_RANGE,
    apply_flip: bool = True
) -> np.ndarray:
    """
    Applies data augmentation to resolve class imbalance and prevent overfitting.
    Mathematical formulation: Research.pdf Eq. (3)-(6)
    
    1. Translation (Eq. 4):
       D_translated(x, y) = D_t(x - Delta_x, y - Delta_y) with Delta_x, Delta_y in [-5, 15].
    2. Flipping (Eq. 5):
       Horizontal: D_hf(x, y) = D_t(width - x - 1, y)
       Vertical:   D_vf(x, y) = D_t(x, height - y - 1)
    3. Rotation (Eq. 6):
       D_rotated(x, y) = D_t(x*cos(theta) - y*sin(theta), x*sin(theta) + y*cos(theta))
       with theta in [-10, 10] degrees.

    Args:
        image: Input RGB image.
        translation_range: (min_shift, max_shift) in pixels along x and y.
        rotation_range: (min_angle, max_angle) in degrees.
        apply_flip: Whether to apply random horizontal or vertical flips.

    Returns:
        Augmented RGB image.
    """
    h, w = image.shape[:2]
    augmented = image.copy()

    # 1. Random Translation (Eq. 4)
    dx = random.randint(translation_range[0], translation_range[1])
    dy = random.randint(translation_range[0], translation_range[1])
    m_trans = np.float32([[1, 0, dx], [0, 1, dy]])
    augmented = cv2.warpAffine(augmented, m_trans, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 2. Random Flipping (Eq. 5)
    if apply_flip:
        flip_choice = random.choice(['none', 'horizontal', 'vertical', 'both'])
        if flip_choice == 'horizontal':
            augmented = cv2.flip(augmented, 1)
        elif flip_choice == 'vertical':
            augmented = cv2.flip(augmented, 0)
        elif flip_choice == 'both':
            augmented = cv2.flip(augmented, -1)

    # 3. Random Rotation (Eq. 6)
    angle = random.uniform(rotation_range[0], rotation_range[1])
    center = (w // 2, h // 2)
    m_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    augmented = cv2.warpAffine(augmented, m_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

    return augmented


def logarithmic_enhancement(image: np.ndarray, c: float = LOG_C) -> np.ndarray:
    """
    Enhances uncontrolled illumination and low contrast conditions using
    Logarithmic Transformation.
    Mathematical formulation: Research.pdf Eq. (7)-(8)
    
        Step 1: Normalization (Eq. 7)
            D_tNorm = D_t / 255.0
        Step 2: Log Transformation (Eq. 8)
            D_tLog = c * log(1 + D_tNorm)
            where c is the scaling constant (set to 2.0 as in paper).

    The result is mapped back to the [0, 255] uint8 displayable and indexable range.

    Args:
        image: Input RGB image with uint8 values in [0, 255].
        c: Scaling constant, defaults to 2.0.

    Returns:
        Contrast-enhanced RGB image (uint8, [0, 255]).
    """
    # Eq. 7: Normalize to [0, 1]
    norm = image.astype(np.float32) / 255.0
    
    # Eq. 8: Logarithmic transform
    log_transformed = c * np.log1p(norm)
    
    # Maximum possible value when norm=1.0 is c * ln(2.0)
    max_val = c * np.log1p(1.0)
    
    # Rescale to [0, 255] uint8
    enhanced = (log_transformed / max_val * 255.0)
    enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
    
    return enhanced


def preprocess_image(
    image_input: Union[str, np.ndarray],
    target_size: Tuple[int, int] = IMAGE_SIZE,
    enhance: bool = True,
    augment: bool = False
) -> np.ndarray:
    """
    Full image preprocessing pipeline:
    1. Read image from path (or validate array)
    2. Convert to RGB color format
    3. Resize to target size (224x224)
    4. Optional Data Augmentation (for training)
    5. Logarithmic Contrast Enhancement

    Args:
        image_input: File path (str) or loaded numpy image array.
        target_size: Target dimensions (width, height).
        enhance: Whether to apply logarithmic contrast enhancement.
        augment: Whether to apply random translation, rotation, flipping.

    Returns:
        Preprocessed RGB image as numpy uint8 array of shape (224, 224, 3).
    """
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = image_input.copy()
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # 1. Resize to 224x224 (Eq. 2)
    image = resize_image(image, target_size=target_size)

    # 2. Optional Augmentation (Eq. 3-6)
    if augment:
        image = augment_image(image)

    # 3. Logarithmic Enhancement (Eq. 7-8)
    if enhance:
        image = logarithmic_enhancement(image, c=LOG_C)

    return image
