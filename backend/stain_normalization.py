"""
Stain Normalization for Histopathology Images

Implements Macenko and Reinhard stain normalization methods for H&E stained
histopathology slides. This helps standardize images from different labs/scanners
to improve AI model consistency.

Reference:
- Macenko, M. et al. "A method for normalizing histology slides for quantitative analysis"
- Reinhard, E. et al. "Color Transfer between Images"
"""

import numpy as np
from PIL import Image
from typing import Union, Tuple, Optional
import io
import logging

logger = logging.getLogger(__name__)


class StainNormalizer:
    """
    Stain normalization for H&E histopathology images.

    Supports:
    - Macenko method (SVD-based)
    - Reinhard method (color transfer)
    - Vahadane method (sparse NMF)
    """

    # Standard H&E stain vectors (reference target)
    # These represent typical hematoxylin (purple) and eosin (pink) colors
    HE_REF = np.array([
        [0.5626, 0.2159],  # Hematoxylin
        [0.7201, 0.8012],  # Eosin
        [0.4062, 0.5581]   # Background
    ])

    # Reference maximum stain concentrations
    MAX_C_REF = np.array([1.9705, 1.0308])

    def __init__(self, method: str = 'macenko'):
        """
        Initialize stain normalizer.

        Args:
            method: Normalization method ('macenko', 'reinhard', 'vahadane')
        """
        self.method = method.lower()
        self.target_means = None
        self.target_stds = None
        self.target_stain_matrix = None

    def fit(self, target_image: Union[Image.Image, np.ndarray]):
        """
        Fit normalizer to a target/reference image.

        Args:
            target_image: Reference image to normalize to
        """
        if isinstance(target_image, Image.Image):
            target_image = np.array(target_image.convert('RGB'))

        if self.method == 'reinhard':
            self._fit_reinhard(target_image)
        elif self.method == 'macenko':
            self._fit_macenko(target_image)
        elif self.method == 'vahadane':
            self._fit_vahadane(target_image)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _fit_reinhard(self, image: np.ndarray):
        """Fit Reinhard method - compute target LAB statistics."""
        lab = self._rgb_to_lab(image)
        self.target_means = np.mean(lab, axis=(0, 1))
        self.target_stds = np.std(lab, axis=(0, 1))

    def _fit_macenko(self, image: np.ndarray):
        """Fit Macenko method - compute target stain matrix."""
        self.target_stain_matrix, self.target_max_c = self._get_stain_matrix(image)

    def _fit_vahadane(self, image: np.ndarray):
        """Fit Vahadane method - sparse NMF decomposition."""
        # Simplified - use Macenko as fallback
        self._fit_macenko(image)

    def transform(
        self,
        source_image: Union[Image.Image, np.ndarray, bytes],
        return_pil: bool = True
    ) -> Union[Image.Image, np.ndarray]:
        """
        Transform/normalize a source image.

        Args:
            source_image: Image to normalize
            return_pil: Return PIL Image if True, else numpy array

        Returns:
            Normalized image
        """
        # Handle different input types
        if isinstance(source_image, bytes):
            source_image = Image.open(io.BytesIO(source_image)).convert('RGB')
        if isinstance(source_image, Image.Image):
            source_image = np.array(source_image.convert('RGB'))

        if self.method == 'reinhard':
            normalized = self._transform_reinhard(source_image)
        elif self.method == 'macenko':
            normalized = self._transform_macenko(source_image)
        else:
            normalized = self._transform_macenko(source_image)

        # Clip to valid range
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)

        if return_pil:
            return Image.fromarray(normalized)
        return normalized

    def _transform_reinhard(self, image: np.ndarray) -> np.ndarray:
        """Apply Reinhard color transfer."""
        if self.target_means is None:
            # Use default reference if not fitted
            self.target_means = np.array([70.0, 0.0, 0.0])
            self.target_stds = np.array([20.0, 5.0, 5.0])

        # Convert to LAB
        lab = self._rgb_to_lab(image)

        # Compute source statistics
        source_means = np.mean(lab, axis=(0, 1))
        source_stds = np.std(lab, axis=(0, 1))

        # Prevent division by zero
        source_stds = np.maximum(source_stds, 1e-6)

        # Normalize
        lab_normalized = (lab - source_means) * (self.target_stds / source_stds) + self.target_means

        # Convert back to RGB
        return self._lab_to_rgb(lab_normalized)

    def _transform_macenko(self, image: np.ndarray) -> np.ndarray:
        """Apply Macenko stain normalization."""
        # Get source stain matrix
        source_stain_matrix, source_max_c = self._get_stain_matrix(image)

        # Use reference if not fitted
        if self.target_stain_matrix is None:
            self.target_stain_matrix = self.HE_REF
            self.target_max_c = self.MAX_C_REF

        # Convert to optical density
        od = self._rgb_to_od(image)

        # Get stain concentrations
        concentrations = self._get_concentrations(od, source_stain_matrix)

        # Normalize concentrations
        concentrations = concentrations * (self.target_max_c / (source_max_c + 1e-6))

        # Reconstruct image with target stain matrix
        od_normalized = np.dot(concentrations, self.target_stain_matrix.T)

        # Convert back to RGB
        return self._od_to_rgb(od_normalized.reshape(image.shape))

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to optical density."""
        image = image.astype(np.float32) / 255.0
        image = np.maximum(image, 1e-6)  # Avoid log(0)
        return -np.log(image)

    def _od_to_rgb(self, od: np.ndarray) -> np.ndarray:
        """Convert optical density to RGB."""
        rgb = np.exp(-od)
        return (rgb * 255).astype(np.uint8)

    def _get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract stain matrix using SVD."""
        # Convert to OD
        od = self._rgb_to_od(image)
        od_flat = od.reshape(-1, 3)

        # Remove background (low OD)
        od_threshold = 0.15
        mask = np.all(od_flat > od_threshold, axis=1)
        od_filtered = od_flat[mask]

        if len(od_filtered) < 100:
            # Not enough tissue - return reference
            return self.HE_REF, self.MAX_C_REF

        # SVD
        try:
            _, _, vh = np.linalg.svd(od_filtered, full_matrices=False)

            # Project onto plane defined by first two principal components
            plane = vh[:2, :]
            projected = np.dot(od_filtered, plane.T)

            # Find extreme angles (stain vectors)
            angles = np.arctan2(projected[:, 1], projected[:, 0])

            # Get percentile angles for robustness
            min_angle = np.percentile(angles, 1)
            max_angle = np.percentile(angles, 99)

            # Stain vectors
            v1 = np.array([np.cos(min_angle), np.sin(min_angle)])
            v2 = np.array([np.cos(max_angle), np.sin(max_angle)])

            # Project back to OD space
            stain1 = np.dot(v1, plane)
            stain2 = np.dot(v2, plane)

            # Normalize
            stain1 = stain1 / np.linalg.norm(stain1)
            stain2 = stain2 / np.linalg.norm(stain2)

            # Ensure hematoxylin is first (more blue)
            if stain1[2] < stain2[2]:
                stain1, stain2 = stain2, stain1

            stain_matrix = np.column_stack([stain1, stain2])

            # Get max concentrations
            concentrations = self._get_concentrations(od_flat, stain_matrix)
            max_c = np.percentile(concentrations, 99, axis=0)

            return stain_matrix, max_c

        except Exception as e:
            logger.warning(f"SVD failed: {e}, using reference stain matrix")
            return self.HE_REF, self.MAX_C_REF

    def _get_concentrations(
        self,
        od: np.ndarray,
        stain_matrix: np.ndarray
    ) -> np.ndarray:
        """Get stain concentrations using least squares."""
        od_flat = od.reshape(-1, 3)

        # Solve least squares: OD = C * Stain^T
        # C = OD * (Stain^T)^-1
        try:
            concentrations, _, _, _ = np.linalg.lstsq(
                stain_matrix, od_flat.T, rcond=None
            )
            return concentrations.T
        except:
            return np.zeros((od_flat.shape[0], 2))

    def _rgb_to_lab(self, image: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space."""
        # Normalize to 0-1
        rgb = image.astype(np.float32) / 255.0

        # RGB to XYZ
        mask = rgb > 0.04045
        rgb[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
        rgb[~mask] = rgb[~mask] / 12.92

        # sRGB to XYZ matrix
        m = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])

        xyz = np.dot(rgb, m.T)

        # XYZ to LAB
        # Reference white D65
        ref = np.array([0.95047, 1.0, 1.08883])
        xyz = xyz / ref

        mask = xyz > 0.008856
        xyz[mask] = xyz[mask] ** (1/3)
        xyz[~mask] = 7.787 * xyz[~mask] + 16/116

        lab = np.zeros_like(xyz)
        lab[..., 0] = 116 * xyz[..., 1] - 16  # L
        lab[..., 1] = 500 * (xyz[..., 0] - xyz[..., 1])  # a
        lab[..., 2] = 200 * (xyz[..., 1] - xyz[..., 2])  # b

        return lab

    def _lab_to_rgb(self, lab: np.ndarray) -> np.ndarray:
        """Convert LAB to RGB color space."""
        # LAB to XYZ
        xyz = np.zeros_like(lab)
        xyz[..., 1] = (lab[..., 0] + 16) / 116  # Y
        xyz[..., 0] = lab[..., 1] / 500 + xyz[..., 1]  # X
        xyz[..., 2] = xyz[..., 1] - lab[..., 2] / 200  # Z

        mask = xyz > 0.2068966
        xyz[mask] = xyz[mask] ** 3
        xyz[~mask] = (xyz[~mask] - 16/116) / 7.787

        # Reference white D65
        ref = np.array([0.95047, 1.0, 1.08883])
        xyz = xyz * ref

        # XYZ to RGB
        m_inv = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ])

        rgb = np.dot(xyz, m_inv.T)

        # Gamma correction
        mask = rgb > 0.0031308
        rgb[mask] = 1.055 * (rgb[mask] ** (1/2.4)) - 0.055
        rgb[~mask] = 12.92 * rgb[~mask]

        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def normalize_stain(
    image: Union[Image.Image, np.ndarray, bytes],
    method: str = 'macenko',
    target_image: Optional[Union[Image.Image, np.ndarray]] = None
) -> Image.Image:
    """
    Convenience function to normalize stain in a histopathology image.

    Args:
        image: Input image to normalize
        method: Normalization method ('macenko', 'reinhard')
        target_image: Optional target image to match (uses standard reference if None)

    Returns:
        Stain-normalized image
    """
    normalizer = StainNormalizer(method=method)

    if target_image is not None:
        normalizer.fit(target_image)

    return normalizer.transform(image)


def assess_staining_quality(
    image: Union[Image.Image, np.ndarray, bytes]
) -> dict:
    """
    Assess the staining quality of a histopathology image.

    Returns metrics for:
    - Stain intensity (under/over staining)
    - Color balance (H&E ratio)
    - Uniformity
    - Background percentage

    Args:
        image: Input histopathology image

    Returns:
        Dictionary with quality metrics
    """
    if isinstance(image, bytes):
        image = Image.open(io.BytesIO(image)).convert('RGB')
    if isinstance(image, Image.Image):
        image = np.array(image.convert('RGB'))

    # Convert to different color spaces
    hsv = _rgb_to_hsv(image)

    # Tissue mask (non-white areas)
    gray = np.mean(image, axis=2)
    tissue_mask = gray < 220
    tissue_percent = np.mean(tissue_mask) * 100

    if tissue_percent < 5:
        return {
            'quality_score': 0.2,
            'quality_label': 'Poor',
            'issues': ['Insufficient tissue content'],
            'tissue_percent': tissue_percent,
            'stain_intensity': 'unknown',
            'color_balance': 'unknown',
            'uniformity': 'unknown',
            'recommendations': ['Capture image with more tissue visible']
        }

    # Analyze tissue regions only
    tissue_hsv = hsv[tissue_mask]
    tissue_rgb = image[tissue_mask]

    # Stain intensity (based on saturation and value)
    mean_saturation = np.mean(tissue_hsv[:, 1])
    mean_value = np.mean(tissue_hsv[:, 2])

    if mean_saturation < 30:
        stain_intensity = 'under-stained'
        intensity_score = 0.5
    elif mean_saturation > 180:
        stain_intensity = 'over-stained'
        intensity_score = 0.6
    else:
        stain_intensity = 'optimal'
        intensity_score = 1.0

    # Color balance (ratio of pink to purple)
    # Hematoxylin (purple): hue 200-280
    # Eosin (pink): hue 300-360 or 0-30
    hues = tissue_hsv[:, 0]
    hematoxylin_mask = (hues > 200/360*255) & (hues < 280/360*255)
    eosin_mask = (hues > 300/360*255) | (hues < 30/360*255)

    h_ratio = np.mean(hematoxylin_mask)
    e_ratio = np.mean(eosin_mask)

    if h_ratio < 0.1 or e_ratio < 0.1:
        color_balance = 'imbalanced'
        balance_score = 0.5
    elif abs(h_ratio - e_ratio) > 0.3:
        color_balance = 'moderate'
        balance_score = 0.7
    else:
        color_balance = 'balanced'
        balance_score = 1.0

    # Uniformity (standard deviation of intensity across image)
    intensity_std = np.std(gray[tissue_mask])

    if intensity_std > 50:
        uniformity = 'non-uniform'
        uniformity_score = 0.5
    elif intensity_std > 30:
        uniformity = 'moderate'
        uniformity_score = 0.8
    else:
        uniformity = 'uniform'
        uniformity_score = 1.0

    # Overall quality score
    quality_score = (
        intensity_score * 0.35 +
        balance_score * 0.35 +
        uniformity_score * 0.2 +
        min(tissue_percent / 30, 1.0) * 0.1
    )

    # Quality label
    if quality_score >= 0.8:
        quality_label = 'Good'
    elif quality_score >= 0.6:
        quality_label = 'Acceptable'
    elif quality_score >= 0.4:
        quality_label = 'Marginal'
    else:
        quality_label = 'Poor'

    # Issues and recommendations
    issues = []
    recommendations = []

    if stain_intensity == 'under-stained':
        issues.append('Under-staining detected')
        recommendations.append('Consider re-staining or adjusting stain time')
    elif stain_intensity == 'over-stained':
        issues.append('Over-staining detected')
        recommendations.append('Consider de-staining or adjusting stain concentration')

    if color_balance == 'imbalanced':
        issues.append('H&E color imbalance')
        recommendations.append('Check staining protocol')

    if uniformity == 'non-uniform':
        issues.append('Non-uniform staining')
        recommendations.append('Check for uneven stain distribution')

    if tissue_percent < 30:
        issues.append('Low tissue content')
        recommendations.append('Center tissue in field of view')

    if not issues:
        issues.append('No significant issues detected')
        recommendations.append('Image quality is suitable for analysis')

    return {
        'quality_score': round(quality_score, 2),
        'quality_label': quality_label,
        'stain_intensity': stain_intensity,
        'color_balance': color_balance,
        'uniformity': uniformity,
        'tissue_percent': round(tissue_percent, 1),
        'issues': issues,
        'recommendations': recommendations,
        'detailed_metrics': {
            'mean_saturation': round(float(mean_saturation), 1),
            'mean_value': round(float(mean_value), 1),
            'hematoxylin_ratio': round(float(h_ratio), 3),
            'eosin_ratio': round(float(e_ratio), 3),
            'intensity_std': round(float(intensity_std), 1)
        }
    }


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB to HSV."""
    rgb = rgb.astype(np.float32) / 255.0

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    diff = maxc - minc

    # Hue
    h = np.zeros_like(maxc)
    mask = diff != 0

    r_mask = mask & (maxc == r)
    g_mask = mask & (maxc == g)
    b_mask = mask & (maxc == b)

    h[r_mask] = (60 * ((g[r_mask] - b[r_mask]) / diff[r_mask]) + 360) % 360
    h[g_mask] = (60 * ((b[g_mask] - r[g_mask]) / diff[g_mask]) + 120)
    h[b_mask] = (60 * ((r[b_mask] - g[b_mask]) / diff[b_mask]) + 240)

    # Saturation
    s = np.zeros_like(maxc)
    s[maxc != 0] = (diff[maxc != 0] / maxc[maxc != 0])

    # Value
    v = maxc

    # Scale to 0-255
    hsv = np.stack([h / 360 * 255, s * 255, v * 255], axis=-1)
    return hsv.astype(np.uint8)


if __name__ == '__main__':
    # Test stain normalization
    print("Stain Normalization Test")
    print("=" * 50)

    # Create test image (pink H&E-like colors)
    test_img = Image.new('RGB', (224, 224))
    pixels = test_img.load()
    for i in range(224):
        for j in range(224):
            # Simulate H&E staining
            if (i + j) % 2 == 0:
                # Purple (hematoxylin)
                pixels[i, j] = (150 + np.random.randint(-20, 20),
                               100 + np.random.randint(-20, 20),
                               180 + np.random.randint(-20, 20))
            else:
                # Pink (eosin)
                pixels[i, j] = (230 + np.random.randint(-20, 20),
                               180 + np.random.randint(-20, 20),
                               190 + np.random.randint(-20, 20))

    # Test normalization
    normalizer = StainNormalizer(method='macenko')
    normalized = normalizer.transform(test_img)

    print(f"Original size: {test_img.size}")
    print(f"Normalized size: {normalized.size}")

    # Test quality assessment
    quality = assess_staining_quality(test_img)
    print(f"\nStaining Quality Assessment:")
    print(f"  Score: {quality['quality_score']}")
    print(f"  Label: {quality['quality_label']}")
    print(f"  Intensity: {quality['stain_intensity']}")
    print(f"  Balance: {quality['color_balance']}")
    print(f"  Tissue %: {quality['tissue_percent']}")
