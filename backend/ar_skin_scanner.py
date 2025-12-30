"""
Real-Time AR Skin Scanner

Provides fast analysis for AR overlay:
- Quick lesion detection and bounding boxes
- Risk level color coding (green/yellow/red)
- Optimized for real-time camera feed processing
- Returns coordinates for AR overlay rendering
"""

import io
import time
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from PIL import Image

class RiskLevel(Enum):
    LOW = "low"           # Green - benign appearance
    MODERATE = "moderate" # Yellow - monitor recommended
    HIGH = "high"         # Red - dermatologist visit recommended
    UNKNOWN = "unknown"   # Gray - unclear, better image needed

@dataclass
class DetectedRegion:
    """A detected skin region of interest."""
    id: str
    x: float  # Normalized 0-1 (center x)
    y: float  # Normalized 0-1 (center y)
    width: float  # Normalized 0-1
    height: float  # Normalized 0-1
    risk_level: str
    confidence: float
    label: str
    description: str
    color: str  # Hex color for AR overlay

@dataclass
class ScanResult:
    """Result of AR skin scan."""
    regions: List[DetectedRegion]
    scan_time_ms: int
    frame_quality: str
    guidance: str
    timestamp: str

class ARSkinScanner:
    """Real-time skin scanning for AR overlay."""

    def __init__(self):
        # Risk level colors (hex)
        self.risk_colors = {
            RiskLevel.LOW: "#22C55E",      # Green
            RiskLevel.MODERATE: "#F59E0B",  # Yellow/Amber
            RiskLevel.HIGH: "#EF4444",      # Red
            RiskLevel.UNKNOWN: "#6B7280",   # Gray
        }

        # Detection thresholds
        self.min_region_size = 0.02  # Minimum 2% of image
        self.max_region_size = 0.3   # Maximum 30% of image

        # ABCDE criteria thresholds
        self.asymmetry_threshold = 0.3
        self.border_irregularity_threshold = 0.4
        self.color_variance_threshold = 0.25
        self.diameter_threshold = 6.0  # mm (assuming standard photo distance)

    def scan_frame(self, image_data: bytes, fast_mode: bool = True) -> ScanResult:
        """
        Scan a camera frame for skin regions of interest.

        Args:
            image_data: Raw image bytes from camera
            fast_mode: If True, use faster but less accurate detection

        Returns:
            ScanResult with detected regions and metadata
        """
        start_time = time.time()

        # Load image
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        img_array = np.array(image)
        height, width = img_array.shape[:2]

        # Assess frame quality
        frame_quality = self._assess_frame_quality(img_array)

        # Detect skin regions
        if fast_mode:
            regions = self._fast_detect_regions(img_array)
        else:
            regions = self._detailed_detect_regions(img_array)

        # Analyze each region
        detected_regions = []
        for i, region in enumerate(regions):
            analyzed = self._analyze_region(img_array, region, i)
            if analyzed:
                detected_regions.append(analyzed)

        # Generate guidance
        guidance = self._generate_guidance(detected_regions, frame_quality)

        scan_time_ms = int((time.time() - start_time) * 1000)

        return ScanResult(
            regions=detected_regions,
            scan_time_ms=scan_time_ms,
            frame_quality=frame_quality,
            guidance=guidance,
            timestamp=datetime.now().isoformat()
        )

    def _assess_frame_quality(self, img_array: np.ndarray) -> str:
        """Assess quality of the camera frame."""
        # Check brightness
        brightness = np.mean(img_array)
        if brightness < 50:
            return "too_dark"
        if brightness > 220:
            return "too_bright"

        # Check blur (using Laplacian variance)
        gray = np.mean(img_array, axis=2)
        laplacian_var = np.var(np.abs(np.diff(gray, axis=0))) + \
                       np.var(np.abs(np.diff(gray, axis=1)))
        if laplacian_var < 100:
            return "blurry"

        # Check if skin is detected
        skin_ratio = self._detect_skin_ratio(img_array)
        if skin_ratio < 0.1:
            return "no_skin_detected"

        return "good"

    def _detect_skin_ratio(self, img_array: np.ndarray) -> float:
        """Detect ratio of skin pixels in image."""
        # Simple skin detection using HSV-like thresholds
        r, g, b = img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

        # Skin color heuristics (works for various skin tones)
        skin_mask = (
            (r > 60) & (g > 40) & (b > 20) &
            (r > g) & (r > b) &
            (np.abs(r.astype(int) - g.astype(int)) > 10) &
            (r < 250) & (g < 250) & (b < 250)
        )

        return np.mean(skin_mask)

    def _fast_detect_regions(self, img_array: np.ndarray) -> List[Dict]:
        """Fast region detection using color thresholding."""
        height, width = img_array.shape[:2]
        regions = []

        # Convert to grayscale
        gray = np.mean(img_array, axis=2)

        # Detect darker regions (potential lesions)
        # Divide image into grid for faster processing
        grid_size = 8
        cell_h, cell_w = height // grid_size, width // grid_size

        for i in range(grid_size):
            for j in range(grid_size):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w

                cell = gray[y1:y2, x1:x2]
                cell_color = img_array[y1:y2, x1:x2]

                # Check for color anomalies
                mean_brightness = np.mean(cell)
                color_variance = np.std(cell_color)

                # Detect if this cell has potential lesion
                if color_variance > 30 or mean_brightness < np.mean(gray) - 20:
                    # Refine detection within cell
                    sub_regions = self._refine_detection(cell_color, x1, y1, width, height)
                    regions.extend(sub_regions)

        # Merge overlapping regions
        regions = self._merge_overlapping_regions(regions)

        return regions[:10]  # Limit to 10 regions for performance

    def _refine_detection(self, cell: np.ndarray, offset_x: int, offset_y: int,
                         img_width: int, img_height: int) -> List[Dict]:
        """Refine detection within a cell."""
        regions = []
        h, w = cell.shape[:2]

        # Find darkest region within cell
        gray = np.mean(cell, axis=2)
        threshold = np.percentile(gray, 30)

        dark_mask = gray < threshold
        if np.sum(dark_mask) < 100:  # Too small
            return []

        # Find centroid of dark region
        y_coords, x_coords = np.where(dark_mask)
        if len(y_coords) == 0:
            return []

        center_x = (np.mean(x_coords) + offset_x) / img_width
        center_y = (np.mean(y_coords) + offset_y) / img_height

        # Estimate size
        region_width = (np.max(x_coords) - np.min(x_coords)) / img_width
        region_height = (np.max(y_coords) - np.min(y_coords)) / img_height

        if region_width > self.min_region_size and region_height > self.min_region_size:
            regions.append({
                'x': center_x,
                'y': center_y,
                'width': region_width * 1.5,  # Add padding
                'height': region_height * 1.5,
                'pixels': cell[dark_mask] if len(y_coords) > 0 else None
            })

        return regions

    def _detailed_detect_regions(self, img_array: np.ndarray) -> List[Dict]:
        """More detailed region detection (slower but more accurate)."""
        # For detailed mode, use sliding window approach
        height, width = img_array.shape[:2]
        regions = []

        # Multi-scale detection
        scales = [0.05, 0.1, 0.15, 0.2]

        for scale in scales:
            window_h = int(height * scale)
            window_w = int(width * scale)
            stride = window_h // 2

            for y in range(0, height - window_h, stride):
                for x in range(0, width - window_w, stride):
                    window = img_array[y:y+window_h, x:x+window_w]

                    # Check if window contains potential lesion
                    if self._window_has_lesion(window):
                        regions.append({
                            'x': (x + window_w / 2) / width,
                            'y': (y + window_h / 2) / height,
                            'width': scale,
                            'height': scale,
                            'pixels': window
                        })

        # Merge and deduplicate
        regions = self._merge_overlapping_regions(regions)

        return regions[:10]

    def _window_has_lesion(self, window: np.ndarray) -> bool:
        """Check if a window likely contains a lesion."""
        gray = np.mean(window, axis=2)

        # Check for significant color variation
        color_std = np.std(window)
        if color_std < 15:
            return False

        # Check for dark center pattern
        h, w = gray.shape
        center = gray[h//4:3*h//4, w//4:3*w//4]
        border = np.concatenate([gray[:h//4].flatten(), gray[3*h//4:].flatten(),
                                gray[:, :w//4].flatten(), gray[:, 3*w//4:].flatten()])

        if np.mean(center) < np.mean(border) - 10:
            return True

        return False

    def _merge_overlapping_regions(self, regions: List[Dict]) -> List[Dict]:
        """Merge overlapping detected regions."""
        if len(regions) <= 1:
            return regions

        merged = []
        used = set()

        for i, r1 in enumerate(regions):
            if i in used:
                continue

            current = r1.copy()

            for j, r2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                # Check overlap
                if self._regions_overlap(current, r2):
                    # Merge
                    current = self._merge_two_regions(current, r2)
                    used.add(j)

            merged.append(current)
            used.add(i)

        return merged

    def _regions_overlap(self, r1: Dict, r2: Dict) -> bool:
        """Check if two regions overlap."""
        dx = abs(r1['x'] - r2['x'])
        dy = abs(r1['y'] - r2['y'])

        overlap_x = dx < (r1['width'] + r2['width']) / 2
        overlap_y = dy < (r1['height'] + r2['height']) / 2

        return overlap_x and overlap_y

    def _merge_two_regions(self, r1: Dict, r2: Dict) -> Dict:
        """Merge two regions into one."""
        x = (r1['x'] + r2['x']) / 2
        y = (r1['y'] + r2['y']) / 2
        width = max(r1['width'], r2['width']) * 1.2
        height = max(r1['height'], r2['height']) * 1.2

        return {
            'x': x, 'y': y,
            'width': min(width, self.max_region_size),
            'height': min(height, self.max_region_size),
            'pixels': r1.get('pixels')
        }

    def _analyze_region(self, img_array: np.ndarray, region: Dict, idx: int) -> Optional[DetectedRegion]:
        """Analyze a detected region and assign risk level."""
        height, width = img_array.shape[:2]

        # Extract region pixels
        x1 = max(0, int((region['x'] - region['width']/2) * width))
        x2 = min(width, int((region['x'] + region['width']/2) * width))
        y1 = max(0, int((region['y'] - region['height']/2) * height))
        y2 = min(height, int((region['y'] + region['height']/2) * height))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = img_array[y1:y2, x1:x2]

        # Calculate ABCDE criteria scores
        asymmetry = self._calculate_asymmetry(roi)
        border_score = self._calculate_border_irregularity(roi)
        color_score = self._calculate_color_variance(roi)
        diameter_score = region['width'] * region['height']  # Relative size

        # Determine risk level
        risk_score = 0
        risk_factors = []

        if asymmetry > self.asymmetry_threshold:
            risk_score += 1
            risk_factors.append("asymmetry")

        if border_score > self.border_irregularity_threshold:
            risk_score += 1
            risk_factors.append("irregular border")

        if color_score > self.color_variance_threshold:
            risk_score += 1
            risk_factors.append("color variation")

        if diameter_score > 0.01:  # Large lesion
            risk_score += 1
            risk_factors.append("size")

        # Assign risk level
        if risk_score >= 3:
            risk_level = RiskLevel.HIGH
            label = "High Risk Lesion"
            description = f"Shows: {', '.join(risk_factors)}. Dermatologist visit recommended."
        elif risk_score >= 2:
            risk_level = RiskLevel.MODERATE
            label = "Monitor This Spot"
            description = f"Shows: {', '.join(risk_factors)}. Track changes over time."
        elif risk_score >= 1:
            risk_level = RiskLevel.LOW
            label = "Low Risk"
            description = "Appears benign. Continue regular skin checks."
        else:
            risk_level = RiskLevel.LOW
            label = "Normal"
            description = "No concerning features detected."

        confidence = min(0.95, 0.5 + risk_score * 0.15)

        return DetectedRegion(
            id=f"region_{idx}",
            x=region['x'],
            y=region['y'],
            width=region['width'],
            height=region['height'],
            risk_level=risk_level.value,
            confidence=confidence,
            label=label,
            description=description,
            color=self.risk_colors[risk_level]
        )

    def _calculate_asymmetry(self, roi: np.ndarray) -> float:
        """Calculate asymmetry score of region."""
        gray = np.mean(roi, axis=2)
        h, w = gray.shape

        # Compare left-right
        left = gray[:, :w//2]
        right = np.flip(gray[:, w//2:], axis=1)

        # Ensure same size
        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        lr_diff = np.mean(np.abs(left - right)) / 255

        # Compare top-bottom
        top = gray[:h//2, :]
        bottom = np.flip(gray[h//2:, :], axis=0)

        min_h = min(top.shape[0], bottom.shape[0])
        top = top[:min_h, :]
        bottom = bottom[:min_h, :]

        tb_diff = np.mean(np.abs(top - bottom)) / 255

        return (lr_diff + tb_diff) / 2

    def _calculate_border_irregularity(self, roi: np.ndarray) -> float:
        """Calculate border irregularity score."""
        gray = np.mean(roi, axis=2)

        # Simple edge detection
        edges_x = np.abs(np.diff(gray, axis=1))
        edges_y = np.abs(np.diff(gray, axis=0))

        # Border irregularity from edge variance
        edge_variance = (np.std(edges_x) + np.std(edges_y)) / 2

        return edge_variance / 50  # Normalize

    def _calculate_color_variance(self, roi: np.ndarray) -> float:
        """Calculate color variance within region."""
        # Check variance in each color channel
        r_var = np.std(roi[:,:,0]) / 255
        g_var = np.std(roi[:,:,1]) / 255
        b_var = np.std(roi[:,:,2]) / 255

        return (r_var + g_var + b_var) / 3

    def _generate_guidance(self, regions: List[DetectedRegion], quality: str) -> str:
        """Generate user guidance based on scan results."""
        if quality == "too_dark":
            return "Move to better lighting for accurate scanning"
        if quality == "too_bright":
            return "Too bright - reduce lighting or move to shade"
        if quality == "blurry":
            return "Hold camera steady for clearer image"
        if quality == "no_skin_detected":
            return "Point camera at skin area to scan"

        if not regions:
            return "Scanning... Move camera slowly across skin"

        high_risk = sum(1 for r in regions if r.risk_level == "high")
        moderate_risk = sum(1 for r in regions if r.risk_level == "moderate")

        if high_risk > 0:
            return f"Found {high_risk} area(s) requiring attention. Tap for details."
        if moderate_risk > 0:
            return f"Found {moderate_risk} spot(s) to monitor. Tap to learn more."

        return "Skin appears healthy. Continue regular checks."


# FastAPI integration
def create_ar_scanner_router():
    """Create FastAPI router for AR scanner endpoints."""
    from fastapi import APIRouter, HTTPException, File, UploadFile
    from pydantic import BaseModel

    router = APIRouter(prefix="/api/ar-scanner", tags=["AR Skin Scanner"])
    scanner = ARSkinScanner()

    class RegionResponse(BaseModel):
        id: str
        x: float
        y: float
        width: float
        height: float
        risk_level: str
        confidence: float
        label: str
        description: str
        color: str

    class ScanResponse(BaseModel):
        regions: List[RegionResponse]
        scan_time_ms: int
        frame_quality: str
        guidance: str
        timestamp: str

    @router.post("/scan", response_model=ScanResponse)
    async def scan_frame(
        image: UploadFile = File(...),
        fast_mode: bool = True
    ):
        """
        Scan a camera frame for skin lesions.

        Optimized for real-time AR overlay.
        Returns detected regions with bounding boxes and risk levels.
        """
        try:
            image_data = await image.read()
            result = scanner.scan_frame(image_data, fast_mode)

            return ScanResponse(
                regions=[RegionResponse(
                    id=r.id,
                    x=r.x,
                    y=r.y,
                    width=r.width,
                    height=r.height,
                    risk_level=r.risk_level,
                    confidence=r.confidence,
                    label=r.label,
                    description=r.description,
                    color=r.color
                ) for r in result.regions],
                scan_time_ms=result.scan_time_ms,
                frame_quality=result.frame_quality,
                guidance=result.guidance,
                timestamp=result.timestamp
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scan failed: {str(e)}")

    @router.get("/colors")
    async def get_risk_colors():
        """Get color codes for each risk level."""
        return {
            "low": "#22C55E",
            "moderate": "#F59E0B",
            "high": "#EF4444",
            "unknown": "#6B7280"
        }

    return router
