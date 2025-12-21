"""
Dermoscopic Feature Detection Module

Analyzes dermoscopic patterns for melanoma detection including:
- Pigment network (reticular, globular, branched)
- Globules and dots
- Streaks and pseudopods
- Blue-white veil
- Vascular patterns
- Regression structures

This is the gold standard for melanoma detection used by dermatologists.
"""

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import io
import base64
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

# Thread pool for CPU-bound operations
_executor = ThreadPoolExecutor(max_workers=2)

# Maximum image dimension for processing (resize larger images)
MAX_IMAGE_DIMENSION = 800


class DermoscopicFeatureDetector:
    """
    Detects dermoscopic features using computer vision and deep learning.

    Note: This implementation uses classical computer vision techniques.
    For production, you should train a CNN on dermoscopy datasets like PH2 or HAM10000.
    """

    def __init__(self):
        self.feature_descriptions = {
            'pigment_network': {
                'reticular': 'Regular mesh-like pattern - typically benign',
                'atypical': 'Irregular, widened network - concerning for melanoma',
                'branched': 'Tree-like branching pattern - concerning'
            },
            'globules': {
                'regular': 'Uniform round structures - typically benign',
                'irregular': 'Varying sizes and shapes - concerning',
                'peripheral': 'Located at edges - concerning for melanoma'
            },
            'streaks': {
                'radial': 'Lines radiating from center - concerning',
                'pseudopods': 'Finger-like projections - concerning'
            },
            'blue_white_veil': 'Irregular blue-white area - highly concerning for melanoma',
            'vascular_patterns': {
                'dotted': 'Small dots - may indicate melanoma',
                'linear_irregular': 'Irregular linear vessels - concerning',
                'hairpin': 'Looped vessels - typically benign'
            },
            'regression': 'White or blue areas indicating regression - requires monitoring'
        }

    def _resize_if_needed(self, img_array: np.ndarray) -> np.ndarray:
        """
        Resize image if it exceeds maximum dimensions for faster processing.
        """
        h, w = img_array.shape[:2]
        max_dim = max(h, w)

        if max_dim > MAX_IMAGE_DIMENSION:
            scale = MAX_IMAGE_DIMENSION / max_dim
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[DERMOSCOPY] Resized image from {w}x{h} to {new_w}x{new_h}")

        return img_array

    def analyze(self, image_bytes: bytes) -> Dict:
        """
        Comprehensive dermoscopic feature analysis.

        Args:
            image_bytes: Image bytes

        Returns:
            Dictionary with all detected features and risk assessment
        """
        start_time = time.time()

        # Load image
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image.convert('RGB'))

        # Validate minimum image size (need at least 10x10 for meaningful analysis)
        min_size = 10
        height, width = img_array.shape[:2]
        if width < min_size or height < min_size:
            raise ValueError(
                f"Image too small for dermoscopy analysis. "
                f"Minimum size is {min_size}x{min_size} pixels, "
                f"but received {width}x{height} pixels."
            )

        # Resize if needed for faster processing
        img_array = self._resize_if_needed(img_array)

        print(f"[DERMOSCOPY] Processing image of size {img_array.shape[1]}x{img_array.shape[0]}")

        # Run all feature detections
        results = {
            'pigment_network': self.detect_pigment_network(img_array),
            'globules': self.detect_globules(img_array),
            'streaks': self.detect_streaks(img_array),
            'blue_white_veil': self.detect_blue_white_veil(img_array),
            'vascular_patterns': self.detect_vascular_patterns(img_array),
            'regression': self.detect_regression(img_array),
            'color_analysis': self.analyze_colors(img_array),
            'symmetry_analysis': self.analyze_symmetry(img_array)
        }

        # Generate feature visualization overlays
        results['overlays'] = self.generate_overlays(img_array, results)

        # Calculate overall risk score
        results['risk_assessment'] = self.assess_risk(results)

        # Generate 7-point checklist score
        results['seven_point_score'] = self.calculate_seven_point_checklist(results)

        # ABCD dermoscopy score
        results['abcd_score'] = self.calculate_abcd_score(results)

        elapsed = time.time() - start_time
        print(f"[DERMOSCOPY] Analysis completed in {elapsed:.2f}s")

        return results

    async def analyze_async(self, image_bytes: bytes) -> Dict:
        """
        Async wrapper for analyze() that runs CPU-bound work in thread pool.
        This prevents blocking the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(_executor, self.analyze, image_bytes)

    def detect_pigment_network(self, img: np.ndarray) -> Dict:
        """
        Detect pigment network patterns using edge detection and frequency analysis.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # Detect edges
        edges = cv2.Canny(enhanced, 50, 150)

        # Morphological operations to detect network pattern
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours representing network lines
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Analyze network regularity
        network_detected = len(contours) > 10

        if network_detected:
            # Calculate regularity by analyzing contour properties
            areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]

            if len(areas) > 5:
                area_std = np.std(areas)
                area_mean = np.mean(areas)
                regularity_score = 1 - min(area_std / (area_mean + 1), 1.0)

                # Classify network type
                if regularity_score > 0.7:
                    network_type = 'reticular'
                    risk = 'low'
                elif regularity_score > 0.4:
                    network_type = 'atypical'
                    risk = 'moderate'
                else:
                    network_type = 'branched'
                    risk = 'high'
            else:
                network_type = 'sparse'
                risk = 'low'
                regularity_score = 0.5
        else:
            network_type = 'absent'
            risk = 'none'
            regularity_score = 0.0

        return {
            'detected': network_detected,
            'type': network_type,
            'regularity_score': float(regularity_score),
            'risk_level': risk,
            'description': self.feature_descriptions['pigment_network'].get(network_type, 'No pigment network detected'),
            'contour_count': len(contours),
            'coordinates': self._extract_coordinates(contours[:20])  # Top 20 contours
        }

    def detect_globules(self, img: np.ndarray) -> Dict:
        """
        Detect globules and dots using blob detection and color analysis.
        """
        # Convert to LAB color space for better color detection
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply threshold to detect dark regions (globules are typically dark)
        _, binary = cv2.threshold(l_channel, 100, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to isolate globules
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # Detect circular features using Hough Circles
        circles = cv2.HoughCircles(
            l_channel,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=10,
            param1=50,
            param2=30,
            minRadius=3,
            maxRadius=30
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            globule_count = len(circles[0])

            # Analyze size distribution
            radii = [c[2] for c in circles[0]]
            size_std = np.std(radii)
            size_mean = np.mean(radii)

            # Regular globules have similar sizes
            if size_std / (size_mean + 1) < 0.3:
                globule_type = 'regular'
                risk = 'low'
            else:
                globule_type = 'irregular'
                risk = 'moderate'

            # Check if globules are peripheral
            img_center = np.array([img.shape[1] / 2, img.shape[0] / 2])
            distances = [np.linalg.norm([c[0], c[1]] - img_center) for c in circles[0]]
            avg_distance = np.mean(distances)

            if avg_distance > min(img.shape[:2]) * 0.3:
                globule_type = 'peripheral'
                risk = 'high'

            coordinates = [(int(c[0]), int(c[1]), int(c[2])) for c in circles[0]]
        else:
            globule_count = 0
            globule_type = 'absent'
            risk = 'none'
            coordinates = []
            size_std = 0.0
            size_mean = 0.0

        return {
            'detected': globule_count > 0,
            'count': int(globule_count),
            'type': globule_type,
            'size_variability': float(size_std / (size_mean + 1)),
            'risk_level': risk,
            'description': self.feature_descriptions['globules'].get(globule_type, 'No globules detected'),
            'coordinates': coordinates
        }

    def detect_streaks(self, img: np.ndarray) -> Dict:
        """
        Detect radial streaming and pseudopods using directional filters.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Detect edges
        edges = cv2.Canny(gray, 50, 150)

        # Use Hough Line Transform to detect linear structures
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=20,
            maxLineGap=10
        )

        if lines is not None:
            # Calculate center of image
            center = np.array([img.shape[1] / 2, img.shape[0] / 2])

            # Analyze line directions relative to center
            radial_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]

                # Calculate if line is radial (pointing away from center)
                line_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                direction = line_center - center
                line_vec = np.array([x2 - x1, y2 - y1])

                # Check if line direction aligns with radial direction
                if np.dot(direction, line_vec) > 0:
                    radial_lines.append(line[0])

            streak_count = len(radial_lines)

            if streak_count > 3:
                streak_type = 'radial'
                risk = 'high'
            elif streak_count > 0:
                streak_type = 'pseudopods'
                risk = 'moderate'
            else:
                streak_type = 'absent'
                risk = 'none'

            coordinates = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in radial_lines]
        else:
            streak_count = 0
            streak_type = 'absent'
            risk = 'none'
            coordinates = []

        return {
            'detected': streak_count > 0,
            'count': int(streak_count),
            'type': streak_type,
            'risk_level': risk,
            'description': self.feature_descriptions['streaks'].get(streak_type, 'No streaks detected'),
            'coordinates': coordinates
        }

    def detect_blue_white_veil(self, img: np.ndarray) -> Dict:
        """
        Detect blue-white veil using color space analysis.
        """
        # Convert to HSV for color detection
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # Define blue-white color range
        # Blue hue range
        lower_blue = np.array([90, 20, 100])
        upper_blue = np.array([130, 255, 255])

        # White range (high value, low saturation)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 60, 255])

        # Create masks
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Combine masks (blue AND white nearby)
        combined = cv2.bitwise_or(blue_mask, white_mask)

        # Calculate percentage of image with blue-white veil
        veil_percentage = (np.sum(combined > 0) / combined.size) * 100

        if veil_percentage > 5:
            detected = True
            if veil_percentage > 15:
                risk = 'high'
                intensity = 'strong'
            elif veil_percentage > 10:
                risk = 'moderate'
                intensity = 'moderate'
            else:
                risk = 'low'
                intensity = 'mild'
        else:
            detected = False
            risk = 'none'
            intensity = 'absent'

        # Find contours of veil regions
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = self._extract_coordinates(contours)

        return {
            'detected': detected,
            'coverage_percentage': float(veil_percentage),
            'intensity': intensity,
            'risk_level': risk,
            'description': self.feature_descriptions['blue_white_veil'] if detected else 'No blue-white veil detected',
            'coordinates': coordinates
        }

    def detect_vascular_patterns(self, img: np.ndarray) -> Dict:
        """
        Detect vascular patterns using red channel analysis.
        """
        # Extract red channel (vessels appear darker in red channel)
        red_channel = img[:, :, 0]

        # Enhance vessels using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        tophat = cv2.morphologyEx(red_channel, cv2.MORPH_TOPHAT, kernel)

        # Threshold to isolate vessels
        _, binary = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Detect vessel patterns
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 5:
            # Analyze vessel shapes
            vessel_features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 5:  # Filter small noise
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter ** 2)
                        vessel_features.append(circularity)

            if vessel_features:
                avg_circularity = np.mean(vessel_features)

                # Classify vessel type
                if avg_circularity > 0.7:
                    vessel_type = 'dotted'
                    risk = 'moderate'
                elif avg_circularity > 0.3:
                    vessel_type = 'hairpin'
                    risk = 'low'
                else:
                    vessel_type = 'linear_irregular'
                    risk = 'high'

                detected = True
            else:
                detected = False
                vessel_type = 'absent'
                risk = 'none'
        else:
            detected = False
            vessel_type = 'absent'
            risk = 'none'

        coordinates = self._extract_coordinates(contours[:10])

        return {
            'detected': detected,
            'type': vessel_type,
            'risk_level': risk,
            'description': self.feature_descriptions['vascular_patterns'].get(vessel_type, 'No vascular patterns detected'),
            'vessel_count': len(contours),
            'coordinates': coordinates
        }

    def detect_regression(self, img: np.ndarray) -> Dict:
        """
        Detect regression structures (white or blue scar-like areas).
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l_channel = lab[:, :, 0]

        # Detect very bright areas (white regression)
        _, white_regression = cv2.threshold(l_channel, 200, 255, cv2.THRESH_BINARY)

        # Calculate coverage
        white_percentage = (np.sum(white_regression > 0) / white_regression.size) * 100

        # Also check for bluish regression
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower_blue = np.array([90, 30, 80])
        upper_blue = np.array([130, 255, 200])
        blue_regression = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_percentage = (np.sum(blue_regression > 0) / blue_regression.size) * 100

        total_regression = white_percentage + blue_percentage

        if total_regression > 5:
            detected = True
            if total_regression > 20:
                severity = 'extensive'
                risk = 'high'
            elif total_regression > 10:
                severity = 'moderate'
                risk = 'moderate'
            else:
                severity = 'mild'
                risk = 'low'
        else:
            detected = False
            severity = 'absent'
            risk = 'none'

        # Find regression regions
        combined_regression = cv2.bitwise_or(white_regression, blue_regression)
        contours, _ = cv2.findContours(combined_regression, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coordinates = self._extract_coordinates(contours)

        return {
            'detected': detected,
            'coverage_percentage': float(total_regression),
            'severity': severity,
            'risk_level': risk,
            'description': self.feature_descriptions['regression'] if detected else 'No regression detected',
            'coordinates': coordinates
        }

    def analyze_colors(self, img: np.ndarray) -> Dict:
        """
        Analyze color distribution (melanomas typically have 3+ colors).
        """
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img

        # Reshape for k-means
        pixels = img_rgb.reshape((-1, 3))
        pixels = np.float32(pixels)

        # Use k-means to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 6  # Look for up to 6 colors
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

        # Calculate percentage of each color
        unique, counts = np.unique(labels, return_counts=True)
        percentages = (counts / len(labels)) * 100

        # Filter out colors that are < 5% of image
        significant_colors = np.sum(percentages > 5)

        # Melanoma typically has 3+ distinct colors
        if significant_colors >= 5:
            color_variety = 'very_high'
            risk = 'high'
        elif significant_colors >= 3:
            color_variety = 'high'
            risk = 'moderate'
        elif significant_colors >= 2:
            color_variety = 'moderate'
            risk = 'low'
        else:
            color_variety = 'low'
            risk = 'low'

        return {
            'distinct_colors': int(significant_colors),
            'variety': color_variety,
            'risk_level': risk,
            'dominant_colors': centers.astype(int).tolist(),
            'color_percentages': percentages.tolist()
        }

    def analyze_symmetry(self, img: np.ndarray) -> Dict:
        """
        Analyze lesion symmetry (asymmetry is concerning).
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Segment lesion
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find largest contour
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            lesion = max(contours, key=cv2.contourArea)

            # Calculate moments
            M = cv2.moments(lesion)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Flip horizontally and vertically
                h, w = binary.shape
                flipped_h = cv2.flip(binary, 1)
                flipped_v = cv2.flip(binary, 0)

                # Calculate difference
                diff_h = cv2.absdiff(binary, flipped_h)
                diff_v = cv2.absdiff(binary, flipped_v)

                asymmetry_h = np.sum(diff_h > 0) / binary.size
                asymmetry_v = np.sum(diff_v > 0) / binary.size

                overall_asymmetry = (asymmetry_h + asymmetry_v) / 2

                if overall_asymmetry > 0.3:
                    symmetry = 'highly_asymmetric'
                    risk = 'high'
                elif overall_asymmetry > 0.15:
                    symmetry = 'asymmetric'
                    risk = 'moderate'
                else:
                    symmetry = 'symmetric'
                    risk = 'low'
            else:
                overall_asymmetry = 0.0
                symmetry = 'unknown'
                risk = 'none'
        else:
            overall_asymmetry = 0.0
            symmetry = 'unknown'
            risk = 'none'

        return {
            'asymmetry_score': float(overall_asymmetry),
            'classification': symmetry,
            'risk_level': risk
        }

    def calculate_seven_point_checklist(self, features: Dict) -> Dict:
        """
        Calculate 7-point checklist score (clinical dermoscopy scoring system).

        Major criteria (2 points each):
        - Atypical pigment network
        - Blue-white veil
        - Atypical vascular pattern

        Minor criteria (1 point each):
        - Irregular streaks
        - Irregular dots/globules
        - Irregular pigmentation
        - Regression structures

        Score >= 3: Consider melanoma
        """
        score = 0
        criteria_met = []

        # Major criteria (2 points each)
        if features['pigment_network']['type'] in ['atypical', 'branched']:
            score += 2
            criteria_met.append('Atypical pigment network (Major, +2)')

        if features['blue_white_veil']['detected']:
            score += 2
            criteria_met.append('Blue-white veil (Major, +2)')

        if features['vascular_patterns']['type'] == 'linear_irregular':
            score += 2
            criteria_met.append('Atypical vascular pattern (Major, +2)')

        # Minor criteria (1 point each)
        if features['streaks']['detected']:
            score += 1
            criteria_met.append('Irregular streaks (Minor, +1)')

        if features['globules']['type'] == 'irregular':
            score += 1
            criteria_met.append('Irregular globules (Minor, +1)')

        if features['color_analysis']['distinct_colors'] >= 3:
            score += 1
            criteria_met.append('Irregular pigmentation (Minor, +1)')

        if features['regression']['detected']:
            score += 1
            criteria_met.append('Regression structures (Minor, +1)')

        # Interpretation
        if score >= 3:
            interpretation = 'MELANOMA SUSPECTED - Urgent dermatologist referral recommended'
            urgency = 'urgent'
        elif score >= 1:
            interpretation = 'Monitor closely - Follow-up with dermatologist recommended'
            urgency = 'routine'
        else:
            interpretation = 'Low suspicion for melanoma - Routine monitoring'
            urgency = 'none'

        return {
            'score': score,
            'max_score': 9,
            'criteria_met': criteria_met,
            'interpretation': interpretation,
            'urgency': urgency
        }

    def calculate_abcd_score(self, features: Dict) -> Dict:
        """
        Calculate ABCD dermoscopy score.

        A - Asymmetry (0-2 points)
        B - Border (0-8 points)
        C - Color (1-6 points)
        D - Dermoscopic structures (1-5 points)

        Total Dermoscopy Score (TDS) = (A × 1.3) + (B × 0.1) + (C × 0.5) + (D × 0.5)

        TDS < 4.75: Benign
        TDS 4.75-5.45: Suspicious
        TDS > 5.45: Melanoma
        """
        # A - Asymmetry (0-2)
        asymmetry = features['symmetry_analysis']['asymmetry_score']
        if asymmetry > 0.3:
            a_score = 2
        elif asymmetry > 0.15:
            a_score = 1
        else:
            a_score = 0

        # B - Border (0-8) - based on irregularity
        border_score = 0
        if features['pigment_network']['type'] in ['atypical', 'branched']:
            border_score += 2
        if features['streaks']['detected']:
            border_score += 2
        if features['globules']['type'] == 'peripheral':
            border_score += 2
        b_score = min(border_score, 8)

        # C - Color (1-6) - number of colors
        c_score = min(features['color_analysis']['distinct_colors'], 6)

        # D - Dermoscopic structures (1-5)
        structures = 0
        if features['pigment_network']['detected']:
            structures += 1
        if features['globules']['detected']:
            structures += 1
        if features['streaks']['detected']:
            structures += 1
        if features['blue_white_veil']['detected']:
            structures += 1
        if features['regression']['detected']:
            structures += 1
        d_score = structures

        # Calculate Total Dermoscopy Score
        tds = (a_score * 1.3) + (b_score * 0.1) + (c_score * 0.5) + (d_score * 0.5)

        # Interpretation
        if tds > 5.45:
            classification = 'MELANOMA'
            recommendation = 'Immediate biopsy recommended'
        elif tds >= 4.75:
            classification = 'SUSPICIOUS'
            recommendation = 'Close monitoring or biopsy recommended'
        else:
            classification = 'BENIGN'
            recommendation = 'Routine follow-up'

        return {
            'asymmetry_score': a_score,
            'border_score': b_score,
            'color_score': c_score,
            'structures_score': d_score,
            'total_score': round(tds, 2),
            'classification': classification,
            'recommendation': recommendation
        }

    def assess_risk(self, features: Dict) -> Dict:
        """
        Overall risk assessment based on all detected features.
        """
        risk_factors = []
        risk_score = 0

        # High-risk features
        if features['blue_white_veil']['detected']:
            risk_factors.append('Blue-white veil present (high risk)')
            risk_score += 3

        if features['pigment_network']['type'] in ['atypical', 'branched']:
            risk_factors.append(f"Atypical pigment network ({features['pigment_network']['type']})")
            risk_score += 2

        if features['streaks']['type'] == 'radial':
            risk_factors.append('Radial streaming present')
            risk_score += 2

        if features['globules']['type'] in ['irregular', 'peripheral']:
            risk_factors.append(f"Irregular globules ({features['globules']['type']})")
            risk_score += 2

        if features['vascular_patterns']['type'] == 'linear_irregular':
            risk_factors.append('Atypical vascular pattern')
            risk_score += 2

        # Moderate risk features
        if features['regression']['detected']:
            risk_factors.append('Regression structures present')
            risk_score += 1

        if features['color_analysis']['distinct_colors'] >= 4:
            risk_factors.append(f"{features['color_analysis']['distinct_colors']} distinct colors")
            risk_score += 1

        if features['symmetry_analysis']['classification'] == 'highly_asymmetric':
            risk_factors.append('Highly asymmetric lesion')
            risk_score += 1

        # Overall risk level
        if risk_score >= 6:
            overall_risk = 'HIGH'
            urgency = 'Urgent dermatologist evaluation recommended'
        elif risk_score >= 3:
            overall_risk = 'MODERATE'
            urgency = 'Dermatologist evaluation recommended within 2 weeks'
        elif risk_score >= 1:
            overall_risk = 'LOW-MODERATE'
            urgency = 'Routine dermatologist follow-up recommended'
        else:
            overall_risk = 'LOW'
            urgency = 'Routine monitoring'

        return {
            'risk_level': overall_risk,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': urgency
        }

    def generate_overlays(self, img: np.ndarray, features: Dict) -> Dict:
        """
        Generate visual overlays highlighting detected features.
        Returns base64-encoded images.
        """
        overlays = {}

        # Create a copy for each overlay
        h, w = img.shape[:2]

        # Pigment network overlay
        if features['pigment_network']['detected']:
            network_overlay = img.copy()
            for coords in features['pigment_network']['coordinates']:
                if len(coords) > 0:
                    # Convert to numpy array and check if it has valid points
                    contour_array = np.array(coords)
                    if contour_array.size > 0 and len(contour_array.shape) >= 2:
                        # Ensure it has the correct shape for drawContours
                        if len(contour_array.shape) == 2:
                            contour_array = contour_array.reshape(-1, 1, 2)
                        cv2.drawContours(network_overlay, [contour_array], -1, (0, 255, 0), 2)
            overlays['pigment_network'] = self._encode_image(network_overlay)

        # Globules overlay
        if features['globules']['detected']:
            globules_overlay = img.copy()
            for x, y, r in features['globules']['coordinates']:
                cv2.circle(globules_overlay, (x, y), r, (255, 0, 0), 2)
                cv2.circle(globules_overlay, (x, y), 2, (255, 0, 0), -1)
            overlays['globules'] = self._encode_image(globules_overlay)

        # Streaks overlay
        if features['streaks']['detected']:
            streaks_overlay = img.copy()
            for x1, y1, x2, y2 in features['streaks']['coordinates']:
                cv2.line(streaks_overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
            overlays['streaks'] = self._encode_image(streaks_overlay)

        # Combined overlay showing all features
        combined_overlay = img.copy()

        # Draw all features on combined overlay with different colors
        if features['pigment_network']['detected']:
            for coords in features['pigment_network']['coordinates']:
                if len(coords) > 0:
                    contour_array = np.array(coords)
                    if contour_array.size > 0 and len(contour_array.shape) >= 2:
                        if len(contour_array.shape) == 2:
                            contour_array = contour_array.reshape(-1, 1, 2)
                        cv2.drawContours(combined_overlay, [contour_array], -1, (0, 255, 0), 1)

        if features['globules']['detected']:
            for x, y, r in features['globules']['coordinates']:
                cv2.circle(combined_overlay, (x, y), r, (255, 0, 0), 1)

        if features['streaks']['detected']:
            for x1, y1, x2, y2 in features['streaks']['coordinates']:
                cv2.line(combined_overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)

        if features['blue_white_veil']['detected']:
            for coords in features['blue_white_veil']['coordinates']:
                if len(coords) > 0:
                    contour_array = np.array(coords)
                    if contour_array.size > 0 and len(contour_array.shape) >= 2:
                        if len(contour_array.shape) == 2:
                            contour_array = contour_array.reshape(-1, 1, 2)
                        cv2.drawContours(combined_overlay, [contour_array], -1, (255, 255, 0), 1)

        overlays['combined'] = self._encode_image(combined_overlay)

        return overlays

    def _extract_coordinates(self, contours: List, max_count: int = 20) -> List:
        """Extract contour coordinates for JSON serialization."""
        coords = []
        for contour in contours[:max_count]:
            if len(contour) > 0:
                # Ensure contour has valid shape before squeezing
                squeezed = contour.squeeze()
                if squeezed.size > 0 and len(squeezed.shape) >= 1:
                    coords.append(squeezed.tolist())
        return coords

    def _encode_image(self, img: np.ndarray) -> str:
        """Encode image as base64 string."""
        _, buffer = cv2.imencode('.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return base64.b64encode(buffer).decode('utf-8')


# Singleton instance
_detector = None

def get_dermoscopy_detector():
    """Get singleton instance of dermoscopic feature detector."""
    global _detector
    if _detector is None:
        _detector = DermoscopicFeatureDetector()
        print("[OK] Dermoscopic feature detector initialized")
    return _detector
