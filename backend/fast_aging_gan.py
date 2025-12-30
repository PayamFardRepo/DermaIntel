"""
Fast-AgingGAN Model Integration

Based on: https://github.com/HasnainRaz/Fast-AgingGAN
A deep learning model to age faces in the wild, runs at 60+ fps on GPU.
"""

import os
import io
import logging
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Model download URL (GitHub releases)
MODEL_URL = "https://github.com/HasnainRaz/Fast-AgingGAN/raw/master/pretrained_model/state_dict.pth"
MODEL_DIR = Path(__file__).parent / "models" / "fast_aging_gan"
MODEL_PATH = MODEL_DIR / "state_dict.pth"


class ResidualBlock(nn.Module):
    """Residual block for the Generator."""

    def __init__(self, in_features: int):
        super(ResidualBlock, self).__init__()

        conv_block = [
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.BatchNorm2d(in_features)
        ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    """Generator model for face aging using CycleGAN architecture."""

    def __init__(self, ngf: int = 32, n_residual_blocks: int = 9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, 7),
            nn.BatchNorm2d(ngf),
            nn.ReLU()
        ]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class FastAgingGAN:
    """
    Fast-AgingGAN wrapper for easy inference.

    This model ages faces to appear older (~60-70 years old).
    Input should be 512x512 images with a face of at least 256x256 pixels.
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Fast-AgingGAN model.

        Args:
            device: 'cuda', 'cpu', or None for auto-detection
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model: Optional[Generator] = None
        self.is_loaded = False

        logger.info(f"FastAgingGAN initialized on device: {self.device}")

    def download_model(self) -> bool:
        """Download the pretrained model if not present."""
        if MODEL_PATH.exists():
            logger.info(f"Model already exists at {MODEL_PATH}")
            return True

        try:
            logger.info(f"Downloading model from {MODEL_URL}...")
            MODEL_DIR.mkdir(parents=True, exist_ok=True)

            # Download with progress
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

            logger.info(f"Model downloaded successfully to {MODEL_PATH}")
            return True

        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return False

    def load_model(self) -> bool:
        """Load the pretrained model."""
        if self.is_loaded:
            return True

        # Download if needed
        if not MODEL_PATH.exists():
            if not self.download_model():
                return False

        try:
            logger.info("Loading Fast-AgingGAN model...")

            self.model = Generator(ngf=32, n_residual_blocks=9)
            self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()

            self.is_loaded = True
            logger.info("Fast-AgingGAN model loaded successfully!")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess an image for the model.

        Args:
            image: PIL Image

        Returns:
            Preprocessed tensor ready for inference
        """
        # Resize to 512x512
        image = image.resize((512, 512), Image.Resampling.LANCZOS)

        # Convert to numpy array and normalize to [-1, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - 0.5) / 0.5

        # Convert to tensor [C, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        return img_tensor.to(self.device)

    def postprocess(self, tensor: torch.Tensor) -> Image.Image:
        """
        Convert model output tensor back to PIL Image.

        Args:
            tensor: Model output tensor

        Returns:
            PIL Image
        """
        # Remove batch dimension and move to CPU
        img = tensor.squeeze(0).cpu().detach()

        # Convert from [-1, 1] to [0, 255]
        img = (img * 0.5 + 0.5).clamp(0, 1)
        img = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        return Image.fromarray(img)

    def age_face(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Age a face in the input image.

        Args:
            image: PIL Image containing a face

        Returns:
            Aged face image, or None if processing failed
        """
        if not self.is_loaded:
            if not self.load_model():
                return None

        try:
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Store original size for resizing back
            original_size = image.size

            # Preprocess
            input_tensor = self.preprocess(image)

            # Run inference
            with torch.no_grad():
                output_tensor = self.model(input_tensor)

            # Postprocess
            aged_image = self.postprocess(output_tensor)

            # Resize back to original size if needed
            if original_size != (512, 512):
                aged_image = aged_image.resize(original_size, Image.Resampling.LANCZOS)

            return aged_image

        except Exception as e:
            logger.error(f"Error aging face: {e}")
            return None

    def age_face_bytes(self, image_bytes: bytes) -> Optional[bytes]:
        """
        Age a face from image bytes.

        Args:
            image_bytes: Image file bytes

        Returns:
            Aged image as JPEG bytes, or None if processing failed
        """
        try:
            # Load image from bytes
            image = Image.open(io.BytesIO(image_bytes))

            # Age the face
            aged_image = self.age_face(image)

            if aged_image is None:
                return None

            # Convert back to bytes
            output_buffer = io.BytesIO()
            aged_image.save(output_buffer, format='JPEG', quality=90)
            output_buffer.seek(0)

            return output_buffer.getvalue()

        except Exception as e:
            logger.error(f"Error processing image bytes: {e}")
            return None


# Singleton instance for reuse
_aging_model: Optional[FastAgingGAN] = None


def get_aging_model() -> FastAgingGAN:
    """Get or create the singleton FastAgingGAN instance."""
    global _aging_model
    if _aging_model is None:
        _aging_model = FastAgingGAN()
    return _aging_model


def is_model_available() -> bool:
    """Check if the model file exists."""
    return MODEL_PATH.exists()


def download_model() -> bool:
    """Download the model if not present."""
    model = get_aging_model()
    return model.download_model()
