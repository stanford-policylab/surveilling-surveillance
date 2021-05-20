"""Define constants to be used throughout the repository."""
import os
from detectron2.data.catalog import Metadata
# Main paths

# Dataset constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# US latitude/longitude boundaries
US_N = 49.4
US_S = 24.5
US_E = -66.93
US_W = -124.784

# Test image
TEST_IMG_PATH = [".circleci/images/test_image.png"] * 2


SANDBOX_PATH = './sandbox'
TB_PATH = os.path.join(SANDBOX_PATH, 'tb')

META = Metadata()
META.thing_classes = ["Camera", "Camera"]
META.thing_colors = [[20, 200, 60], [11, 119, 32]]
