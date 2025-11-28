import os

JOINTS = ["LEFT_EYE", "RIGHT_EYE", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_KNEE", "RIGHT_KNEE", "LEFT_HIP", "RIGHT_HIP"]

# HINT = "rgb"
# MODEL_SAVE_PATH = "unet_human_pose_rgb.pth"

# HINT = "edge"
# MODEL_SAVE_PATH = "unet_human_pose_edge.pth"

HINT = "both"
MODEL_SAVE_PATH = "unet_human_pose_both.pth"

# HINT = None
# MODEL_SAVE_PATH = "unet_human_pose.pth"

MODEL_PATH = "./pose_landmarker_lite.task"

JITTERS_PER_JOINT = 5
MAX_JITTER = 25
CROP_SIZE = 50

T = 1000
EPOCHS = 40
BATCH_SIZE = 10

MODEL_IMG_SIZE = 256


HEATMAPS_DIR = "./heatmaps"
EDGE_DETECTION_DIR = os.path.join(HEATMAPS_DIR, "edge_detection")
PLAIN_DIR = os.path.join(HEATMAPS_DIR, "plain")
RGB_CROPS_DIR = os.path.join(HEATMAPS_DIR, "rgb_crops")
INPUT_DIR = "./input"
