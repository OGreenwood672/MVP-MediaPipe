import os

JOINTS = [
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "NOSE",
    "RIGHT_ANKLE",
    "LEFT_ANKLE"
]

MPII_TO_MP_MAP = {
    0: 28,  # Right Ankle
    1: 26,  # Right Knee
    2: 24,  # Right Hip
    3: 23,  # Left Hip
    4: 25,  # Left Knee
    5: 27,  # Left Ankle
    6: -1,  # Pelvis
    7: -1,  # Thorax
    8: -1,  # Upper Neck
    9: 0,   # Head Top (Use MP Nose (0))
    10: 16, # Right Wrist
    11: 14, # Right Elbow
    12: 12, # Right Shoulder
    13: 11, # Left Shoulder
    14: 13, # Left Elbow
    15: 15  # Left Wrist
}

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
CROP_SIZE = 130

T = 800
EPOCHS = 15
BATCH_SIZE = 16

MODEL_IMG_SIZE = 256


HEATMAPS_DIR = "./heatmaps"
EDGE_DETECTION_DIR = os.path.join(HEATMAPS_DIR, "edge_detection")
PLAIN_DIR = os.path.join(HEATMAPS_DIR, "plain")
RGB_CROPS_DIR = os.path.join(HEATMAPS_DIR, "rgb_crops")
# INPUT_DIR = "./input"
INPUT_DIR = "images"

ANNOTATIONS = "./mpii_human_pose_v1_u12_1.mat"