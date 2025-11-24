import numpy as np
import random
import cv2
import os
import shutil
import mediapipe as mp

from globals import CROP_SIZE, JITTERS_PER_JOINT, JOINTS, HEATMAPS_DIR, EDGE_DETECTION_DIR, MAX_JITTER, MODEL_PATH, PLAIN_DIR, RGB_CROPS_DIR, INPUT_DIR


def init_mediapipe(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=False
    )
    return PoseLandmarker.create_from_options(options)


def render_gaussian(heatmap, center, std, visibility=1.0):

    h, w = heatmap.shape[:2]

    k_size = int(6 * std + 1)
    k_size += 1 if k_size % 2 == 0 else 0 
    
    x = np.arange(k_size) - k_size // 2
    y = np.arange(k_size) - k_size // 2
    xx, yy = np.meshgrid(x, y)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * std**2))
    
    # Scale by visibility
    kernel = kernel / kernel.max() * visibility

    left_img = center[0] - k_size // 2
    top_img = center[1] - k_size // 2
    right_img = left_img + k_size
    bottom_img = top_img + k_size

    left_kernel, top_kernel = 0, 0
    right_kernel, bottom_kernel = k_size, k_size

    # Prevent drawing off screen
    if left_img < 0: 
        left_kernel = -left_img
        left_img = 0
    if top_img < 0: 
        top_kernel = -top_img
        top_img = 0
    if right_img > w: 
        right_kernel -= (right_img - w)
        right_img = w
    if bottom_img > h: 
        bottom_kernel -= (bottom_img - h)
        bottom_img = h

    # Don't draw if its completely offscreen
    if right_img > left_img and bottom_img > top_img:
        heatmap[top_img:bottom_img, left_img:right_img] = np.maximum(
            heatmap[top_img:bottom_img, left_img:right_img],
            kernel[top_kernel:bottom_kernel, left_kernel:right_kernel]
        )

def paste_crop_to_canvas(canvas, crop, center_x, center_y):
    h_full, w_full = canvas.shape[:2]
    half = CROP_SIZE // 2
    left, top = center_x - half, center_y - half
    right, bottom = left + CROP_SIZE, top + CROP_SIZE

    left_safe, top_safe = max(0, left), max(0, top)
    right_safe, bottom_safe = min(w_full, right), min(h_full, bottom)

    hm_top, hm_left = top_safe - top, left_safe - left
    hm_bottom, hm_right = hm_top + (bottom_safe - top_safe), hm_left + (right_safe - left_safe)

    canvas[top_safe:bottom_safe, left_safe:right_safe] = np.maximum(
        canvas[top_safe:bottom_safe, left_safe:right_safe],
        crop[hm_top:hm_bottom, hm_left:hm_right]
    )

def create_visual_overlay(background, heatmap):

    heatmap_normalised = (heatmap * 255).astype(np.uint8)
    heatmap_coloured = cv2.applyColorMap(heatmap_normalised, cv2.COLORMAP_JET)

    mask = heatmap_normalised > 20 
    overlay = background.copy()
    blended_pixels = (
        0.6 * background[mask].astype(np.float32) + 
        0.4 * heatmap_coloured[mask].astype(np.float32)
    )
    overlay[mask] = blended_pixels.astype(np.uint8)
    return overlay

def render_heatmap(frame_shape, landmarks):

    height, width, _ = frame_shape
    heatmap = np.zeros((height, width), dtype=np.float32)
    for landmark in landmarks:

        center_x = int(landmark.x * width)
        center_y = int(landmark.y * height)
        std = 3 + 20 * (1.0 - landmark.visibility)
        render_gaussian(heatmap, (center_x, center_y), std, landmark.visibility)

    heatmap_normalised = (heatmap * 255).astype(np.uint8)
    return cv2.cvtColor(heatmap_normalised, cv2.COLOR_GRAY2BGR)

def generate_folders():
    if os.path.exists(HEATMAPS_DIR):
        print("Deleting existing heatmaps...")
        shutil.rmtree(HEATMAPS_DIR)

    os.makedirs(HEATMAPS_DIR, exist_ok=True)
    os.makedirs(EDGE_DETECTION_DIR, exist_ok=True)
    os.makedirs(PLAIN_DIR, exist_ok=True)
    os.makedirs(RGB_CROPS_DIR, exist_ok=True)


def generate_sub_folders(file_stem):

    plain_folder = os.path.join(PLAIN_DIR, file_stem)
    os.makedirs(plain_folder, exist_ok=True)
    rgb_crops_folder = os.path.join(RGB_CROPS_DIR, file_stem)
    os.makedirs(rgb_crops_folder, exist_ok=True)
    edge_folder = os.path.join(EDGE_DETECTION_DIR, file_stem)
    os.makedirs(edge_folder, exist_ok=True)

    return plain_folder, rgb_crops_folder, edge_folder

def create_crop(frame, crop_center_x, crop_center_y, width, height):
    
    left = crop_center_x - CROP_SIZE // 2
    top = crop_center_y - CROP_SIZE // 2
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE
    
    if left < 0 or top < 0 or right > width or bottom > height:
        return None
        
    return frame[top:bottom, left:right]

def is_valid_landmark(landmark_name, landmark):
    return not ((landmark.visibility < 0.5 or landmark.presence < 0.5) or not landmark_name in JOINTS)
   

# Heatmaps for training
def create_heatmaps():

    generate_folders()

    landmarker_obj = init_mediapipe(MODEL_PATH)
    with landmarker_obj as landmarker:

        for filename in os.listdir(INPUT_DIR):
            file_stem, _ = os.path.splitext(filename)        

            image_path = os.path.join(INPUT_DIR, filename)
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Could not load {filename}")
                continue

            height, width, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            if not result.pose_landmarks:
                print(f"No pose detected in {filename}")
                continue

            plain_folder, rgb_crops_folder, edge_folder = generate_sub_folders(file_stem)

            edges_full = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
            kernel = np.ones((3,3), np.uint8) # Thicken the edges, so they do not disappear on sacling
            edges_full = cv2.dilate(edges_full, kernel, iterations=1)

            for idx, landmark in enumerate(result.pose_landmarks[0]):

                landmark_name = mp.solutions.pose.PoseLandmark(idx).name

                if not is_valid_landmark(landmark_name, landmark):
                    continue
                
                # joint location
                center = int(landmark.x * width), int(landmark.y * height)

                # Create plain ol' heatmap and save
                single_joint_heatmap = render_heatmap(frame.shape, [landmark])
                cv2.imwrite(os.path.join(plain_folder, f"{landmark_name}+plain.png"), single_joint_heatmap)

                for j in range(JITTERS_PER_JOINT):
                    
                    dx = random.randint(-MAX_JITTER, MAX_JITTER)
                    dy = random.randint(-MAX_JITTER, MAX_JITTER)
                    crop_center_x = center[0] + dx
                    crop_center_y = center[1] + dy
                    
                    rgb_crop = create_crop(frame, crop_center_x, crop_center_y, width, height)
                    edges_crop = create_crop(edges_full, crop_center_x, crop_center_y, width, height)

                    if rgb_crop is None or edges_crop is None:
                        continue

                    heatmap_crop = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
                    render_gaussian(
                        heatmap_crop,
                        (CROP_SIZE // 2 - dx, CROP_SIZE // 2 - dy),
                        std=3
                    )
                    heatmap_crop = (heatmap_crop * 255).astype(np.uint8)

                    # Saved hints and assoicaiated heatmaps
                    filename = f"{landmark_name}+{j}"
                    cv2.imwrite(f"{rgb_crops_folder}/{filename}+rgb.png", rgb_crop)
                    cv2.imwrite(f"{rgb_crops_folder}/{filename}+heatmap.png", heatmap_crop)
                    cv2.imwrite(f"{edge_folder}/{filename}+edge.png", edges_crop)
                    cv2.imwrite(f"{edge_folder}/{filename}+heatmap.png", heatmap_crop)

if __name__ == "__main__":
    create_heatmaps()