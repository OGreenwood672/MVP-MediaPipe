import numpy as np
import cv2
import os
import shutil
import mediapipe as mp
import scipy.io as sio
from tqdm import tqdm
import fiftyone as fo
import fiftyone.utils.huggingface as fouh


from globals import ANNOTATIONS, CROP_SIZE, HEATMAPS_DIR, EDGE_DETECTION_DIR, JOINTS, MODEL_PATH, MPII_TO_MP_MAP, PLAIN_DIR, RGB_CROPS_DIR, INPUT_DIR


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

    left_img = int(center[0] - k_size // 2)
    top_img = int(center[1] - k_size // 2)
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
        return True
    return False

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

# def create_crop(frame, crop_center_x, crop_center_y, width, height):
    
#     left = crop_center_x - CROP_SIZE // 2
#     top = crop_center_y - CROP_SIZE // 2
#     right = left + CROP_SIZE
#     bottom = top + CROP_SIZE
    
#     if left < 0 or top < 0 or right > width or bottom > height:
#         return None
        
#     return frame[top:bottom, left:right]

def create_crop(frame, crop_center_x, crop_center_y, width, height):
    
    left = crop_center_x - CROP_SIZE // 2
    top = crop_center_y - CROP_SIZE // 2
    right = left + CROP_SIZE
    bottom = top + CROP_SIZE

    if len(frame.shape) == 3:
        crop = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=frame.dtype)
    else:
        crop = np.zeros((CROP_SIZE, CROP_SIZE), dtype=frame.dtype)

    src_left = max(0, left)
    src_top = max(0, top)
    src_right = min(width, right)
    src_bottom = min(height, bottom)

    dst_left = src_left - left
    dst_top = src_top - top
    dst_right = dst_left + (src_right - src_left)
    dst_bottom = dst_top + (src_bottom - src_top)

    if src_right > src_left and src_bottom > src_top:
        crop[dst_top:dst_bottom, dst_left:dst_right] = frame[src_top:src_bottom, src_left:src_right]
        
    return crop

def is_valid_landmark(landmark_name, landmark):
    return landmark.visibility >= 0.5 and landmark.presence >= 0.5 and landmark_name in JOINTS

def get_annotations(annotation_path):
    mat = sio.loadmat(annotation_path)
    release = mat['RELEASE']
    annolist = release['annolist'][0, 0]
    return annolist

def get_closest_landmark_joint(landmarks, width, height, mpi_joint, point):
    closest = (-1, 99999999999)
    for i, landmark in enumerate(landmarks):

        if MPII_TO_MP_MAP[mpi_joint] != i:
            continue

        distance_squared = (landmark.x * width - point[0])**2 + (landmark.y * height - point[1])**2
        if distance_squared < closest[1]:
            closest = (i, distance_squared)
        
    return closest[0] if closest[0] != -1 and closest[1] < CROP_SIZE ** 2 else None

def create_filename_to_index_map(img_list):
    filename_to_index = {}
    for index, img_entry in enumerate(img_list):
        try:
            name_field = img_entry['name']
            filename = str(name_field[0][0].item())
            filename_to_index[filename] = index
        except (IndexError, AttributeError):
            continue
    return filename_to_index


# Heatmaps for training
def create_heatmaps():

    generate_folders()

    dataset = fouh.load_from_hub("Voxel51/MPII_Human_Pose_Dataset", max_samples=800)

    annolist = get_annotations(ANNOTATIONS)
    img_list = annolist['image'][0]
    rect_list = annolist['annorect'][0]

    filename_map = create_filename_to_index_map(img_list)


    landmarker_obj = init_mediapipe(MODEL_PATH)
    with landmarker_obj as landmarker:

        print("Creating heatmaps...")
        for sample in dataset.iter_samples(progress=True):
            
            filepath = sample.filepath
            filename = os.path.basename(filepath)
            file_stem, _ = os.path.splitext(filename)

            if filename not in filename_map:
                continue
            index = filename_map[filename]

            rects_entry = rect_list[index]
            if rects_entry.size == 0:
                continue

            frame = cv2.imread(filepath)
            if frame is None:
                print(f"Could not load {filename}")
                continue

            height, width, _ = frame.shape

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = landmarker.detect(mp_image)
            if not result.pose_landmarks:
                print(f"No pose detected in {filename}")
                continue
            landmarks = result.pose_landmarks[0]

            plain_folder, rgb_crops_folder, edge_folder = None, None, None

            edges_full = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
            kernel = np.ones((3,3), np.uint8) # Thicken the edges, so they do not disappear on scaling
            edges_full = cv2.dilate(edges_full, kernel, iterations=1)

            for person_index, rect in enumerate(rects_entry):
                if 'annopoints' not in rect.dtype.names:
                    continue

                annopoints_entry = rect['annopoints']
                if annopoints_entry.size == 0:
                    continue
                    
                try:
                    point_wrapper = annopoints_entry.flatten()[0]
                    if 'point' not in point_wrapper.dtype.names:
                        continue
                    points_array = point_wrapper['point']
                    if points_array.size == 0:
                        continue
                    points = points_array.flatten()[0].flatten()
                except (IndexError, AttributeError):
                    continue
                
                for point_mat in points:

                    try:
                        px = float(point_mat['x'].flatten()[0])
                        py = float(point_mat['y'].flatten()[0])
                        p_id = int(point_mat['id'].flatten()[0])
                        # p_visibility = float(point_mat['is_visible'].flatten()[0] if 'is_visible' in point_mat.dtype.names else 1.0)
                    except (IndexError, AttributeError):
                        continue

                    closest_mp_landmark_id = get_closest_landmark_joint(landmarks, width, height, p_id, (px, py))
                    if not closest_mp_landmark_id:
                        continue

                    landmark_name = mp.solutions.pose.PoseLandmark(closest_mp_landmark_id).name
                    joint_id = JOINTS.index(landmark_name)
                    closest_mp_landmark = landmarks[closest_mp_landmark_id]
                    
                    # joint location
                    center = int(closest_mp_landmark.x * width), int(closest_mp_landmark.y * height)

                    # Crop center is mediapipe joint center
                    rgb_crop = create_crop(frame, center[0], center[1], width, height)
                    edges_crop = create_crop(edges_full, center[0], center[1], width, height)

                    if rgb_crop is None or edges_crop is None:
                        continue
                    
                    dx, dy = center[0] - px, center[1] - py
                    # Actual heatmap has the true joint location
                    heatmap_crop = np.zeros((CROP_SIZE, CROP_SIZE), dtype=np.float32)
                    on_screen = render_gaussian(
                        heatmap_crop,
                        (CROP_SIZE // 2 - dx, CROP_SIZE // 2 - dy),
                        std=3
                    )
                    if not on_screen:
                        continue

                    heatmap_crop = (heatmap_crop * 255).astype(np.uint8)

                    if not plain_folder:
                        plain_folder, rgb_crops_folder, edge_folder = generate_sub_folders(file_stem)

                    # Create plain ol' heatmap and save
                    single_joint_heatmap = render_heatmap(frame.shape, [closest_mp_landmark])
                    cv2.imwrite(os.path.join(plain_folder, f"{joint_id}+plain.png"), single_joint_heatmap)


                    # Saved hints and assoicaiated heatmaps
                    filename = f"{person_index}+{joint_id}"
                    cv2.imwrite(f"{rgb_crops_folder}/{filename}+rgb.png", rgb_crop)
                    cv2.imwrite(f"{rgb_crops_folder}/{filename}+heatmap.png", heatmap_crop)
                    cv2.imwrite(f"{edge_folder}/{filename}+edge.png", edges_crop)
                    cv2.imwrite(f"{edge_folder}/{filename}+heatmap.png", heatmap_crop)
    

if __name__ == "__main__":
    create_heatmaps()