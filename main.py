import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import mediapipe as mp

from models import Unet
from humanpose import noise_handler, alphas, alphas_cumprod, betas

from globals import HINT, JOINTS, MODEL_IMG_SIZE, MODEL_SAVE_PATH
from get_data import CROP_SIZE, MODEL_PATH, create_crop, create_heatmaps, create_visual_overlay, init_mediapipe, is_valid_landmark, paste_crop_to_canvas, render_gaussian

transform_rgb = transforms.Compose([
    transforms.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

transform_gray = transforms.Compose([
    transforms.Resize((MODEL_IMG_SIZE, MODEL_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def load_diffusion_model(diffusion_model_path, hint="rgb"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_channels = 1
    if hint == "rgb":
        input_channels = 4
    elif hint == "edge":
        input_channels = 2
    elif hint == "both":
        input_channels = 5



    model = Unet(input_channels=input_channels, output_channels=1, time_dimension=64, num_joints=len(JOINTS)).to(device)
    model.load_state_dict(torch.load(diffusion_model_path, map_location=device))
    model.eval()

    print("Model loaded successfully")

    return model, device

def denoise(diffusion_model, timestep, img, hint_tensor=None, joint_tensor=None):

    for t in reversed(range(0, int(timestep))):
        
        if hint_tensor is not None:
            model_input = torch.cat((img, hint_tensor), dim=1)
        else:
            model_input = img
        
        predicted_noise = diffusion_model(model_input, torch.tensor([t], device=device).long(), joint_index=joint_tensor)
        

        # Reverse Diffusion Sampling Loop - see notes.md
        alpha = alphas[t]
        alpha_hat = alphas_cumprod[t]
        beta = betas[t]
        noise = torch.randn_like(img) if t > 0 else torch.zeros_like(img)
        
        a = 1 / torch.sqrt(alpha)
        b = (1 - alpha) / (torch.sqrt(1 - alpha_hat))
        
        img = a * (img - b * predicted_noise)
        
        sigma = torch.sqrt(beta)
        img = img + sigma * noise
    
    return img


@torch.no_grad()
def save_examples(diffusion_model, sample_heatmap_path, hint_path, joint_id, timestep=50, save_folder="output-joint"):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model.eval()

    os.makedirs(save_folder, exist_ok=True)
    save_noisy = os.path.join(save_folder, f"{joint_id}_{timestep}_noisy.jpg")
    save_denoised = os.path.join(save_folder, f"{joint_id}_{timestep}_denoised.jpg")
    save_original = os.path.join(save_folder, f"{joint_id}_{timestep}_original.jpg")
    save_hint = os.path.join(save_folder, f"{joint_id}_{timestep}_hint.jpg")

    print("Model loaded")
    print(f"Analysing: {sample_heatmap_path}, with hint {hint_path}")
    print(f"Looking at {joint_id} at timestep {timestep}")

    heatmap = transform_gray(Image.open(sample_heatmap_path).convert('L')).unsqueeze(0).to(device)

    t_tensor = torch.tensor([timestep], device=device).long()
    noisy_heatmap, _ = noise_handler(device)(heatmap, t_tensor)
    
    save_image((noisy_heatmap + 1) / 2, save_noisy)

    hint_tensor = None
    if diffusion_model.input_channels == 4:
        hint_tensor = transform_rgb(Image.open(hint_path).convert('RGB')).unsqueeze(0).to(device)
    elif diffusion_model.input_channels == 2:
        hint_tensor = transform_gray(Image.open(hint_path).convert('L')).unsqueeze(0).to(device)

    joint_tensor = torch.tensor([joint_id], device=device).long()
    
    img_current = denoise(diffusion_model, timestep, noisy_heatmap, hint_tensor, joint_tensor)

    save_image((img_current + 1) / 2, save_denoised)
    save_image((heatmap + 1) / 2, save_original)
    if hint_tensor is not None:
        save_image((hint_tensor + 1) / 2, save_hint)

@torch.no_grad()
def generate_heatmap_with_refinements(diffusion_model, full_image_path, hint_mode="rgb", timestep=500, save_folder="output-pose"):
    os.makedirs(save_folder, exist_ok=True)
    save_refined = os.path.join(save_folder, f"refined_{timestep}.jpg")
    save_original = os.path.join(save_folder, f"original.jpg")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    diffusion_model.eval()

    frame = cv2.imread(full_image_path)
    height, width, _ = frame.shape
    edges_full_img = None
    if hint_mode == "edge" or hint_mode == "both":
        edges_full_img = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 100, 200)
        kernel = np.ones((3,3), np.uint8) # Thicken the edges, so they do not disappear on scaling
        edges_full_img = cv2.dilate(edges_full_img, kernel, iterations=1)
    
    # Our Refined HeatMap and MediaPipe original HeatMap blank canvases
    refined_heatmap = np.zeros((height, width), dtype=np.float32)
    mp_heatmap = np.zeros((height, width), dtype=np.float32)

    landmarker = init_mediapipe(MODEL_PATH)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    landmarks = landmarker.detect(mp_image).pose_landmarks[0]

    for index, landmark in enumerate(landmarks):

        landmark_name = mp.solutions.pose.PoseLandmark(index).name
        if not is_valid_landmark(landmark_name, landmark):
            continue
        
        # Landmark location
        center_x = int(landmark.x * width)
        center_y = int(landmark.y * height)

        render_gaussian(mp_heatmap, (center_x, center_y), std=5)

        hint_tensor = None
        if hint_mode == "edge":
            edge_crop = create_crop(edges_full_img, center_x, center_y, width, height)
            edge_crop_img = Image.fromarray(edge_crop).convert('L')
            hint_tensor = transform_gray(edge_crop_img).unsqueeze(0).to(device)
        elif hint_mode == "rgb":
            rgb_crop = create_crop(frame, center_x, center_y, width, height)
            rgb_img = Image.fromarray(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
            hint_tensor = transform_rgb(rgb_img).unsqueeze(0).to(device)
        elif hint_mode == "both":
            rgb_crop = create_crop(frame, center_x, center_y, width, height)
            rgb_img = Image.fromarray(cv2.cvtColor(rgb_crop, cv2.COLOR_BGR2RGB))
            rgb_tensor = transform_rgb(rgb_img)
            edge_crop = create_crop(edges_full_img, center_x, center_y, width, height)
            edge_crop_img = Image.fromarray(edge_crop).convert('L')
            edge_tensor = transform_gray(edge_crop_img)
            hint_tensor = torch.cat((rgb_tensor, edge_tensor), dim=0).unsqueeze(0).to(device)
        
        # Create heatmap from MediaPipe for single joint (for the model)
        mp_single_joint_map = np.zeros((height, width), dtype=np.float32)
        render_gaussian(mp_single_joint_map, (center_x, center_y), std=5)
        mp_single_joint_map = (mp_single_joint_map * 255).astype(np.uint8)

        mp_heatmap_crop = create_crop(mp_single_joint_map, center_x, center_y, width, height)
        if mp_heatmap_crop is None: continue

        heatmap_for_model = transform_gray(Image.fromarray(mp_heatmap_crop).convert('L')).unsqueeze(0).to(device)

        timestep_tensor = torch.tensor([timestep], device=device).long()
        noisy_heatmap, _ = noise_handler(device)(heatmap_for_model, timestep_tensor)

        joint_idx_val = JOINTS.index(landmark_name)
        joint_tensor = torch.tensor([joint_idx_val], device=device).long()

        img = denoise(diffusion_model, timestep, noisy_heatmap, hint_tensor, joint_tensor)

        # Remove any extreme values and normalise back to [0, 1]
        heatmap_generated = torch.clamp((img + 1) / 2, 0, 1).squeeze().cpu().numpy()
        heatmap_small = cv2.resize(heatmap_generated, (CROP_SIZE, CROP_SIZE))

        paste_crop_to_canvas(refined_heatmap, heatmap_small, center_x, center_y)
    
    cv2.imwrite(save_original, create_visual_overlay(frame, mp_heatmap))
    print(f"Saved MediaPipe {save_original}")

    cv2.imwrite(save_refined, create_visual_overlay(frame, refined_heatmap))
    print(f"Saved Refined to {save_refined}")


if __name__ == "__main__":
    CREATE_HEATMAPS = False
    if CREATE_HEATMAPS:
        create_heatmaps()

    # diffusion_model, device = load_diffusion_model("unet_human_pose_rgb.pth")

    # sample_heatmap_path = r"heatmaps\rgb_crops\50-Ways-to-Fall_mp4-74_jpg.rf.e77565139ac67ed988572292de9bb142\LEFT_ELBOW+0+heatmap.png"
    # hint_path = r"heatmaps\rgb_crops\50-Ways-to-Fall_mp4-74_jpg.rf.e77565139ac67ed988572292de9bb142\LEFT_ELBOW+0+rgb.png"
    # save_examples(diffusion_model, sample_heatmap_path, hint_path, 5, 90)

    # diffusion_model, device = load_diffusion_model("unet_human_pose_edge.pth", hint="edge")

    # sample_heatmap_path = r"heatmaps\edge_detection\50-Ways-to-Fall_mp4-74_jpg.rf.e77565139ac67ed988572292de9bb142\LEFT_ELBOW+0+heatmap.png"
    # hint_path = r"heatmaps\edge_detection\50-Ways-to-Fall_mp4-74_jpg.rf.e77565139ac67ed988572292de9bb142\LEFT_ELBOW+0+edge.png"
    # save_examples(diffusion_model, sample_heatmap_path, hint_path, 13, 90)
    

    full_img_path = r"input\50-Ways-to-Fall_mp4-147_jpg.rf.d2498ea8b48a31bb97d903af5537b2b7.jpg"
    diffusion_model, device = load_diffusion_model(MODEL_SAVE_PATH, hint=HINT)

    for t in [100, 200, 300, 400, 500, 600]:
        generate_heatmap_with_refinements(
            diffusion_model, 
            full_img_path, 
            hint_mode=HINT, 
            timestep=t
        )