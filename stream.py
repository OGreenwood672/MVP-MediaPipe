
###! EXPERIMENTATION ONLY

import cv2
import numpy as np
import mediapipe as mp

# --- Configuration ---
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "./pose_landmarker_lite.task"

# --- 1. Pre-calculate the Gaussian Kernel ---
def generate_gaussian_kernel(size=71, sigma=20):
    """
    Creates a 2D Gaussian kernel (a smooth hill).
    """
    k = cv2.getGaussianKernel(size, sigma)
    kernel = k @ k.T 
    kernel = kernel / kernel.max() # Normalize peak to 1.0
    return kernel

GAUSSIAN_KERNEL = generate_gaussian_kernel()
KERNEL_R = GAUSSIAN_KERNEL.shape[0] // 2

def render_heatmap_no_accumulation(frame_shape, landmarks):
    h, w = frame_shape[:2]
    
    # Use Float32 matrix
    heatmap_layer = np.zeros((h, w), dtype=np.float32)
    
    for norm_lm in landmarks:
        cx = int(norm_lm.x * w)
        cy = int(norm_lm.y * h)
        
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue

        # Calculate bounds (clipping to screen edges)
        x1, x2 = max(0, cx - KERNEL_R), min(w, cx + KERNEL_R + 1)
        y1, y2 = max(0, cy - KERNEL_R), min(h, cy + KERNEL_R + 1)
        
        # Calculate bounds on the kernel stamp
        kx1 = max(0, KERNEL_R - (cx - x1))
        kx2 = kx1 + (x2 - x1)
        ky1 = max(0, KERNEL_R - (cy - y1))
        ky2 = ky1 + (y2 - y1)
        
        # --- THE FIX: USE MAXIMUM INSTEAD OF ADD ---
        # We take the brighter of the two values (current vs new).
        # This prevents "double counting" heat.
        heatmap_slice = heatmap_layer[y1:y2, x1:x2]
        kernel_slice = GAUSSIAN_KERNEL[ky1:ky2, kx1:kx2]
        
        heatmap_layer[y1:y2, x1:x2] = np.maximum(heatmap_slice, kernel_slice)

    # Convert to Color
    heatmap_uint8 = (heatmap_layer * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    
    return heatmap_color

def main():
    # Use IMAGE mode for raw, unsmoothed data
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=False
    )

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Starting Heatmap (No Accumulation). Press ESC to exit.")

    with PoseLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                
                # Synchronous detection
                detection_result = landmarker.detect(mp_image)

                output_frame = frame_bgr.copy()

                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    
                    # Generate the non-accumulating heatmap
                    heatmap = render_heatmap_no_accumulation(frame_bgr.shape, landmarks)
                    
                    # Blend: 60% Video, 60% Heatmap
                    output_frame = cv2.addWeighted(frame_bgr, 0.6, heatmap, 0.6, 0)

                cv2.imshow("Heatmap", output_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()