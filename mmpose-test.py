###! EXPERIMENTATION ONLY

import torch
import cv2
import numpy as np
import os
import time
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# --- 1. PATCH FOR PYTORCH 2.6+ ---
_original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = safe_load

# --- 2. SETUP ---
register_all_modules()

device = 'cuda'
config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-384x288.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-384x288-c161b7de_20220915.pth'

if not os.path.exists(config_file):
    os.system(f"mim download mmpose --config {config_file[:-3]} --dest .")

print(f"Loading Model on {device}...")
model = init_model(config_file, checkpoint_file, device=device)
model.cfg.model.test_cfg['output_heatmaps'] = True 

# Get Model Input Size (Usually W=288, H=384)
model_w, model_h = model.cfg.codec.input_size
model_aspect = model_w / model_h 

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Running 'Robot Eye View' Mode... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    
    # --- 1. PRE-PROCESS (Mimic what the model sees) ---
    h, w = frame.shape[:2]
    
    # Calculate the center crop to match model aspect ratio (3:4)
    target_h = h
    target_w = int(target_h * model_aspect) 
    start_x = (w - target_w) // 2
    end_x = start_x + target_w
    
    # Crop the center
    crop_img = frame[:, start_x:end_x]
    
    # Resize to exact model input size (288x384)
    # THIS is the image the model actually "sees"
    model_input_img = cv2.resize(crop_img, (model_w, model_h))

    # --- 2. INFERENCE ---
    results = inference_topdown(model, model_input_img)
    
    if hasattr(results[0], 'pred_fields') and 'heatmaps' in results[0].pred_fields:
        heatmap_tensor = results[0].pred_fields.heatmaps.cpu().numpy()
        
        # --- 3. RAW HEATMAP VISUALIZATION ---
        
        # Collapse all joints into one map
        raw_heatmap = np.max(heatmap_tensor, axis=0)
        
        # Resize heatmap (72x96) up to Model Input Size (288x384)
        # We use CUBIC interpolation to smooth the tiny grid, 
        # or NEAREST if you want to see the actual blocky pixels.
        heatmap_overlay = cv2.resize(raw_heatmap, (model_w, model_h), interpolation=cv2.INTER_CUBIC)
        
        # Normalize raw values (0.0 to 1.0) to (0 to 255)
        # We perform a clip to remove background noise (optional)
        heatmap_overlay = np.maximum(heatmap_overlay, 0)
        
        # Note: I removed the squaring (**2) so you see the raw shape density
        norm_heatmap = cv2.normalize(heatmap_overlay, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        colored_map = cv2.applyColorMap(norm_heatmap, cv2.COLORMAP_JET)
        
        # Overlay onto the Model Input Image
        # This alignment is mathematically perfect because they are the same coordinate space
        final_view = cv2.addWeighted(model_input_img, 0.5, colored_map, 0.5, 0)
        
    else:
        final_view = model_input_img

    # --- 4. DISPLAY ---
    # The result is small (288x384). Let's scale it up 2x just for your screen visibility.
    # We scale the MERGED result, so they can't drift apart.
    display_view = cv2.resize(final_view, (model_w * 2, model_h * 2), interpolation=cv2.INTER_NEAREST)
    
    cv2.putText(display_view, "Robot's Eye View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Raw Model Output', display_view)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()