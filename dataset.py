from torchvision import transforms
import os
from PIL import Image
import mediapipe as mp
import torch
from torch.utils.data import Dataset

from globals import JOINTS

class RGBDataset(Dataset):
    def __init__(self, rgb_dir):

        self.rgb_dir = rgb_dir
        # [photoname + joint + jitter number + type + .png]
        self.rgb_files = [folder + "+" + f for folder in os.listdir(rgb_dir) for f in os.listdir(os.path.join(rgb_dir, folder)) if f.endswith("+rgb.png")]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        self.mp_pose = mp.solutions.pose.PoseLandmark

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_filename = self.rgb_files[idx]
                
        split_name = rgb_filename.split("+")
        photo_folder, joint_name, filename = split_name[0], split_name[1], '+'.join(split_name[1:])

        heatmap_filename = filename.replace("+rgb.png", "+heatmap.png")
        
        try:
            joint_id = JOINTS.index(joint_name)
        except ValueError:
            joint_id = -1
            print("No Joint Name found for: ", rgb_filename)
        
        rgb_path = os.path.join(self.rgb_dir, photo_folder, filename)
        hm_path = os.path.join(self.rgb_dir, photo_folder, heatmap_filename)
        
        rgb_img = Image.open(rgb_path).convert("RGB")
        hm_img = Image.open(hm_path).convert("L")
        
        rgb_tensor = self.transform(rgb_img)
        hm_tensor = self.transform(hm_img)
        
        return rgb_tensor, hm_tensor, torch.tensor(joint_id).long()
    
class EdgeDataset(Dataset):
    def __init__(self, edge_dir):

        self.edge_dir = edge_dir
        self.edge_files = [folder + "+" + f for folder in os.listdir(edge_dir) for f in os.listdir(os.path.join(edge_dir, folder)) if f.endswith("+edge.png")]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        self.mp_pose = mp.solutions.pose.PoseLandmark

    def __len__(self):
        return len(self.edge_files)

    def __getitem__(self, idx):
        edge_filename = self.edge_files[idx]
                
        split_name = edge_filename.split("+")
        photo_folder, joint_name, filename = split_name[0], split_name[1], '+'.join(split_name[1:])

        heatmap_filename = filename.replace("+edge.png", "+heatmap.png")
        
        try:
            joint_id = JOINTS.index(joint_name)
        except ValueError:
            joint_id = -1
            print("No Joint Name found for: ", edge_filename)
        
        edge_path = os.path.join(self.edge_dir, photo_folder, filename)
        hm_path = os.path.join(self.edge_dir, photo_folder, heatmap_filename)
        
        edge_img = Image.open(edge_path).convert("L")
        hm_img = Image.open(hm_path).convert("L")
        
        edge_tensor = self.transform(edge_img)
        hm_tensor = self.transform(hm_img)
        
        return edge_tensor, hm_tensor, torch.tensor(joint_id).long()
    
class PlainDataset(Dataset):
    def __init__(self, plain_dir):

        self.plain_dir = plain_dir
        self.plain_files = [folder + "+" + f for folder in os.listdir(plain_dir) for f in os.listdir(os.path.join(plain_dir, folder)) if f.endswith("+plain.png")]
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        self.mp_pose = mp.solutions.pose.PoseLandmark

    def __len__(self):
        return len(self.plain_files)

    def __getitem__(self, idx):
        plain_filename = self.plain_files[idx]
                
        split_name = plain_filename.split("+")
        photo_folder, joint_name, filename = split_name[0], split_name[1], '+'.join(split_name[1:])
        
        try:
            joint_id = JOINTS.index(joint_name)
        except ValueError:
            joint_id = -1
            print("No Joint Name found for: ", plain_filename)
        
        plain_path = os.path.join(self.plain_dir, photo_folder, filename)
        
        plain_img = Image.open(plain_path).convert("L")
        
        plain_tensor = self.transform(plain_img)
        
        return plain_tensor, torch.tensor(joint_id).long()