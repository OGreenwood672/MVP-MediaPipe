from torchvision import transforms
import os
from PIL import Image
import mediapipe as mp
import torch
from torch.utils.data import Dataset

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
        photo_folder, joint_id, filename = split_name[0], int(split_name[2]), '+'.join(split_name[1:])

        heatmap_filename = filename.replace("+rgb.png", "+heatmap.png")
                
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
        photo_folder, joint_id, filename = split_name[0], int(split_name[2]), '+'.join(split_name[1:])

        heatmap_filename = filename.replace("+edge.png", "+heatmap.png")

        
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
        photo_folder, joint_id, filename = split_name[0], int(split_name[2]), '+'.join(split_name[1:])
        
        plain_path = os.path.join(self.plain_dir, photo_folder, filename)
        
        plain_img = Image.open(plain_path).convert("L")
        
        plain_tensor = self.transform(plain_img)
        
        return plain_tensor, torch.tensor(joint_id).long()
    
class HintDataset(Dataset):
    def __init__(self, rgb_dir, edge_dir):

        self.rgb_dir = rgb_dir
        self.edge_dir = edge_dir
        self.files = []
        for folder in sorted(os.listdir(rgb_dir)):
            for f in sorted(os.listdir(os.path.join(rgb_dir, folder))):
                if f.endswith("+rgb.png"):
                    self.files.append((folder + "+" + f, folder + "+" + f.replace("+rgb.png", "+edge.png")))
        
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        self.mp_pose = mp.solutions.pose.PoseLandmark

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        rgb_file, edge_file = self.files[idx]
                
        split_name = rgb_file.split("+")
        photo_folder, joint_id, rgb_filename = split_name[0], int(split_name[2]), '+'.join(split_name[1:])
        edge_filename = '+'.join(edge_file.split("+")[1:])

        # Same heatmap for both
        heatmap_filename = rgb_filename.replace("+rgb.png", "+heatmap.png")
        
        rgb_path = os.path.join(self.rgb_dir, photo_folder, rgb_filename)
        edge_path = os.path.join(self.edge_dir, photo_folder, edge_filename)
        
        hm_path = os.path.join(self.rgb_dir, photo_folder, heatmap_filename)
        
        rgb_img = Image.open(rgb_path).convert("RGB")
        edge_img = Image.open(edge_path).convert("L")
        hm_img = Image.open(hm_path).convert("L")
        
        rgb_tensor = self.transform(rgb_img)
        edge_tensor = self.transform(edge_img)
        hm_tensor = self.transform(hm_img)

        combined_tensor = torch.cat((rgb_tensor, edge_tensor), dim=0)
        
        return combined_tensor, hm_tensor, torch.tensor(joint_id).long()
