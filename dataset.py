import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple, List


class NuScenesDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset for loading camera frames from a specific NuScenes scene.
    """

    def __init__(self, nusc, scene_index: int = 0):
        self.nusc = nusc
        self.scene_index = scene_index
        self.scene = nusc.scene[scene_index]

        self.sample_tokens: List[str] = []
        self.cam_tokens: List[str] = []

        # Traverse all sequential samples within the selected scene
        curr_sample_token = self.scene['first_sample_token']
        while curr_sample_token:
            self.sample_tokens.append(curr_sample_token)
            sample = nusc.get('sample', curr_sample_token)

            # Extract front camera token for this sample frame
            cam_token = sample['data']['CAM_FRONT']
            self.cam_tokens.append(cam_token)

            # Move to the next chronological frame in the scene
            curr_sample_token = sample['next']

    def __len__(self) -> int:
        return len(self.cam_tokens)

    def __getitem__(self, idx: int) -> Image.Image:
        cam_token = self.cam_tokens[idx]
        image_path, _, _ = self.nusc.get_sample_data(cam_token)
        image = Image.open(image_path).convert("RGB")
        return image


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version='v1.0-mini', dataroot='data/nuscenes', verbose=False)
    test_dataset = NuScenesDataset(nusc, scene_index=1)
    print("--- Dataset Test ---")
    print(f"Target Scene: {nusc.scene[1]['name']}")
    print(f"Total Frames Loaded: {len(test_dataset)}")