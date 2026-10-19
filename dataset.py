import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class NuScenesDataset(Dataset):
    def __init__(self, nusc, transform=None):
        self.nusc = nusc
        self.image_paths = []
        self.transform = transform

        for scene in self.nusc.scene:
           current_data_token = scene['first_sample_token']

           while current_data_token != '':
               sample = self.nusc.get('sample', current_data_token)

               cam_front_token = sample['data']['CAM_FRONT']

               data_path,_,_ = self.nusc.get_sample_data(cam_front_token)

               self.image_paths.append(data_path)

               current_data_token = sample['next']

    def __len__(self):
        return len(self.image_paths)


    def __getitem__(self,idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from torch.utils.data import DataLoader

    my_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)
    dataset = NuScenesDataset(nusc, transform=my_transform)

    dataloader = DataLoader(dataset,batch_size=4, shuffle=True)

    first_batch = next(iter(dataloader))

    print(first_batch.shape)
