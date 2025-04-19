import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import os
import pandas as pd
from PIL import Image

# ==== 설정 ====
latent_dim = 100
batch_size = 128
num_samples_per_class = 5000
# attr_to_extract = ['Eyeglasses', 'Smiling', 'Male']

# ==== 디바이스 설정 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 전처리 ====
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==== Custom CelebA Dataset ====
class CustomCelebA(Dataset):
    def __init__(self, img_dir, attr_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.attr_df = pd.read_csv(attr_path, delim_whitespace=True, skiprows=1)
        self.img_names = self.attr_df.index.tolist()
        self.attr_names = self.attr_df.columns.tolist()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        attr = torch.tensor((self.attr_df.loc[img_name].values + 1) // 2, dtype=torch.long)
        return image, attr

# ==== Dataset 준비 ====
data_root = '../datasets/celeba'
celeba_dataset = CustomCelebA(
    img_dir=os.path.join(data_root, 'img_align_celeba'),
    attr_path=os.path.join(data_root, 'list_attr_celeba.txt'),
    transform=transform
)

img = celeba_dataset[0][0]
celeba_loader = DataLoader(celeba_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# ==== attr_names 전체 추출 ====
attr_names = celeba_dataset.attr_names
attr_indices = {attr: attr_names.index(attr) for attr in attr_names}  # 40개 전부 사용
attr_to_extract = attr_names  # <- 모든 속성 추출로 변경

# ==== Encoder 로드 ====
class Encoder(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), # 64x64 → 32x32
            nn.BatchNorm2d(64), 
            nn.ReLU(True),
            
            nn.Conv2d(64, 128, 4, 2, 1), # 32 → 16
            nn.BatchNorm2d(128), 
            nn.ReLU(True),
            
            nn.Conv2d(128, 256, 4, 2, 1), # 16 → 8
            nn.BatchNorm2d(256), 
            nn.ReLU(True),
            
            nn.Conv2d(256, 512, 4, 2, 1), # 8 → 4
            nn.BatchNorm2d(512), 
            nn.ReLU(True),
            
            nn.Conv2d(512, z_dim, 4, 1, 0),  # Output: (B, z_dim, 1, 1) # 4 → 1
        )
    def forward(self, x):
        return self.main(x).squeeze()

encoder = Encoder(z_dim=latent_dim).to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder/encoder_epoch_200.pth"))
encoder.eval()

# ==== 방향 벡터 추출 ====
z_attrs = {}

for attr in attr_to_extract:
    pos_z = []
    neg_z = []
    count_pos = 0
    count_neg = 0
    print(f"[INFO] Extracting for attribute: {attr}")

    with torch.no_grad():
        for img, attr_tensor in tqdm(celeba_loader):
            img = img.to(device)
            z = encoder(img)
            attr_value = attr_tensor[:, attr_indices[attr]]

            for i in range(img.size(0)):
                if attr_value[i] == 1 and count_pos < num_samples_per_class:
                    pos_z.append(z[i].detach().cpu())
                    count_pos += 1
                elif attr_value[i] == 0 and count_neg < num_samples_per_class:
                    neg_z.append(z[i].detach().cpu())
                    count_neg += 1

            if count_pos >= num_samples_per_class and count_neg >= num_samples_per_class:
                break

    if len(pos_z) > 0 and len(neg_z) > 0:
        z_pos = torch.stack(pos_z)
        z_neg = torch.stack(neg_z)
        direction = z_pos.mean(0) - z_neg.mean(0)
        z_attrs[attr] = direction
        print(f"[INFO] v_{attr.lower()} extracted. Norm: {direction.norm():.4f}")
    else:
        print(f"[WARN] Not enough samples for attribute: {attr}")

# ==== 저장 ====
os.makedirs("./latent_directions", exist_ok=True)
for attr, vec in z_attrs.items():
    torch.save(vec, f"latent_directions/v_{attr.lower()}.pt")
    print(f"[SAVED] v_{attr.lower()} → latent_directions/v_{attr.lower()}.pt")
