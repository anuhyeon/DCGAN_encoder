import torch
import torch.nn as nn
from torchvision import transforms, utils as vutils
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ==== 설정 ====
latent_dim = 100
attribute =  'eyeglasses'  #'wearing_lipstick' # blond_hair'  # eyeglasses, smiling, male
alpha = 2.0  # 조작 강도

# ==== 디바이스 설정 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 데이터 전처리 ====
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==== FFHQ 이미지 샘플 로드 (또는 CelebA 테스트셋 등) ====
data = ImageFolder(root="../datasets/ffhq", transform=transform)
dataloader = DataLoader(data, batch_size=16, shuffle=False)

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


# ==== Generator 로드 ====
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution (batch, 100, 1, 1) 
            # output_size = (input_size - 1) * stride - 2 * padding + kernel_size + output_padding
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # State size: (batch, 512, 4, 4)
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # State size: (batch, 256, 8, 8)
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # State size: (batch, 128, 16, 16)
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # State size: (batch, 64, 32, 32)
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: (batch, 3, 64, 64)
        )

    def forward(self, input):
        return self.main(input)

encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder/encoder_epoch_200.pth"))
encoder.eval()

generator = Generator().to(device)
generator.load_state_dict(torch.load("checkpoints/netG/netG_epoch_180_check.pth"))  # <- FFHQ로 학습된 G
generator.eval()

# ==== 조작할 latent 방향 불러오기 ====
v_attr = torch.load(f"latent_directions/v_{attribute}.pt").to(device)

# ==== 이미지 1 batch만 조작 ====
img_batch, _ = next(iter(dataloader))
img_batch = img_batch.to(device)

with torch.no_grad():
    z = encoder(img_batch)  # (B, 100)
    z = z.view(z.size(0), 100, 1, 1)
    z_edit = z + alpha * v_attr.view(1, -1, 1, 1)

    recon = generator(z)
    edited = generator(z_edit)

# ==== 시각화 ====
# recon: Generator(Encoder(img))
# edited: Generator(Encoder(img) + alpha * v_attr)
# original: raw input image (denormalized)

combined = torch.cat([img_batch, recon, edited], dim=0)
os.makedirs("output/edited", exist_ok=True)
vutils.save_image(combined, f"output/edited/recon_plus_{attribute}.jpg", nrow=img_batch.size(0), normalize=True)
# vutils.save_image(img_batch.cpu(), f"output/edited/origin{attribute}.jpg", nrow=img_batch.size(0), normalize=True)

print(f"[SAVED] 조작된 이미지 저장 완료: output/edited/recon_plus_{attribute}.jpg")