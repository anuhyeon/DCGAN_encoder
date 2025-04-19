import os, glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import datasets, utils as vutils
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from decgan_train import Generator  # G 구조 그대로 사용
import random
import numpy as np

# ==== 랜덤시드 고정 ====
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ==== 하이퍼파라미터 ====
resume_epoch = 30
latent_dim = 100
image_size = 64
batch_size = 128
num_epochs = 200
lr = 0.0002
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== FFHQ 데이터셋 클래스 ====
class FFHQDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.paths = glob.glob(os.path.join(root_dir, '*'))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ==== Encoder 구조 ====
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
        return self.main(x)

# ==== 학습 및 시각화 ====
def plot_losses(losses, path='logs/encoder/encoder_loss_plot.png'):
    plt.plot(losses, label='Reconstruction Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Encoder Training Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(path)
    plt.close()

def train_encoder():
    # 데이터, 모델, 손실, 옵티마이저
    dataset = FFHQDataset('../datasets/ffhq/ffhq256', transform)
    # dataset = datasets.ImageFolder(root='../datasets/ffhq', transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    netG = Generator().to(device)
    netG.load_state_dict(torch.load('./checkpoints/netG/netG_epoch_180_check.pth'))
    netG.eval()
    for param in netG.parameters():
        param.requires_grad = False
        
    netE = Encoder(z_dim=latent_dim).to(device)
    
    # resume 설정
    if resume_epoch > 0:
        netE.load_state_dict(torch.load(f"checkpoints/encoder/encoder_epoch_{resume_epoch:03d}.pth"))
        print(f"[INFO] Resuming from epoch {resume_epoch}")
    
    optimizer = torch.optim.Adam(netE.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    loss_history = []

    for epoch in range(num_epochs):
        netE.train()
        running_loss = 0.0
        for real_imgs in tqdm(dataloader, desc=f"[Epoch {epoch+1}/{num_epochs}]"):
            real_imgs = real_imgs.to(device)
            z = netE(real_imgs)
            recon_imgs = netG(z)

            loss = criterion(recon_imgs, real_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * real_imgs.size(0)

        avg_loss = running_loss / len(dataloader.dataset)
        loss_history.append(avg_loss)
        
        print(f"[{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

        # 이미지 및 모델 저장
        if (epoch+1) % 10 == 0:
            comparison = torch.cat([real_imgs[:64], recon_imgs[:64]], dim=3)
            save_image(comparison, f"output/recon_encoder/compare/compare_epoch_{epoch+1:03d}.png", nrow=8, normalize=True)
            save_image(recon_imgs.data[:64], f"output/recon_encoder/recon/recon_epoch_{epoch+1:03d}.png", normalize=True)
            torch.save(netE.state_dict(), f"checkpoints/encoder/encoder_epoch_{epoch+1:03d}.pth")

    plot_losses(loss_history)
    print("Encoder 학습 완료, 모델과 loss plot 저장됨.")

if __name__ == '__main__':
    train_encoder()
