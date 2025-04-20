import torch
import torch.nn as nn
from torchvision import transforms, utils as vutils
from PIL import Image
import os

# ==== 설정 ====
latent_dim = 100
attribute = 'smiling' #'wearing_lipstick'  # wearing_lipstick,blond_hair ,eyeglasses, smiling, male
alpha = 2  # 조작 강도
img_path = 'images/jpg_resized/IMG_3824.png'  # <- 직접 찍은 이미지 경로

# ==== 디바이스 설정 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== 이미지 전처리 ====
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ==== 역정규화 함수 ====
def denormalize(tensor):
    return tensor * 0.5 + 0.5

# ==== 모델 정의 ====
class Encoder(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Conv2d(512, z_dim, 4, 1, 0),
        )
    def forward(self, x):
        return self.main(x).squeeze()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)

# ==== 모델 로드 ====
encoder = Encoder().to(device)
encoder.load_state_dict(torch.load("checkpoints/encoder/encoder_epoch_200.pth"))
encoder.eval()

generator = Generator().to(device)
generator.load_state_dict(torch.load("checkpoints/netG/netG_epoch_180_check.pth"))
generator.eval()

# ==== 방향 벡터 불러오기 ====
v_attr = torch.load(f"latent_directions/v_{attribute}.pt").to(device)

# ==== 이미지 로드 ====
img = Image.open(img_path).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)  # (1, 3, 64, 64)

with torch.no_grad():
    z = encoder(img_tensor)  # (1, 100)
    z = z.view(1, 100, 1, 1)

    z_edit = z + alpha * v_attr.view(1, -1, 1, 1)

    recon = generator(z)
    edited = generator(z_edit)

# ==== 시각화 ====
original = denormalize(img_tensor.cpu())
recon = denormalize(recon.cpu())
edited = denormalize(edited.cpu())

# 3개 이미지 하나로 이어 붙이기
combined = torch.cat([original, recon, edited], dim=0)

# 저장
os.makedirs("output/demo", exist_ok=True)
vutils.save_image(
    combined,
    f"output/demo/your_face_recon_vs_{attribute}.jpg",
    nrow=3,
    padding=5,
    #normalize=True
)


print(f"[SAVED] 시연 이미지 저장 완료: output/demo/your_face_recon_vs_{attribute}.jpg")
