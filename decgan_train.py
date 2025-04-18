import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils as vutils
import torch.optim as optim
import os
from tqdm import tqdm

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution (batch, 100, 1, 1)
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

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input: (batch, 3, 64, 64)
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (batch, 64, 32, 32)
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (batch, 128, 16, 16)
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (batch, 256, 8, 8)
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (batch, 512, 4, 4)
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size: (batch, 1, 1, 1)
        )

    def forward(self, input):
        # Output shape: (batch, 1, 1, 1) → reshape to (batch,)
        return self.main(input).view(-1, 1).squeeze(1)

# Weight Initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Show generated images
def show_generated_images(images, num_images=64):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    images = vutils.make_grid(images[:num_images], padding=2, normalize=True)
    images = np.transpose(images.cpu(), (1, 2, 0))  # CHW -> HWC
    # plt.imshow(images)
    # plt.show()
    name = './output/visualization/generated_images.jpg'
    plt.imsave(name, images)
    plt.close
    

def save_generated_images(images, num_images, epoch, idx):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    images = vutils.make_grid(images[:num_images], padding=2, normalize=True)
    images = np.transpose(images.cpu(), (1, 2, 0))  # CHW -> HWC
    # plt.imshow(images)  # 필요 시 시각화 가능
    fname = './output/visualization/image_' + str(epoch) + '_' + str(idx) + '.jpg'
    plt.imsave(fname, images.numpy())
    plt.close()
    
def visualization(lossD_list,lossG_list,D_x_list,D_G_z2_list,start_epoch,num_epochs):
        # 결과 시각화 저장
    os.makedirs("logs", exist_ok=True)
    epochs = list(range(start_epoch, num_epochs))

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(epochs, lossD_list, label="Loss_D")
    plt.plot(epochs, lossG_list, label="Loss_G")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()
    plt.savefig("logs/loss_curve.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.title("D(x) and D(G(z)) During Training")
    plt.plot(epochs, D_x_list, label="D(x)")
    plt.plot(epochs, D_G_z2_list, label="D(G(z))")
    plt.xlabel("Epoch")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid()
    plt.savefig("logs/prob_curve.png")
    plt.close()


def main():
    lossD_list = []
    lossG_list = []
    D_x_list = []
    D_G_z2_list = []
    D_G_z1_list = []
    # 생성자 및 판별자 정의 (Generator, Discriminator 클래스가 미리 정의되어 있어야 함)
    netG = Generator()
    netD = Discriminator()
    # resume 관련 설정
    resume = True
    resume_epoch = 150
    if resume:
        netG.load_state_dict(torch.load(f"checkpoints/netG/netG_epoch_{resume_epoch}.pth"))
        netD.load_state_dict(torch.load(f"checkpoints/netD/netD_epoch_{resume_epoch}.pth"))
        start_epoch = resume_epoch + 1
        print(f" Resuming training from epoch {start_epoch}")
    else:
        start_epoch = 0
        # 가중치 초기화
        netG.apply(weights_init)
        netD.apply(weights_init)

    # 데이터 전처리 설정
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # 데이터셋 및 데이터로더
    dataset = datasets.ImageFolder(root='../datasets/ffhq', transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 학습 환경 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    netG.to(device)
    netD.to(device)

    # 손실 함수 및 최적화 함수 설정
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 학습 관련 설정
    num_epochs = 350
    fixed_noise = torch.randn(64, 100, 1, 1, device=device)  # 고정된 z 입력으로 생성 이미지 변화 확인
    
    for epoch in range(start_epoch,num_epochs):
        loop = tqdm(enumerate(dataloader, 0), total=len(dataloader), leave=False)
        for i, data in loop:#enumerate(dataloader, 0):
            ############################
            # (1) Discriminator 학습
            ############################
            netD.zero_grad()
            
            # 실제 이미지 로딩
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            
            real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
            fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

            # 실제 이미지 판별  -> netD만 학습시키기 
            output = netD(real_data).view(-1)
            errD_real = criterion(output, real_label)
            errD_real.backward() # 당dusgl netD만 grad계산함. output에 들어있는grad -> 이후 netD에 대한 연산 그래프는 다 사라짐
            # errD_real.backward(retain_graph=True)  # 그래프 유지

            D_x = output.mean().item()  #-=-=-=-=-=-=-=- Discriminator가 real 이미지를 "진짜"라고 판단한 평균 확률값

            # 가짜 이미지 생성
            noise = torch.randn(batch_size, 100, 1, 1, device=device)
            fake_data = netG(noise) # netG 연산그래프 생성
            output = netD(fake_data.detach()).view(-1)  # fake_data가 backward 과정에서 연산그래프가 사라지면 안되니깐(나중에 netG 파라미터 업데이트 할 때 재활용해야하기 때문에) 연산그래프는 냅두고 fake_data의 값만 detach해서 전달 
            errD_fake = criterion(output, fake_label)
            errD_fake.backward() # netD만 grad계산함, netGX -> detach()했기 때문.여기서 ake_data의 값만 detach해서 전달했기 때문에 netG의 fake_data에대한 연산그래프는 남아있고 netD의 연산그래프는 사라짐
            # 여기서 기존에 errD_real.backward()를 통해 grad가 구해졌을 텐데 바로 errD_fake.backward() 를 또 했기 때문에 grad가 누적되어 더해짐 -> loss두개 더한거 가지고 grad한번 계산한거랑 같다고 보면 됨.
            D_G_z1 = output.mean().item() #-=-=-=-=-=-=-=-=-= Generator가 만든 fake 이미지를 Discriminator가 "진짜"라고 판단한 확률 (Discriminator 학습 시 기준)

            # 판별자 전체 loss 및 가중치 업데이트
            errD = errD_real + errD_fake # 로그용
            optimizerD.step() # 더해진 기울기를 가지고 파라미터 업데이트 -> 기울기를 두번 계산(backward두번)해서 더해서 한번 .step하는 것과 loss들을 먼저 더해서 한번에 backward 한번하는 거랑 같음. 대신에 연속으로 backward두번 할때는 연산그래프를 보존해야함.
            
            ##########################################
            # (2) Generator 학습: maximize log(D(G(z)))
            ##########################################
            netG.zero_grad()

            # 이전 단계에서 만든 fake_data를 그대로 사용
            output = netD(fake_data).view(-1) # 기존에 fake_data재활용 -> detach 안했으므로 연산그래프(backward과정에서 역추적 할 수 있도록 도와주는 친구)가 있음
            errG = criterion(output, real_label)  # Generator는 "real"처럼 속이길 원함
            errG.backward() 
            D_G_z2 = output.mean().item() #-=-=-=-=-=-=-=-
            optimizerG.step()

            ##########################################
            # 중간 학습 상태 출력 및 이미지 저장
            ##########################################
            if i % 50 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # tqdm.write('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                #     % (epoch, num_epochs, i, len(dataloader),
                #         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                            
                # fixed noise로 생성된 이미지 시각화
                fake_images = netG(fixed_noise)
                save_generated_images(fake_images, 64, epoch=epoch, idx=i)
            
            # tqdm 출력 내용
            loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
            loop.set_postfix(loss_D=errD.item(), loss_G=errG.item())
        
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
        # 에폭 단위 기록
        lossD_list.append(errD.item())
        lossG_list.append(errG.item())
        D_x_list.append(D_x)
        D_G_z2_list.append(D_G_z2)  # Generator 입장에서 D(G(z)) 추적
        D_G_z1_list.append(D_G_z1)  # Discriminator 입장에서 D(G(z)) 추적     
        ####################################
        # 에폭 단위 가중치 저장
        ####################################
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            torch.save(netG.state_dict(), f"checkpoints/netG/netG_epoch_{epoch}.pth")
            torch.save(netD.state_dict(), f"checkpoints/netD/netD_epoch_{epoch}.pth")
            print(f"Saved model weights at epoch {epoch}")
    
    visualization(lossD_list, lossG_list, D_x_list, D_G_z2_list, start_epoch, start_epoch + len(lossD_list))
            
        

    # #visualization
    # fake_images = netG(fixed_noise)
    # show_generated_images(fake_images)

if __name__ == "__main__":
    main()