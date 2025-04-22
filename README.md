# FFHQ Latent Direction Editing with DCGAN and CelebA Attributes

본 프로젝트는 가장 기본적인 DCGAN과 Encoder를 활용하여 FFHQ 얼굴 이미지의 latent space를 조작함으로써 얼굴의 속성 (예: 웃는 얼굴, 안경 착용 등)을 제어할 수 있는 GAN 기반 생성 시스템을 구현한 것입니다. CelebA의 속성 라벨을 기반으로 방향 벡터를 추출하여 FFHQ 얼굴 이미지에 적용합니다.

---

## 🔧 Environment Setup

### 1. Conda 가상환경 설정
아래 명령어를 통해 `environment.yaml` 파일로부터 가상환경을 복원할 수 있습니다

```bash
conda env create -f environment.yaml
conda activate dcgan
```
아래와 같은 에러가 발생하면 
```swift
CondaValueError: prefix already exists or is not writable
```
environment.yaml 파일을 텍스트 편집기로 열고 
```prefix: /home/ ~ /``` 
이 부분을 삭제해주세요.

---

## 📁 Dataset Structure

### 1. FFHQ Dataset
```
../datasets/ffhq/
    ffhq256/               # 256x256 해상도의 이미지 (Encoder 학습용)
    img_align_ffhq/        # 64x64 해상도 전처리 이미지 (DCGAN 학습용)
```

### 2. CelebA Dataset
```
../datasets/celeba/
    img_align_celeba/        # CelebA 이미지
    list_attr_celeba.txt     # 40가지 속성 라벨 정보
```

---

## 🧪 Training

### 1. DCGAN 학습
- 학습 스크립트: `dcgan_train.py`
- 실행 시 생성자(Generator)와 판별자(Discriminator)가 학습되며 다음 디렉토리들이 생성됩니다

```
checkpoints/netG/     # Generator 가중치 저장 경로
checkpoints/netD/     # Discriminator 가중치 저장 경로
output/reconGAN64/    # 정성적 시각화: 고정된 z로 생성한 이미지 저장
```

```bash
python dcgan_train.py
```

---

### 2. Encoder 학습
- 학습 스크립트: `encoder_train.py`
- 고정된 Generator를 이용해 Encoder가 학습됩니다 (MSE loss 기반)
- 다음 디렉토리들이 생성됩니다

```
checkpoints/encoder/         # Encoder 가중치 저장 경로
output/recon_encoder/real/   # 원본 이미지 저장
output/recon_encoder/recon/  # 재구성 이미지 저장
output/recon_encoder/compare/# 원본 vs 재구성 비교
logs/encoder/                # 손실 그래프 저장 (loss_curve.png)
```

```bash
python encoder_train.py
```

---

## 📐 Latent Direction Extraction

### 3. 속성 벡터 추출 (direction vector)
- 스크립트: `extraction_direction.py`
- 사전 학습된 Encoder를 사용하여 CelebA의 각 속성에 대해 latent space 내의 방향 벡터를 추출합니다.
- 각 속성은 양의 샘플(예: 안경 있음)과 음의 샘플(안경 없음)의 평균 벡터 차이로 계산됩니다.
- 결과는 `.pt` 파일로 저장됩니다

```
latent_directions/
    v_smile.pt
    v_eyeglasses.pt
    v_male.pt
    ... (총 40개)
```

```bash
python extraction_direction.py
```

---

## 🧪 DEMO 실행 방법

### 4. iPhone 사진 활용한 데모 실행

#### (1) 이미지 준비 및 변환

iPhone으로 찍은 사진을 아래 위치에 저장
```
DCGAN_encoder/heic_images/your_image.HEIC
```

```bash
python HEIC2JPG.py  # heic -> jpg로 변환
```

#### (2) 속성 조작 실행
```bash
python demo.py
```

- **변경해야 할 변수들:**
  - `image_path`: 변환된 이미지 경로
  - `attr_name`: 사용할 속성 방향 벡터 이름 (예: `"smile"`, `"eyeglasses"`)
  - `alpha`: 적용 강도

- 실행 결과는 아래 디렉토리에 저장됩니다
```
output/demo/
    input.jpg
    edited.jpg
```

---

### 5. FFHQ 기반 속성 조작 빠른 테스트
```bash
python recon_want.py
```

- 기본 이미지 하나를 z로 인코딩한 후 속성을 조작하고 재생성 결과를 보여줍니다.
- 속성 이름은 스크립트 내에서 직접 수정
```python
attr_name = "smile"  # "eyeglasses", "male" 등으로 변경 가능
```

- 결과는 아래에 저장:
```
output/edited/
    original.png
    edited_smile.png
```

---

## 📌 Directory Summary
```
checkpoints/
    netG/       # DCGAN Generator
    netD/       # DCGAN Discriminator
    encoder/    # Encoder 가중치

latent_directions/
    v_smile.pt, v_eyeglasses.pt, ...  # 총 40개 속성 방향벡터

output/
    reconGAN64/        # DCGAN 이미지 생성 과정
    recon_encoder/     # Encoder 재구성 결과
    demo/              # 사용자가 넣은 이미지 조작 결과
    edited/            # FFHQ 이미지에 속성 적용 결과

logs/
    encoder/encoder_loss_plot.png
    dcgan/dcgan_loss_plot.png # 해당 LOSS 그래프는 날라감.
```

---

## 📝 Acknowledgements
- FFHQ dataset: [NVidia Research](https://github.com/NVlabs/ffhq-dataset)
- CelebA dataset: [Liu et al., 2015](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- DCGAN reference: [Radford et al., 2016]



