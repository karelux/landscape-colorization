# Landscape Colorization / Sieci neuronowe

## Informacje o projekcie
* Autorzy: Michał Karelus, Damian Wisłocki.
* Uczelnia: WSZIB, 2026.
* Problem: Koloryzacja obrazu jako problem wielwymiarowy łączący regresję kanałów RGB oraz segmentację kontekstu obiektów.
* Architektura: Konwolucyjny Autoenkoder inspirowany siecią U-Net.

## Instrukcja uruchomienia (Google Colab)

Projekt został zoptymalizowany pod kątem pracy w chmurze z wykorzystaniem akceleracji GPU.

### 1. Konfiguracja środowiska
Otwórz Google Colab i zmień typ środowiska na GPU: `Środowisko wykonawcze` -> `Zmień typ środowiska wykonawczego` -> `GPU T4`.

### 2. Wykonaj poniższe kroki w komórkach notatnika:

```python
# 1. Sklonuj repozytorium
!git clone https://github.com/karelux/landscape-colorization.git
%cd landscape-colorization

# 2. Zainstaluj niezbędne biblioteki
!pip install -r requirements.txt -q

# 3. Pobierz i przygotuj zbiór danych (należy uzupełnić plik swoim username oraz API Tokenem z kaggle)
!python setup_data.py

# 4. Rozpocznij proces trenowania modelu (5 epok)
!python train.py

# 5. Wyświetl wyniki wizualne i oblicz metrykę SSIM
import torch
import matplotlib.pyplot as plt
import random
import numpy as np
from skimage.metrics import structural_similarity as ssim
from model import ColorizationNet
from data_loader import get_loaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ColorizationNet().to(device)
model.load_state_dict(torch.load("colorization_model.pth", map_location=device))
model.eval()

_, _, test_loader, _ = get_loaders()

# PORÓWNANIE WIZUALNE
all_images, all_outputs, all_grays = [], [], []

with torch.no_grad():
    iterator = iter(test_loader)
    images, _ = next(iterator)
    images = images.to(device)
    gray = torch.mean(images, dim=1, keepdim=True)
    output = model(gray)

    for _ in range(5):
        idx = random.randint(0, len(images) - 1)
        all_images.append(images[idx].cpu())
        all_grays.append(gray[idx][0].cpu())
        all_outputs.append(output[idx].cpu())

for i in range(5):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(all_grays[i], cmap='gray'); plt.title("Wejście (B&W)"); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(all_outputs[i].permute(1, 2, 0)); plt.title("Model AI"); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(all_images[i].permute(1, 2, 0)); plt.title("Oryginał"); plt.axis('off')
    plt.show()

# SSIM
print("Obliczanie średniego SSIM...")
all_ssim = []
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        gray = torch.mean(images, dim=1, keepdim=True).to(device)
        outputs = model(gray)
        orig = images.cpu().permute(0, 2, 3, 1).numpy()
        pred = outputs.cpu().permute(0, 2, 3, 1).numpy()
        for i in range(len(orig)):
            score = ssim(orig[i], pred[i], data_range=1.0, channel_axis=-1)
            all_ssim.append(score)

print(f"Średni wynik SSIM na zbiorze testowym: {np.mean(all_ssim):.4f}")

