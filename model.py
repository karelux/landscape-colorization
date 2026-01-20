import torch
import torch.nn as nn

class ColorizationNet(nn.Module):
    def __init__(self):
        super(ColorizationNet, self).__init__()
        # Encoder: Kompresja obrazu i ekstrakcja cech
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # Decoder: Dekompresja i rekonstrukcja kanałów kolorów
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.dec1(self.enc1(x))

if __name__ == "__main__":
    model = ColorizationNet()
    print("Model U-Net został zdefiniowany pomyślnie.")
