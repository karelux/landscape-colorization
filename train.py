import torch
import torch.optim as optim
import torch.nn as nn
from model import ColorizationNet
from data_loader import get_loaders

# Konfiguracja urządzenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicjalizacja modelu, optymalizatora i funkcji straty
model = ColorizationNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Pobranie danych
train_loader, _, _, _ = get_loaders()

print(f"Rozpoczynam trenowanie na: {device}... (Epoka 1/5)")

for epoch in range(5):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)
        # Konwersja do skali szarości (wejście modelu)
        gray = torch.mean(images, dim=1, keepdim=True).to(device)

        optimizer.zero_grad()
        outputs = model(gray)
        loss = criterion(outputs, images)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoka {epoch+1} zakończona. Średni błąd MSE: {running_loss/len(train_loader):.4f}")

# Zapisanie modelu po treningu
torch.save(model.state_dict(), "colorization_model.pth")
