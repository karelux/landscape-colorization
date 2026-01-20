import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

base_path = 'landscape_data/Landscape Classification/Landscape Classification'

def get_loaders(batch_size=16):
    train_dir = os.path.join(base_path, 'Training Data')
    val_dir = os.path.join(base_path, 'Validation Data')
    test_dir = os.path.join(base_path, 'Testing Data')

    # Transformacje: zmiana rozmiaru i konwersja na tensor
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    train_set = datasets.ImageFolder(root=train_dir, transform=transform)
    val_set = datasets.ImageFolder(root=val_dir, transform=transform)
    test_set = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader, train_set

if __name__ == "__main__":
    _, _, _, train_set = get_loaders()
    print(f"Sukces! Dane gotowe. Liczba zdjęć w treningu: {len(train_set)}")
