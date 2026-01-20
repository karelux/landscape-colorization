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

# 3. Pobierz i przygotuj zbiór danych (wymaga klucza Kaggle w pliku)
!python setup_data.py

# 4. Rozpocznij proces trenowania modelu (5 epok)
!python train.py

# 5. Wyświetl wyniki wizualne i oblicz metrykę SSIM
!python results.py
