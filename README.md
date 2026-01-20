# Landscape Colorization / Sieci neuronowe

## Informacje o projekcie
* [cite_start]Autorzy: Michał Karelus, Damian Wisłocki[cite: 2].
* [cite_start]Uczelnia: WSZIB, 2026[cite: 3].
* [cite_start]Problem: Koloryzacja obrazu jako problem wielwymiarowy łączący regresję kanałów RGB oraz segmentację kontekstu obiektów[cite: 8].
* [cite_start]Architektura: Konwolucyjny Autoenkoder inspirowany siecią U-Net[cite: 51].

* # Instrukcja uruchomienia (Google Colab)

[cite_start]Projekt został zoptymalizowany pod kątem pracy w chmurze z wykorzystaniem akceleracji GPU[cite: 29].

# 1. Otwórz Google Colab i zmień typ środowiska na GPU (Runtime -> Change runtime type -> T4 GPU).
# 2. Wykonaj poniższe kroki w komórkach notatnika:

```python
# 1. Sklonuj repozytorium
!git clone https://github.com/TWOJA_NAZWA/landscape-colorization.git
%cd landscape-colorization

# 2. Zainstaluj niezbędne biblioteki
!pip install -r requirements.txt -q

# 3. Pobierz i przygotuj zbiór danych (wymaga klucza Kaggle w pliku)
!python setup_data.py

# 4. Rozpocznij proces trenowania modelu (5 epok)
!python train.py

# 5. Wyświetl wyniki wizualne i oblicz metrykę SSIM
!python results.py
