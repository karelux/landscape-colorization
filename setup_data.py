import os

os.environ['KAGGLE_USERNAME'] = " " #Username z Kaggle
os.environ['KAGGLE_KEY'] = " " #API Token z Kaggle

# Pobieranie danych
os.system("kaggle datasets download -d utkarshsaxenadn/landscape-recognition-image-dataset-12k-images")
os.system("unzip -q landscape-recognition-image-dataset-12k-images.zip -d landscape_data")
print("Sukces! Dane są gotowe.")
