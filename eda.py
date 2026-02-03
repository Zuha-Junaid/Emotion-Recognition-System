import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random

# 1. Configuration
TRAIN_DIR = "train"

def perform_eda():
    print("Starting Exploratory Data Analysis...")
    
    # Automatically get the list of emotion folders present in 'train'
    # This avoids the "FileNotFoundError" if names are capitalized
    emotions = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]
    print(f"Detected emotions: {emotions}")
    
    train_counts = []
    for emotion in emotions:
        path = os.path.join(TRAIN_DIR, emotion)
        count = len(os.listdir(path))
        train_counts.append({'Emotion': emotion, 'Count': count})
    
    df = pd.DataFrame(train_counts)
    
    # --- VISUALIZATION 1: BAR CHART ---
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Emotion', y='Count', data=df, palette='magma')
    plt.title('Dataset Distribution')
    plt.savefig('distribution.png')
    print("✅ Saved distribution.png")
    
    # --- VISUALIZATION 2: SAMPLES ---
    plt.figure(figsize=(15, 5))
    for i, emotion in enumerate(emotions):
        folder_path = os.path.join(TRAIN_DIR, emotion)
        random_img = random.choice(os.listdir(folder_path))
        img = Image.open(os.path.join(folder_path, random_img))
        plt.subplot(1, len(emotions), i+1)
        plt.imshow(img, cmap='gray')
        plt.title(emotion)
        plt.axis('off')
    plt.savefig('samples.png')
    print("✅ Saved samples.png")
    plt.show()

if __name__ == "__main__":
    if os.path.exists(TRAIN_DIR):
        perform_eda()
    else:
        print(f"Error: '{TRAIN_DIR}' folder not found in {os.getcwd()}")