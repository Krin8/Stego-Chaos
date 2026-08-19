import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

img_dir = '/Users/navneetbavineni/chaos_dataset_ready/images'
out_dir = '/Users/navneetbavineni/.gemini/antigravity-ide/brain/9ed0992d-55da-44a7-a63b-86a31f25e1c0'

images = ['0000.png', '0001.png', '0002.png', '0003.png']

for i, img_name in enumerate(images):
    path = os.path.join(img_dir, img_name)
    if not os.path.exists(path):
        continue
    
    # Load original
    orig_img = Image.open(path)
    
    # Convert to RGB (how it's done in the dataset)
    rgb_img = orig_img.convert('RGB')
    
    # Convert to numpy to show the channels
    orig_arr = np.array(orig_img)
    rgb_arr = np.array(rgb_img)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    if len(orig_arr.shape) == 2:
        axes[0].imshow(orig_arr, cmap='gray')
        axes[0].set_title(f"Original Grayscale\nShape: {orig_arr.shape} (1 Channel)")
    else:
        axes[0].imshow(orig_arr)
        axes[0].set_title(f"Original shape: {orig_arr.shape}")
        
    # Show the combined RGB image
    axes[1].imshow(rgb_arr)
    axes[1].set_title(f"Final Combined RGB\nShape: {rgb_arr.shape} (3 Channels)")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'combined_rgb_{img_name}'))
    plt.close()

print("Combined images saved.")
