import os
import numpy as np
from PIL import Image

data_dir = '/Users/navneetbavineni/src/pytorch_data_dir'
chaos_train = os.path.join(data_dir, 'archive', 'CHAOS_Train_Sets', 'Train_Sets')

empty_count = 0
total_count = 0

if not os.path.exists(chaos_train):
    print("CHAOS Train directory not found at", chaos_train)
else:
    for root, dirs, files in os.walk(chaos_train):
        if 'Ground' in root:
            for file in files:
                if file.lower().endswith('.png'):
                    total_count += 1
                    filepath = os.path.join(root, file)
                    img = np.array(Image.open(filepath).convert('L'))
                    if np.all(img == 0):
                        empty_count += 1

print(f"Total labels: {total_count}")
print(f"Empty labels (no organ): {empty_count}")
