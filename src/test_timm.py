"""
Visualize Self and KNN Correspondence for Medical Image Segmentation

This script demonstrates how self-correspondence and KNN correspondence work
in the context of unsupervised medical image segmentation using the CHAOS dataset.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms as T

from data import CHAOS

def load_example_images():
    """Load example CT images and labels directly from raw CHAOS DICOM dataset"""
    root = "/Users/navneetbavineni/src/pytorch_data_dir"
    dataset = CHAOS(
        root=root,
        modality="CT",
        image_set="train",
        transform=T.ToTensor(),
        target_transform=lambda x: torch.from_numpy(np.array(x)).unsqueeze(0),
        n_classes=5,
        use_preprocessed_data=False,
    )
    
    images = []
    labels = []
    
    # Pick sample slices that contain liver/organ annotations.
    # We start from the middle of the dataset where organs (like the liver) are 
    # fully visible and large, making our unsupervised heuristic more robust.
    start_idx = len(dataset) // 2
    for i in range(start_idx, len(dataset)):
        img, label, mask = dataset[i]
        lbl_arr = np.array(label.cpu()) if hasattr(label, 'cpu') else np.array(label)
        img_arr = img.cpu().numpy().transpose(1, 2, 0) if hasattr(img, 'cpu') else np.array(img)
        if lbl_arr.max() > 0:
            images.append(img_arr)
            labels.append(lbl_arr)
        if len(images) >= 5:
            break
            
    if len(images) < 2:
        for i in range(min(5, len(dataset))):
            img, label, mask = dataset[i]
            lbl_arr = np.array(label.cpu()) if hasattr(label, 'cpu') else np.array(label)
            img_arr = img.cpu().numpy().transpose(1, 2, 0) if hasattr(img, 'cpu') else np.array(img)
            images.append(img_arr)
            labels.append(lbl_arr)
            
    return images, labels

def compute_feature_similarity(patch1, patch2):
    """Compute cosine similarity between two image patches"""
    # Flatten and normalize
    p1_flat = patch1.reshape(-1, patch1.shape[-1]).astype(float)
    p2_flat = patch2.reshape(-1, patch2.shape[-1]).astype(float)
    
    # Normalize
    p1_norm = p1_flat / (np.linalg.norm(p1_flat, axis=1, keepdims=True) + 1e-8)
    p2_norm = p2_flat / (np.linalg.norm(p2_flat, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarity matrix
    similarity = np.dot(p1_norm, p2_norm.T)
    return similarity

def visualize_self_correspondence(image, patch_size=16):
    """
    Self-correspondence: Compares patches within the SAME image
    - Helps identify repeated patterns and self-similarities
    - Useful for texture analysis and symmetry detection
    """
    h, w, c = image.shape
    
    # 100% UNSUPERVISED QUERY POINT SELECTION:
    # We find all pixels in the typical liver intensity range (0.2 to 0.7)
    # on the left side of the image (patient's right), and take the MEDIAN coordinate.
    # This guarantees we land squarely in the center of the liver mass, avoiding bone edges.
    gray = image.mean(axis=-1)
    liver_mask = (gray > 0.2) & (gray < 0.7)
    liver_mask[:, int(w * 0.45):] = False  # strictly left side of patient
    
    y_idx, x_idx = np.where(liver_mask)
    
    if len(y_idx) > 0:
        center_h, center_w = int(np.median(y_idx)), int(np.median(x_idx))
    else:
        center_h, center_w = h // 2, w // 4
        
    # Center the patch on this point, keeping it within bounds
    top_h = min(max(0, center_h - patch_size // 2), h - patch_size)
    top_w = min(max(0, center_w - patch_size // 2), w - patch_size)
    
    # Extract query patch
    center_patch = image[top_h:top_h+patch_size, 
                         top_w:top_w+patch_size]
    
    # Compute similarity with all patches in the image
    similarity_map = np.zeros((h - patch_size, w - patch_size))
    
    for i in range(h - patch_size):
        for j in range(w - patch_size):
            patch = image[i:i+patch_size, j:j+patch_size]
            sim = compute_feature_similarity(center_patch, patch)
            similarity_map[i, j] = sim.mean()
    
    return similarity_map, center_patch, top_h, top_w

def visualize_knn_correspondence(image1, image2, patch_size=16, k=3):
    """
    KNN correspondence: Compares patches across DIFFERENT images
    - Finds the k most similar patches in image2 for each patch in image1
    - Establishes cross-image correspondence for consistent segmentation
    """
    h1, w1, c1 = image1.shape
    h2, w2, c2 = image2.shape
    
    # 100% UNSUPERVISED QUERY POINT SELECTION (Median of soft tissue)
    gray1 = image1.mean(axis=-1)
    liver_mask1 = (gray1 > 0.2) & (gray1 < 0.7)
    liver_mask1[:, int(w1 * 0.45):] = False
    
    y_idx, x_idx = np.where(liver_mask1)
    
    if len(y_idx) > 0:
        center_h1, center_w1 = int(np.median(y_idx)), int(np.median(x_idx))
    else:
        center_h1, center_w1 = h1 // 2, w1 // 4
        
    # Center the patch on this point, keeping it within bounds
    top_h1 = min(max(0, center_h1 - patch_size // 2), h1 - patch_size)
    top_w1 = min(max(0, center_w1 - patch_size // 2), w1 - patch_size)
    
    # Extract query patch
    query_patch = image1[top_h1:top_h1+patch_size, 
                         top_w1:top_w1+patch_size]
    
    # Find k most similar patches in image2
    similarities = []
    positions = []
    
    for i in range(h2 - patch_size):
        for j in range(w2 - patch_size):
            patch = image2[i:i+patch_size, j:j+patch_size]
            sim = compute_feature_similarity(query_patch, patch)
            similarities.append(sim.mean())
            positions.append((i, j))
    
    # Get top-k matches
    similarities = np.array(similarities)
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    
    knn_patches = []
    knn_positions = []
    knn_similarities = []
    
    for idx in top_k_indices:
        i, j = positions[idx]
        knn_patch = image2[i:i+patch_size, j:j+patch_size]
        knn_patches.append(knn_patch)
        knn_positions.append((i, j))
        knn_similarities.append(similarities[idx])
    
    return knn_patches, knn_positions, knn_similarities, query_patch, top_h1, top_w1

def main():
    """Main visualization function"""
    print("Loading example CT images from preprocessed folder...")
    images, labels = load_example_images()
    
    if len(images) < 2:
        print("Need at least 2 images for KNN correspondence demo")
        return
    
    # Use first two images for demonstration
    img1, img2 = images[0], images[1]
    label1, label2 = labels[0], labels[1]
    
    # Create visualization
    fig = plt.figure(figsize=(20, 12))
    
    # === Row 1: Original Images and Labels ===
    plt.subplot(3, 4, 1)
    plt.imshow(img1)
    plt.title("Original CT Image 1")
    plt.axis('off')
    
    plt.subplot(3, 4, 2)
    plt.imshow(label1, cmap='gray')
    plt.title("Ground Truth Label 1")
    plt.axis('off')
    
    plt.subplot(3, 4, 3)
    plt.imshow(img2)
    plt.title("Original CT Image 2")
    plt.axis('off')
    
    plt.subplot(3, 4, 4)
    plt.imshow(label2, cmap='gray')
    plt.title("Ground Truth Label 2")
    plt.axis('off')
    
    # === Row 2: Self-Correspondence ===
    print("Computing self-correspondence for Image 1...")
    self_sim_map, center_patch, q_h, q_w = visualize_self_correspondence(img1)
    
    plt.subplot(3, 4, 5)
    plt.imshow(img1)
    plt.title("Image 1 with Query Patch")
    # Highlight query patch region
    plt.gca().add_patch(plt.Rectangle((q_w, q_h), 16, 16, 
                                     linewidth=2, edgecolor='red', facecolor='none'))
    plt.axis('off')
    
    plt.subplot(3, 4, 6)
    plt.imshow(center_patch)
    plt.title("Query Patch (16x16)")
    plt.axis('off')
    
    plt.subplot(3, 4, 7)
    plt.imshow(self_sim_map, cmap='hot')
    plt.title("Self-Correspondence Map\n(Similarity to query patch)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.subplot(3, 4, 8)
    plt.imshow(label1, cmap='gray')
    plt.title("Ground Truth Overlay")
    plt.axis('off')
    
    # === Row 3: KNN Correspondence ===
    print("Computing KNN correspondence between Image 1 and Image 2...")
    knn_patches, knn_positions, knn_sims, query_patch, q_h1, q_w1 = visualize_knn_correspondence(img1, img2, k=3)
    
    plt.subplot(3, 4, 9)
    plt.imshow(img1)
    plt.title("Image 1 (Query)")
    # Highlight query patch region
    plt.gca().add_patch(plt.Rectangle((q_w1, q_h1), 16, 16, 
                                     linewidth=2, edgecolor='red', facecolor='none'))
    plt.axis('off')
    
    plt.subplot(3, 4, 10)
    plt.imshow(query_patch)
    plt.title("Query Patch from Image 1")
    plt.axis('off')
    
    plt.subplot(3, 4, 11)
    plt.imshow(img2)
    plt.title("Image 2 (Database)")
    # Highlight KNN matches
    for idx, (pos, sim) in enumerate(zip(knn_positions, knn_sims)):
        i, j = pos
        alpha = 0.3 + 0.7 * (idx / len(knn_positions))  # Fade for less similar matches
        plt.gca().add_patch(plt.Rectangle((j, i), 16, 16, 
                                          linewidth=2, edgecolor=['red', 'green', 'blue'][idx], 
                                          facecolor='none', alpha=alpha))
    plt.axis('off')
    
    # Show KNN patches
    fig_knn = plt.figure(figsize=(12, 4))
    for idx, (patch, pos, sim) in enumerate(zip(knn_patches, knn_positions, knn_sims)):
        plt.subplot(1, 3, idx+1)
        plt.imshow(patch)
        plt.title(f"KNN Match {idx+1}\nPos: {pos}, Sim: {sim:.3f}")
        plt.axis('off')
    
    plt.tight_layout()
    
    # Save figures
    output_dir = Path("../results/correspondence_visualization")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_dir / "correspondence_demo.png", dpi=150, bbox_inches='tight')
    fig_knn.savefig(output_dir / "knn_matches.png", dpi=150, bbox_inches='tight')
    
    print(f"Visualizations saved to {output_dir}")
    print("\n=== CORRESPONDENCE EXPLANATION ===")
    print("SELF-CORRESPONDENCE:")
    print("- Compares patches within the SAME image")
    print("- Identifies self-similarities and repeated patterns")
    print("- Helps with texture analysis and symmetry detection")
    print("- In medical imaging: finds similar tissue patterns within same scan")
    print()
    print("KNN CORRESPONDENCE:")
    print("- Compares patches across DIFFERENT images")
    print("- Finds k most similar patches in a database image")
    print("- Establishes cross-image correspondence for consistency")
    print("- In medical imaging: matches similar anatomical structures across patients")
    print()
    print("IN YOUR MODEL:")
    print("- Self-correspondence helps the model learn consistent features within each image")
    print("- KNN correspondence ensures similar regions across different images get similar labels")
    print("- This is crucial for unsupervised segmentation to maintain consistency")
    
    # plt.show()  # Removed to prevent blocking the terminal

if __name__ == "__main__":
    main()