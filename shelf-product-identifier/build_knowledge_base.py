import os
import glob
import pickle
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance, ImageOps
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
from collections import Counter
import warnings

DATA_PATH = 'data'
EMBEDDINGS_FILE = 'data/knowledge_base_embeddings.pkl'
MODEL_TYPE = "dino2"  # options: "resnet18" or "dino2"

# Augmentation settings for better generalization
ENABLE_AUGMENTATION = True
AUGMENTATION_PER_IMAGE = 3  # Number of augmented versions per original image

# Duplicate removal settings
ENABLE_DUPLICATE_REMOVAL = False
DUPLICATE_THRESHOLD = 0.995  # Only remove if >99.5% similar (exact duplicates only)

def augment_image(image):
    """
    Apply subtle augmentations to improve model robustness.
    Returns augmented PIL Image.
    """
    augmented_images = []
    
    # Augmentation 1: Slight brightness variation
    enhancer = ImageEnhance.Brightness(image)
    aug1 = enhancer.enhance(np.random.uniform(0.88, 1.12))
    augmented_images.append(aug1)
    
    # Augmentation 2: Slight contrast variation
    enhancer = ImageEnhance.Contrast(image)
    aug2 = enhancer.enhance(np.random.uniform(0.92, 1.08))
    augmented_images.append(aug2)
    
    # Augmentation 3: Slight color/saturation variation
    enhancer = ImageEnhance.Color(image)
    aug3 = enhancer.enhance(np.random.uniform(0.95, 1.05))
    augmented_images.append(aug3)
    
    return augmented_images[:AUGMENTATION_PER_IMAGE]

def remove_duplicate_embeddings(embeddings, classes, image_paths, threshold=0.995):
    """
    Remove near-duplicate embeddings that could bias the model.
    Uses cosine similarity threshold.
    Only removes EXACT duplicates (same image added twice), not augmented versions.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    embeddings_array = np.array(embeddings)
    similarities = cosine_similarity(embeddings_array)
    
    # Set diagonal to 0 to ignore self-similarity
    np.fill_diagonal(similarities, 0)
    
    to_remove = set()
    for i in range(len(similarities)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(similarities)):
            if j in to_remove:
                continue
            if similarities[i, j] > threshold:
                # Only remove if they're from the SAME original image (exact duplicates)
                # Check if paths are identical (not augmented versions)
                path_i = str(image_paths[i]).replace('_aug0', '').replace('_aug1', '').replace('_aug2', '')
                path_j = str(image_paths[j]).replace('_aug0', '').replace('_aug1', '').replace('_aug2', '')
                
                if path_i == path_j and classes[i] == classes[j]:
                    # This is likely an exact duplicate, keep first one
                    to_remove.add(j)
    
    if to_remove:
        print(f"Removing {len(to_remove)} exact duplicate images (similarity > {threshold})")
    else:
        print("No exact duplicates found - all embeddings retained")
        
    # Keep only non-duplicates
    keep_indices = [i for i in range(len(embeddings)) if i not in to_remove]
    return (
        [embeddings[i] for i in keep_indices],
        [classes[i] for i in keep_indices],
        [image_paths[i] for i in keep_indices]
    )

def build_knowledge_base():
    """
    Generate embeddings for all images in the knowledge base and save them.
    This should be run whenever you add new images to the knowledge base.
    """
    print("Building knowledge base embeddings...")
    print(f"Augmentation: {'ENABLED' if ENABLE_AUGMENTATION else 'DISABLED'}")
    
    # Get a list of image file paths using glob
    list_imgs = glob.glob(f"{DATA_PATH}/knowledge_base/crops/object/**/*.jpg")
    
    if not list_imgs:
        print(f"No images found in {DATA_PATH}/knowledge_base/crops/object/")
        return
    
    print(f"Found {len(list_imgs)} images in knowledge base")
    
    # Create an instance of the embedding model
    if MODEL_TYPE == "dino2":
        img2vec = Img2VecDino2()
        print("Using DINO2 model for embeddings")
    else:
        img2vec = Img2VecResnet18()
        print("Using ResNet18 model for embeddings")

    # Create empty lists to store classes and embeddings
    classes = []
    embeddings = []
    image_paths = []

    # Iterate over each image file
    total_to_process = len(list_imgs)
    for i, filename in enumerate(list_imgs):
        if i % 50 == 0:
            print(f"Processing image {i+1}/{total_to_process}")
            
        try:
            # Open the image file
            I = Image.open(filename).convert('RGB')

            # Get the feature vector representation of the original image
            vec = img2vec.getVec(I)

            # Extract the folder path and name of the image file
            folder_path = os.path.dirname(filename)
            folder_name = os.path.basename(folder_path)

            # Append the folder name (class), feature vector, and path to the lists
            classes.append(folder_name)
            embeddings.append(vec)
            image_paths.append(filename)
            
            # Apply augmentation if enabled
            if ENABLE_AUGMENTATION:
                augmented_imgs = augment_image(I)
                for aug_idx, aug_img in enumerate(augmented_imgs):
                    aug_vec = img2vec.getVec(aug_img)
                    classes.append(folder_name)
                    embeddings.append(aug_vec)
                    image_paths.append(f"{filename}_aug{aug_idx}")
            
            # Close the image file
            I.close()
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    if not embeddings:
        print("No valid embeddings generated!")
        return

    print(f"\nTotal embeddings generated: {len(embeddings)}")
    
    # Remove near-duplicates (only exact duplicates, not augmented versions)
    if ENABLE_DUPLICATE_REMOVAL:
        embeddings, classes, image_paths = remove_duplicate_embeddings(
            embeddings, classes, image_paths, threshold=DUPLICATE_THRESHOLD
        )
        print(f"Total embeddings after duplicate removal: {len(embeddings)}")
    else:
        print("Duplicate removal: DISABLED - keeping all embeddings")

    # Convert embeddings to numpy array and normalize
    embeddings = np.array(embeddings)
    
    # L2 normalization for better cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Print class distribution
    class_counts = Counter(classes)
    print("\n=== KNOWLEDGE BASE STATISTICS ===")
    print(f"Total images processed: {len(embeddings)}")
    print(f"Number of product classes: {len(class_counts)}")
    print("\nClass distribution:")
    for product, count in sorted(class_counts.items()):
        print(f"  {product}: {count} samples")
    
    # Check for imbalanced classes
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 2.5:
        print(f"\n⚠️  WARNING: Imbalanced dataset detected!")
        print(f"   Largest class: {max_count} samples")
        print(f"   Smallest class: {min_count} samples")
        print(f"   Imbalance ratio: {imbalance_ratio:.2f}x")
        print(f"   Consider adding more images to smaller classes for better performance.")
    else:
        print(f"\n✓ Dataset is well-balanced (ratio: {imbalance_ratio:.2f}x)")
    
    # Check for naming issues
    print("\n=== QUALITY CHECKS ===")
    potential_typos = []
    class_names = list(class_counts.keys())
    for name in class_names:
        if 'sprit' in name.lower() and 'sprite' not in name.lower():
            potential_typos.append(f"'{name}' - possible typo? (should be 'sprite')")
    
    if potential_typos:
        print("⚠️  Potential naming issues detected:")
        for typo in potential_typos:
            print(f"   {typo}")
    else:
        print("✓ No obvious naming issues detected")

    # Adaptive K-neighbors based on smallest class
    optimal_k = min(max(7, min_count // 3), 15)  # Between 7-15 neighbors
    print(f"\n=== KNN CONFIGURATION ===")
    print(f"Optimal K-neighbors: {optimal_k} (based on smallest class size)")
    
    # Create and fit the KNN model
    print("Fitting KNN model...")
    model_knn = NearestNeighbors(metric='cosine', n_neighbors=optimal_k, algorithm='brute')
    model_knn.fit(embeddings)

    # Calculate class centroids for additional validation
    class_centroids = {}
    for class_name in class_counts.keys():
        class_mask = np.array([c == class_name for c in classes])
        class_embeddings = embeddings[class_mask]
        centroid = np.mean(class_embeddings, axis=0)
        class_centroids[class_name] = centroid
    
    # Save everything to a pickle file
    knowledge_base_data = {
        'classes': classes,
        'embeddings': embeddings,
        'image_paths': image_paths,
        'model_knn': model_knn,
        'model_type': MODEL_TYPE,
        'embedding_dim': embeddings.shape[1],
        'class_centroids': class_centroids,
        'optimal_k': optimal_k,
        'class_counts': dict(class_counts),
        'augmentation_enabled': ENABLE_AUGMENTATION
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    
    with open(EMBEDDINGS_FILE, 'wb') as f:
        pickle.dump(knowledge_base_data, f)
    
    print(f"\n✅ Knowledge base saved to: {EMBEDDINGS_FILE}")
    print(f"   Embedding dimensions: {embeddings.shape}")
    print(f"   Model type: {MODEL_TYPE}")
    print(f"   K-neighbors: {optimal_k}")
    print(f"   Augmentation: {'Yes' if ENABLE_AUGMENTATION else 'No'}")
    print("\n🚀 Knowledge base is ready for inference!")

def load_knowledge_base():
    """
    Load the pre-computed knowledge base embeddings.
    Returns the loaded data or None if file doesn't exist.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print(f"Knowledge base file not found: {EMBEDDINGS_FILE}")
        print("Please run build_knowledge_base() first.")
        return None
    
    with open(EMBEDDINGS_FILE, 'rb') as f:
        data = pickle.load(f)
    
    print(f"✅ Loaded knowledge base with {len(data['classes'])} images")
    print(f"   Model type: {data['model_type']}")
    print(f"   Embedding dimensions: {data['embedding_dim']}")
    
    return data

if __name__ == "__main__":
    build_knowledge_base()
