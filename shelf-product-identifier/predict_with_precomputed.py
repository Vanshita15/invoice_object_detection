import os
from ultralytics import YOLO
import glob
import shutil
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image
from collections import Counter
from pathlib import Path
import argparse
from build_knowledge_base import load_knowledge_base

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'
N_NEIGHBORS = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict products using pre-computed knowledge base')
    parser.add_argument('--input', help='Input file path', required=True)

    # Parse the command-line arguments
    args = parser.parse_args()

    # Load pre-computed knowledge base
    print("Loading pre-computed knowledge base...")
    kb_data = load_knowledge_base()
    if kb_data is None:
        print("❌ Please run 'python build_knowledge_base.py' first to create the knowledge base!")
        exit(1)

    # Extract data from knowledge base
    classes = kb_data['classes']
    embeddings = kb_data['embeddings']
    model_knn = kb_data['model_knn']
    kb_model_type = kb_data['model_type']
    class_centroids = kb_data.get('class_centroids', {})
    optimal_k = kb_data.get('optimal_k', N_NEIGHBORS)
    
    print(f"Using K={optimal_k} neighbors for prediction")

    # Initialize the same embedding model used for knowledge base
    if kb_model_type == "dino2":
        img2vec = Img2VecDino2()
        print("Using DINO2 model for inference")
    else:
        img2vec = Img2VecResnet18()
        print("Using ResNet18 model for inference")

    # Run YOLO detection
    print("Running YOLO detection...")
    yolo_model = YOLO(MODEL_PATH)

    PATH = Path(args.input).stem
    print(f"Expected PATH: {PATH}")

    results = yolo_model.predict(
        source=args.input,
        save=True,
        save_crop=True,
        conf=0.5,
        project="data",
        name=PATH,
    )
    
    # Check what directory was actually created
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        actual_path = data_dirs[-1]  # Get the latest one
        print(f"Actual save directory: {actual_path}")
        PATH = actual_path  # Update PATH to the actual directory name

    # Get detected crop images
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    if not list_imgs:
        print(f"No crop images found in {DATA_PATH}/{PATH}/crops/object/")
        exit(1)

    print(f"Classifying {len(list_imgs)} detected objects...")
    
    # Dictionary to store all predictions for summary
    all_predictions = []
    
    for i, IMG_DIR in enumerate(list_imgs):
        if i % 10 == 0:
            print(f"Processing crop {i+1}/{len(list_imgs)}")

        try:
            # Open the target image file
            I = Image.open(IMG_DIR).convert('RGB')
            # Get the feature vector representation of the target image
            vec = img2vec.getVec(I)
            
            # Normalize the vector
            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            
            # Close the target image file
            I.close()

            # Find the nearest neighbors and distances to the target image
            dists, idx = model_knn.kneighbors([vec_norm])

            # Distance-weighted voting with exponential decay
            neighbor_classes = [classes[i] for i in list(idx[0])]
            neighbor_dists = dists[0]

            # Exponential weighting: closer neighbors have exponentially higher influence
            weights = np.exp(-neighbor_dists * 10)  # Scale factor of 10 for sharper decay

            class_weights = {}
            for cls, w in zip(neighbor_classes, weights):
                class_weights[cls] = class_weights.get(cls, 0.0) + w

            # Predicted product is the class with maximum total weight
            product, product_weight = max(class_weights.items(), key=lambda item: item[1])
            total_weight = sum(class_weights.values())

            crop_name = Path(IMG_DIR).stem
            # Confidence is the fraction of total neighbor weight that voted for this product
            confidence = product_weight / total_weight if total_weight > 0 else 0.0
            
            # Additional validation: check distance to class centroid if available
            if class_centroids and product in class_centroids:
                centroid_dist = np.linalg.norm(vec_norm - class_centroids[product])
                # Adjust confidence based on centroid distance
                # If far from centroid, reduce confidence
                if centroid_dist > 0.5:
                    confidence *= 0.85
            
            # Store prediction for summary
            all_predictions.append({
                'crop_name': crop_name,
                'product': product,
                'confidence': confidence
            })

            # Move crop image into the corresponding folder
            dest_dir = f"{DATA_PATH}/{PATH}/crops/{product}"
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            dest_path = f"{dest_dir}/{crop_name}.jpg"
            os.replace(IMG_DIR, dest_path)

            if confidence < 0.6:
                low_conf_dir = f"{DATA_PATH}/{PATH}/crops/low_confidence"
                if not os.path.exists(low_conf_dir):
                    os.makedirs(low_conf_dir)
                shutil.copy2(dest_path, f"{low_conf_dir}/{crop_name}_{product}_{int(confidence*100)}.jpg")

            # Write predictions to text file
            file_path = f"{DATA_PATH}/{PATH}/predictions.txt"
            with open(file_path, "a", newline="") as file:
                row = f"the crop image {crop_name} is predicted as {product} with a {confidence:.0%} probability\n"
                file.write(row)
                
            # Write CSV file
            with open(f"{DATA_PATH}/{PATH}/predictions.csv", "a", newline="") as file:
                file.write(f"{crop_name},{product},{confidence:.0%}\n")
                
        except Exception as e:
            print(f"Error processing {IMG_DIR}: {e}")
            continue
    
    # Print summary with product counts
    print("\n=== PRODUCT DETECTION SUMMARY ===")
    if all_predictions:
        product_counts = Counter([pred['product'] for pred in all_predictions])
        
        for product, count in sorted(product_counts.items()):
            avg_confidence = sum([pred['confidence'] for pred in all_predictions if pred['product'] == product]) / count
            print(f"{product}: {count} items detected (avg confidence: {avg_confidence:.1%})")
        
        print(f"\nTotal objects detected: {len(all_predictions)}")
        
        # Show low confidence predictions
        low_confidence = [pred for pred in all_predictions if pred['confidence'] < 0.6]
        if low_confidence:
            print(f"\n⚠️  {len(low_confidence)} predictions with low confidence (<60%):")
            for pred in low_confidence:
                print(f"   {pred['crop_name']}: {pred['product']} ({pred['confidence']:.1%})")
    else:
        print("No successful predictions made!")
