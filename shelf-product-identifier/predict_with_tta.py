"""
Enhanced prediction with Test-Time Augmentation (TTA) for better accuracy.
This applies multiple transformations to each crop and votes on the results.
"""
import os
from ultralytics import YOLO
import glob
import shutil
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance, ImageOps
from collections import Counter
from pathlib import Path
import argparse
from build_knowledge_base import load_knowledge_base

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'
CONFIDENCE_THRESHOLD = 0.50  # Reject predictions below this
ENABLE_TTA = True  # Test-Time Augmentation

def apply_tta_transforms(image):
    """
    Apply Test-Time Augmentation transforms.
    Returns list of transformed images including original.
    """
    transforms = [image]  # Original
    
    # Brightness variations
    enhancer = ImageEnhance.Brightness(image)
    transforms.append(enhancer.enhance(0.9))
    transforms.append(enhancer.enhance(1.1))
    
    # Contrast variations
    enhancer = ImageEnhance.Contrast(image)
    transforms.append(enhancer.enhance(0.95))
    transforms.append(enhancer.enhance(1.05))
    
    return transforms

def predict_with_tta(img2vec, image, model_knn, classes, class_centroids):
    """
    Predict using Test-Time Augmentation.
    Returns (product, confidence, all_predictions)
    """
    if not ENABLE_TTA:
        # Single prediction without TTA
        vec = img2vec.getVec(image)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        dists, idx = model_knn.kneighbors([vec_norm])
        
        neighbor_classes = [classes[i] for i in idx[0]]
        neighbor_dists = dists[0]
        weights = np.exp(-neighbor_dists * 10)
        
        class_weights = {}
        for cls, w in zip(neighbor_classes, weights):
            class_weights[cls] = class_weights.get(cls, 0.0) + w
        
        product = max(class_weights.items(), key=lambda x: x[1])[0]
        confidence = class_weights[product] / sum(class_weights.values())
        
        return product, confidence, [product]
    
    # TTA: predict on multiple augmentations
    augmented_images = apply_tta_transforms(image)
    all_predictions = []
    all_confidences = []
    
    for aug_img in augmented_images:
        vec = img2vec.getVec(aug_img)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        
        dists, idx = model_knn.kneighbors([vec_norm])
        neighbor_classes = [classes[i] for i in idx[0]]
        neighbor_dists = dists[0]
        
        # Exponential weighting
        weights = np.exp(-neighbor_dists * 10)
        
        class_weights = {}
        for cls, w in zip(neighbor_classes, weights):
            class_weights[cls] = class_weights.get(cls, 0.0) + w
        
        pred_product = max(class_weights.items(), key=lambda x: x[1])[0]
        pred_confidence = class_weights[pred_product] / sum(class_weights.values())
        
        all_predictions.append(pred_product)
        all_confidences.append(pred_confidence)
    
    # Vote across all TTA predictions
    prediction_counts = Counter(all_predictions)
    final_product = prediction_counts.most_common(1)[0][0]
    
    # Confidence is the average confidence of predictions that match final_product
    matching_confidences = [conf for pred, conf in zip(all_predictions, all_confidences) 
                           if pred == final_product]
    final_confidence = np.mean(matching_confidences)
    
    # Boost confidence if all TTA predictions agree
    agreement_ratio = prediction_counts[final_product] / len(all_predictions)
    if agreement_ratio == 1.0:
        final_confidence = min(final_confidence * 1.1, 0.99)
    
    # Validate against centroid
    if class_centroids and final_product in class_centroids:
        vec = img2vec.getVec(image)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        centroid_dist = np.linalg.norm(vec_norm - class_centroids[final_product])
        if centroid_dist > 0.5:
            final_confidence *= 0.9
    
    return final_product, final_confidence, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Enhanced prediction with TTA')
    parser.add_argument('--input', help='Input file path', required=True)
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD,
                       help='Minimum confidence threshold (default: 0.65)')
    parser.add_argument('--no-tta', action='store_true', help='Disable test-time augmentation')

    args = parser.parse_args()
    
    if args.no_tta:
        ENABLE_TTA = False
        print("⚠️  Test-Time Augmentation DISABLED")
    
    CONFIDENCE_THRESHOLD = args.confidence_threshold

    # Load pre-computed knowledge base
    print("Loading pre-computed knowledge base...")
    kb_data = load_knowledge_base()
    if kb_data is None:
        print("❌ Please run 'python build_knowledge_base.py' first!")
        exit(1)

    classes = kb_data['classes']
    embeddings = kb_data['embeddings']
    model_knn = kb_data['model_knn']
    kb_model_type = kb_data['model_type']
    class_centroids = kb_data.get('class_centroids', {})
    optimal_k = kb_data.get('optimal_k', 7)
    
    print(f"Using K={optimal_k} neighbors")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"TTA enabled: {ENABLE_TTA}")

    # Initialize embedding model
    if kb_model_type == "dino2":
        img2vec = Img2VecDino2()
        print("Using DINO2 model")
    else:
        img2vec = Img2VecResnet18()
        print("Using ResNet18 model")

    # Run YOLO detection
    print("\nRunning YOLO detection...")
    yolo_model = YOLO(MODEL_PATH)

    PATH = Path(args.input).stem
    results = yolo_model.predict(
        source=args.input,
        save=True,
        save_crop=True,
        conf=0.5,
        project="data",
        name=PATH,
    )
    
    # Find actual output directory
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        PATH = data_dirs[-1]
        print(f"Output directory: {PATH}")

    # Get detected crops
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    if not list_imgs:
        print(f"❌ No crops found in {DATA_PATH}/{PATH}/crops/object/")
        exit(1)

    print(f"\nClassifying {len(list_imgs)} detected objects...")
    
    all_predictions = []
    rejected_predictions = []
    
    for i, IMG_DIR in enumerate(list_imgs):
        if i % 10 == 0:
            print(f"Processing {i+1}/{len(list_imgs)}...")

        try:
            I = Image.open(IMG_DIR).convert('RGB')
            crop_name = Path(IMG_DIR).stem
            
            # Predict with TTA
            product, confidence, tta_preds = predict_with_tta(
                img2vec, I, model_knn, classes, class_centroids
            )
            
            I.close()
            
            # Check confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                rejected_predictions.append({
                    'crop_name': crop_name,
                    'product': product,
                    'confidence': confidence,
                    'reason': 'low_confidence'
                })
                
                # Move to uncertain folder
                uncertain_dir = f"{DATA_PATH}/{PATH}/crops/uncertain"
                os.makedirs(uncertain_dir, exist_ok=True)
                dest = f"{uncertain_dir}/{crop_name}_{product}_{int(confidence*100)}.jpg"
                shutil.copy2(IMG_DIR, dest)
                os.remove(IMG_DIR)
                continue
            
            all_predictions.append({
                'crop_name': crop_name,
                'product': product,
                'confidence': confidence,
                'tta_agreement': Counter(tta_preds)[product] / len(tta_preds) if ENABLE_TTA else 1.0
            })

            # Move to product folder
            dest_dir = f"{DATA_PATH}/{PATH}/crops/{product}"
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = f"{dest_dir}/{crop_name}.jpg"
            os.replace(IMG_DIR, dest_path)

            # Write predictions
            with open(f"{DATA_PATH}/{PATH}/predictions.txt", "a") as f:
                f.write(f"{crop_name},{product},{confidence:.0%}\n")
                
            with open(f"{DATA_PATH}/{PATH}/predictions.csv", "a") as f:
                f.write(f"{crop_name},{product},{confidence:.2%}\n")
                
        except Exception as e:
            print(f"❌ Error processing {IMG_DIR}: {e}")
            continue
    
    # Print comprehensive summary
    print("\n" + "="*60)
    print("PRODUCT DETECTION SUMMARY")
    print("="*60)
    
    if all_predictions:
        product_counts = Counter([p['product'] for p in all_predictions])
        
        for product, count in sorted(product_counts.items()):
            product_preds = [p for p in all_predictions if p['product'] == product]
            avg_conf = np.mean([p['confidence'] for p in product_preds])
            print(f"{product:20s}: {count:3d} items (avg conf: {avg_conf:.1%})")
        
        print(f"\n{'Total accepted':20s}: {len(all_predictions)} items")
        
        avg_confidence = np.mean([p['confidence'] for p in all_predictions])
        print(f"{'Average confidence':20s}: {avg_confidence:.1%}")
    
    if rejected_predictions:
        print(f"\n⚠️  REJECTED PREDICTIONS (confidence < {CONFIDENCE_THRESHOLD:.0%}):")
        print(f"{'Total rejected':20s}: {len(rejected_predictions)} items")
        
        for pred in rejected_predictions[:10]:  # Show first 10
            print(f"  {pred['crop_name']:15s} → {pred['product']:15s} ({pred['confidence']:.1%})")
        
        if len(rejected_predictions) > 30:
            print(f"  ... and {len(rejected_predictions) - 10} more")
        
        print(f"\n💡 Review uncertain predictions in: {DATA_PATH}/{PATH}/crops/uncertain/")
    
    print("\n" + "="*60)
    print(f"✅ Results saved to: {DATA_PATH}/{PATH}/predictions.csv")
    print("="*60)
