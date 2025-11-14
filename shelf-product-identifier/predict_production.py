"""
PRODUCTION-GRADE PREDICTION with all optimizations for challenging retail images.
Combines: TTA + Ensemble + Smart filtering + Quality checks
"""
import os
from ultralytics import YOLO
import glob
import shutil
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter
from pathlib import Path
import argparse
from build_knowledge_base import load_knowledge_base
import cv2

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'
CONFIDENCE_THRESHOLD = 0.60  # Lower for challenging images
MIN_CROP_SIZE = 20  # Minimum crop dimension in pixels
ENABLE_TTA = True
ENABLE_QUALITY_FILTER = True

def is_valid_crop(img_path):
    """Check if crop is valid (not too small, not too blurry)."""
    try:
        img = cv2.imread(img_path)
        if img is None:
            return False
        
        h, w = img.shape[:2]
        
        # Check size
        if h < MIN_CROP_SIZE or w < MIN_CROP_SIZE:
            return False
        
        # Check if too blurry (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # Too blurry
            return False
        
        return True
    except:
        return False

def preprocess_crop(image):
    """Enhance crop quality before embedding extraction."""
    # Slight sharpening
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.2)
    
    # Slight contrast boost
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.1)
    
    return image

def apply_tta_transforms(image):
    """Apply comprehensive TTA transforms."""
    transforms = [image]  # Original
    
    # Brightness variations (for glare/shadows)
    enhancer = ImageEnhance.Brightness(image)
    transforms.append(enhancer.enhance(0.85))
    transforms.append(enhancer.enhance(0.95))
    transforms.append(enhancer.enhance(1.05))
    transforms.append(enhancer.enhance(1.15))
    
    # Contrast variations
    enhancer = ImageEnhance.Contrast(image)
    transforms.append(enhancer.enhance(0.9))
    transforms.append(enhancer.enhance(1.1))
    
    # Sharpness (for blurry crops)
    enhancer = ImageEnhance.Sharpness(image)
    transforms.append(enhancer.enhance(1.3))
    
    return transforms

def predict_with_advanced_tta(img2vec, image, model_knn, classes, class_centroids):
    """
    Advanced TTA prediction with outlier removal.
    """
    if not ENABLE_TTA:
        # Single prediction
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
    all_class_weights = []
    
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
        all_class_weights.append(class_weights)
    
    # Weighted voting across TTA predictions
    # Give more weight to predictions with higher confidence
    final_class_weights = {}
    for pred, conf, weights_dict in zip(all_predictions, all_confidences, all_class_weights):
        for cls, weight in weights_dict.items():
            # Weight by confidence of this TTA prediction
            final_class_weights[cls] = final_class_weights.get(cls, 0.0) + weight * conf
    
    final_product = max(final_class_weights.items(), key=lambda x: x[1])[0]
    
    # Calculate final confidence
    prediction_counts = Counter(all_predictions)
    agreement_ratio = prediction_counts[final_product] / len(all_predictions)
    
    # Base confidence from weighted voting
    base_confidence = final_class_weights[final_product] / sum(final_class_weights.values())
    
    # Adjust by agreement
    final_confidence = base_confidence * (0.7 + 0.3 * agreement_ratio)
    
    # Boost if very high agreement
    if agreement_ratio >= 0.875:  # 7/8 or more agree
        final_confidence = min(final_confidence * 1.15, 0.98)
    
    # Validate against centroid
    if class_centroids and final_product in class_centroids:
        vec = img2vec.getVec(image)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        centroid_dist = np.linalg.norm(vec_norm - class_centroids[final_product])
        
        # Penalize if far from centroid
        if centroid_dist > 0.6:
            final_confidence *= 0.85
        elif centroid_dist < 0.3:
            # Boost if very close to centroid
            final_confidence = min(final_confidence * 1.05, 0.99)
    
    return final_product, final_confidence, all_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Production-grade prediction')
    parser.add_argument('--input', help='Input image path', required=True)
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument('--no-tta', action='store_true', help='Disable TTA')
    parser.add_argument('--yolo-conf', type=float, default=0.4, help='YOLO confidence threshold')
    
    args = parser.parse_args()
    
    if args.no_tta:
        ENABLE_TTA = False
    
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    
    print("="*70)
    print("PRODUCTION-GRADE PRODUCT DETECTION")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"YOLO confidence: {args.yolo_conf}")
    print(f"Classification confidence: {CONFIDENCE_THRESHOLD:.0%}")
    print(f"TTA: {'ENABLED' if ENABLE_TTA else 'DISABLED'}")
    print(f"Quality filtering: {'ENABLED' if ENABLE_QUALITY_FILTER else 'DISABLED'}")
    print("="*70)
    
    # Load knowledge base
    print("\n[1/5] Loading knowledge base...")
    kb_data = load_knowledge_base()
    if kb_data is None:
        print("❌ Run 'python build_knowledge_base.py' first!")
        exit(1)
    
    classes = kb_data['classes']
    embeddings = kb_data['embeddings']
    model_knn = kb_data['model_knn']
    kb_model_type = kb_data['model_type']
    class_centroids = kb_data.get('class_centroids', {})
    optimal_k = kb_data.get('optimal_k', 7)
    
    print(f"✅ Loaded {len(set(classes))} product classes, {len(classes)} samples")
    print(f"   K-neighbors: {optimal_k}")
    
    # Initialize embedding model
    print("\n[2/5] Initializing embedding model...")
    if kb_model_type == "dino2":
        img2vec = Img2VecDino2()
        print("✅ DINO2 model loaded")
    else:
        img2vec = Img2VecResnet18()
        print("✅ ResNet18 model loaded")
    
    # Run YOLO detection
    print("\n[3/5] Running YOLO object detection...")
    yolo_model = YOLO(MODEL_PATH)
    
    PATH = Path(args.input).stem
    results = yolo_model.predict(
        source=args.input,
        save=True,
        save_crop=True,
        conf=args.yolo_conf,  # Lower threshold for challenging images
        project="data",
        name=PATH,
        iou=0.5,
    )
    
    # Find output directory
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        PATH = data_dirs[-1]
    
    print(f"✅ Detection complete, output: data/{PATH}/")
    
    # Get detected crops
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    if not list_imgs:
        print(f"❌ No objects detected in image!")
        print(f"   Try lowering YOLO confidence: --yolo-conf 0.3")
        exit(1)
    
    print(f"✅ Detected {len(list_imgs)} objects")
    
    # Filter invalid crops
    if ENABLE_QUALITY_FILTER:
        print("\n[4/5] Filtering low-quality crops...")
        valid_imgs = [img for img in list_imgs if is_valid_crop(img)]
        filtered_count = len(list_imgs) - len(valid_imgs)
        if filtered_count > 0:
            print(f"⚠️  Filtered out {filtered_count} low-quality crops")
        list_imgs = valid_imgs
    
    if not list_imgs:
        print("❌ No valid crops after filtering!")
        exit(1)
    
    # Classify products
    print(f"\n[5/5] Classifying {len(list_imgs)} products...")
    
    all_predictions = []
    rejected_predictions = []
    
    for i, IMG_DIR in enumerate(list_imgs):
        if i % 5 == 0 or i == len(list_imgs) - 1:
            print(f"  Progress: {i+1}/{len(list_imgs)}")
        
        try:
            I = Image.open(IMG_DIR).convert('RGB')
            
            # Preprocess
            I = preprocess_crop(I)
            
            crop_name = Path(IMG_DIR).stem
            
            # Predict with advanced TTA
            product, confidence, tta_preds = predict_with_advanced_tta(
                img2vec, I, model_knn, classes, class_centroids
            )
            
            I.close()
            
            # Check confidence threshold
            if confidence < CONFIDENCE_THRESHOLD:
                rejected_predictions.append({
                    'crop': crop_name,
                    'product': product,
                    'conf': confidence,
                    'tta_agreement': Counter(tta_preds)[product] / len(tta_preds) if ENABLE_TTA else 1.0
                })
                
                # Move to uncertain folder
                uncertain_dir = f"{DATA_PATH}/{PATH}/crops/uncertain"
                os.makedirs(uncertain_dir, exist_ok=True)
                dest = f"{uncertain_dir}/{crop_name}_{product}_{int(confidence*100)}.jpg"
                shutil.copy2(IMG_DIR, dest)
                os.remove(IMG_DIR)
                continue
            
            tta_agreement = Counter(tta_preds)[product] / len(tta_preds) if ENABLE_TTA else 1.0
            
            all_predictions.append({
                'crop': crop_name,
                'product': product,
                'conf': confidence,
                'tta_agreement': tta_agreement
            })
            
            # Move to product folder
            dest_dir = f"{DATA_PATH}/{PATH}/crops/{product}"
            os.makedirs(dest_dir, exist_ok=True)
            dest_path = f"{dest_dir}/{crop_name}.jpg"
            os.replace(IMG_DIR, dest_path)
            
            # Write predictions
            with open(f"{DATA_PATH}/{PATH}/predictions.txt", "a") as f:
                f.write(f"{crop_name} → {product} (confidence: {confidence:.1%}, agreement: {tta_agreement:.1%})\n")
            
            with open(f"{DATA_PATH}/{PATH}/predictions.csv", "a") as f:
                f.write(f"{crop_name},{product},{confidence:.2%},{tta_agreement:.1%}\n")
            
        except Exception as e:
            print(f"❌ Error processing {Path(IMG_DIR).name}: {e}")
            continue
    
    # Print comprehensive summary
    print("\n" + "="*70)
    print("DETECTION RESULTS")
    print("="*70)
    
    if all_predictions:
        product_counts = Counter([p['product'] for p in all_predictions])
        
        print(f"\n{'Product':<25} {'Count':>8} {'Avg Conf':>12} {'Avg Agreement':>15}")
        print("-"*70)
        
        for product in sorted(product_counts.keys()):
            count = product_counts[product]
            product_preds = [p for p in all_predictions if p['product'] == product]
            avg_conf = np.mean([p['conf'] for p in product_preds])
            avg_agreement = np.mean([p['tta_agreement'] for p in product_preds])
            
            print(f"{product:<25} {count:>8} {avg_conf:>11.1%} {avg_agreement:>14.1%}")
        
        print("-"*70)
        print(f"{'TOTAL ACCEPTED':<25} {len(all_predictions):>8}")
        
        overall_conf = np.mean([p['conf'] for p in all_predictions])
        overall_agreement = np.mean([p['tta_agreement'] for p in all_predictions])
        
        print(f"\nOverall Statistics:")
        print(f"  Average confidence: {overall_conf:.1%}")
        print(f"  Average TTA agreement: {overall_agreement:.1%}")
        
        # Quality indicators
        high_conf = len([p for p in all_predictions if p['conf'] >= 0.80])
        print(f"  High confidence (≥80%): {high_conf}/{len(all_predictions)} ({high_conf/len(all_predictions):.1%})")
    
    if rejected_predictions:
        print(f"\n⚠️  UNCERTAIN PREDICTIONS (< {CONFIDENCE_THRESHOLD:.0%} confidence):")
        print(f"  Total rejected: {len(rejected_predictions)}")
        print(f"  Location: data/{PATH}/crops/uncertain/")
        
        if len(rejected_predictions) <= 10:
            print("\n  Details:")
            for pred in rejected_predictions:
                print(f"    {pred['crop']:<20} → {pred['product']:<20} ({pred['conf']:.1%})")
    
    print("\n" + "="*70)
    print(f"✅ Complete! Results saved to:")
    print(f"   data/{PATH}/predictions.csv")
    print(f"   data/{PATH}/predictions.txt")
    print(f"   data/{PATH}/crops/ (organized by product)")
    print("="*70)
