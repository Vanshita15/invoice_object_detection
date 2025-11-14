"""
ADVANCED PREDICTION - Solves all major issues:
1. Better YOLO crop quality filtering
2. Per-class adaptive thresholds
3. Similar product disambiguation
4. Quality scoring system
"""
import os
from ultralytics import YOLO
import glob
import shutil
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance, ImageFilter, ImageStat
from collections import Counter, defaultdict
from pathlib import Path
import argparse
from build_knowledge_base import load_knowledge_base
import cv2

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'

# Per-class confidence thresholds (adjust based on class difficulty)
CLASS_THRESHOLDS = {
    'cocacola_can': 0.65,
    'cocacola_zero_can': 0.70,  # Higher because similar to regular coke
    'pepsi': 0.60,
    'pepsi_can': 0.65,
    'sprite_pet': 0.60,
    'sprite_can': 0.65,
    'fanta_pet': 0.55,
    'fanta_can': 0.60,
    'minute_maid': 0.60,
    'mountaindew': 0.60,
    'thums_up': 0.60,
    'thums-up_can': 0.65,
}
DEFAULT_THRESHOLD = 0.60

# Similar product groups (need higher confidence to distinguish)
SIMILAR_GROUPS = [
    ['cocacola_can', 'cocacola_zero_can'],
    ['pepsi', 'pepsi_can'],
    ['sprite_pet', 'sprite_can'],
    ['fanta_pet', 'fanta_can'],
]

def calculate_crop_quality_score(img_path):
    """
    Calculate quality score for a crop (0-100).
    Checks: size, blur, brightness, contrast
    """
    try:
        img = cv2.imread(img_path)
        if img is None:
            return 0
        
        h, w = img.shape[:2]
        score = 100
        
        # Size check
        if h < 30 or w < 30:
            return 0  # Too small
        if h < 50 or w < 50:
            score -= 20  # Small but acceptable
        
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 30:
            return 0  # Too blurry
        elif laplacian_var < 100:
            score -= 15  # Slightly blurry
        
        # Brightness check
        brightness = np.mean(gray)
        if brightness < 30 or brightness > 225:
            score -= 20  # Too dark or too bright
        elif brightness < 50 or brightness > 200:
            score -= 10
        
        # Contrast check
        contrast = np.std(gray)
        if contrast < 20:
            score -= 15  # Low contrast
        
        # Edge detection (good crops have clear edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        if edge_density < 0.05:
            score -= 10  # Very few edges (might be background)
        
        return max(0, score)
        
    except Exception as e:
        return 0

def enhance_crop_quality(image):
    """Enhance crop before embedding extraction."""
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Adaptive sharpening based on image stats
    stat = ImageStat.Stat(image)
    brightness = sum(stat.mean) / len(stat.mean)
    
    # Sharpen
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.3)
    
    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    if brightness < 100:
        image = enhancer.enhance(1.2)  # Boost contrast for dark images
    else:
        image = enhancer.enhance(1.1)
    
    # Adjust brightness if needed
    if brightness < 80:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.15)
    elif brightness > 180:
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.9)
    
    return image

def apply_comprehensive_tta(image):
    """Comprehensive TTA for challenging images."""
    transforms = [image]
    
    # Brightness variations (for different lighting)
    enhancer = ImageEnhance.Brightness(image)
    for factor in [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]:
        transforms.append(enhancer.enhance(factor))
    
    # Contrast variations
    enhancer = ImageEnhance.Contrast(image)
    for factor in [0.9, 0.95, 1.05, 1.1]:
        transforms.append(enhancer.enhance(factor))
    
    # Sharpness (for blurry crops)
    enhancer = ImageEnhance.Sharpness(image)
    transforms.append(enhancer.enhance(1.3))
    transforms.append(enhancer.enhance(1.5))
    
    # Color saturation
    enhancer = ImageEnhance.Color(image)
    transforms.append(enhancer.enhance(0.95))
    transforms.append(enhancer.enhance(1.05))
    
    return transforms

def predict_with_disambiguation(img2vec, image, model_knn, classes, class_centroids):
    """
    Advanced prediction with similar product disambiguation.
    """
    # Get comprehensive TTA predictions
    augmented_images = apply_comprehensive_tta(image)
    all_predictions = []
    all_confidences = []
    all_class_weights_list = []
    
    for aug_img in augmented_images:
        vec = img2vec.getVec(aug_img)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        
        dists, idx = model_knn.kneighbors([vec_norm])
        neighbor_classes = [classes[i] for i in idx[0]]
        neighbor_dists = dists[0]
        
        # Exponential weighting with stronger decay
        weights = np.exp(-neighbor_dists * 12)  # Increased from 10
        
        class_weights = {}
        for cls, w in zip(neighbor_classes, weights):
            class_weights[cls] = class_weights.get(cls, 0.0) + w
        
        pred_product = max(class_weights.items(), key=lambda x: x[1])[0]
        pred_confidence = class_weights[pred_product] / sum(class_weights.values())
        
        all_predictions.append(pred_product)
        all_confidences.append(pred_confidence)
        all_class_weights_list.append(class_weights)
    
    # Aggregate predictions with confidence weighting
    final_class_weights = {}
    for pred, conf, weights_dict in zip(all_predictions, all_confidences, all_class_weights_list):
        for cls, weight in weights_dict.items():
            final_class_weights[cls] = final_class_weights.get(cls, 0.0) + weight * conf
    
    # Get top 2 predictions
    sorted_predictions = sorted(final_class_weights.items(), key=lambda x: x[1], reverse=True)
    top1_product, top1_weight = sorted_predictions[0]
    
    # Calculate base confidence
    prediction_counts = Counter(all_predictions)
    agreement_ratio = prediction_counts[top1_product] / len(all_predictions)
    base_confidence = final_class_weights[top1_product] / sum(final_class_weights.values())
    
    # Adjust by agreement
    final_confidence = base_confidence * (0.65 + 0.35 * agreement_ratio)
    
    # Check if top prediction is in a similar group
    in_similar_group = False
    similar_competitor = None
    
    for group in SIMILAR_GROUPS:
        if top1_product in group:
            in_similar_group = True
            # Check if competitor from same group is in top predictions
            for other_product in group:
                if other_product != top1_product and other_product in final_class_weights:
                    competitor_weight = final_class_weights[other_product]
                    ratio = top1_weight / (competitor_weight + 1e-6)
                    
                    if ratio < 1.5:  # Too close, not confident
                        similar_competitor = other_product
                        # Penalize confidence heavily
                        final_confidence *= 0.7
                    elif ratio < 2.0:
                        # Somewhat close
                        final_confidence *= 0.85
            break
    
    # Boost if very high agreement and no similar competitor
    if agreement_ratio >= 0.80 and not similar_competitor:
        final_confidence = min(final_confidence * 1.15, 0.98)
    
    # Centroid validation
    if class_centroids and top1_product in class_centroids:
        vec = img2vec.getVec(image)
        vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
        centroid_dist = np.linalg.norm(vec_norm - class_centroids[top1_product])
        
        if centroid_dist > 0.6:
            final_confidence *= 0.80
        elif centroid_dist < 0.25:
            final_confidence = min(final_confidence * 1.08, 0.99)
    
    metadata = {
        'agreement': agreement_ratio,
        'similar_competitor': similar_competitor,
        'in_similar_group': in_similar_group,
        'top2': sorted_predictions[:2] if len(sorted_predictions) > 1 else sorted_predictions
    }
    
    return top1_product, final_confidence, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced prediction with quality filtering')
    parser.add_argument('--input', help='Input image path', required=True)
    parser.add_argument('--yolo-conf', type=float, default=0.35, help='YOLO confidence')
    parser.add_argument('--min-quality', type=int, default=40, help='Minimum crop quality score (0-100)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("ADVANCED PRODUCT DETECTION SYSTEM")
    print("="*70)
    print(f"Input: {args.input}")
    print(f"YOLO confidence: {args.yolo_conf}")
    print(f"Minimum quality score: {args.min_quality}/100")
    print(f"Per-class thresholds: ENABLED")
    print(f"Similar product disambiguation: ENABLED")
    print("="*70)
    
    # Load knowledge base
    print("\n[1/6] Loading knowledge base...")
    kb_data = load_knowledge_base()
    if kb_data is None:
        print("❌ Run 'python build_knowledge_base.py' first!")
        exit(1)
    
    classes = kb_data['classes']
    model_knn = kb_data['model_knn']
    kb_model_type = kb_data['model_type']
    class_centroids = kb_data.get('class_centroids', {})
    
    print(f"✅ Loaded {len(set(classes))} product classes")
    
    # Initialize model
    print("\n[2/6] Initializing embedding model...")
    if kb_model_type == "dino2":
        img2vec = Img2VecDino2()
        print("✅ DINO2 loaded")
    else:
        img2vec = Img2VecResnet18()
        print("✅ ResNet18 loaded")
    
    # YOLO detection
    print("\n[3/6] Running YOLO detection...")
    yolo_model = YOLO(MODEL_PATH)
    
    PATH = Path(args.input).stem
    results = yolo_model.predict(
        source=args.input,
        save=True,
        save_crop=True,
        conf=args.yolo_conf,
        project="data",
        name=PATH,
        iou=0.45,  # Lower IOU for better separation
    )
    
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        PATH = data_dirs[-1]
    
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    if not list_imgs:
        print(f"❌ No objects detected!")
        exit(1)
    
    print(f"✅ Detected {len(list_imgs)} objects")
    
    # Quality filtering
    print(f"\n[4/6] Quality filtering (min score: {args.min_quality})...")
    crop_qualities = {}
    valid_imgs = []
    
    for img_path in list_imgs:
        quality = calculate_crop_quality_score(img_path)
        crop_qualities[img_path] = quality
        if quality >= args.min_quality:
            valid_imgs.append(img_path)
    
    filtered = len(list_imgs) - len(valid_imgs)
    if filtered > 0:
        print(f"⚠️  Filtered {filtered} low-quality crops")
        # Save filtered crops for review
        filtered_dir = f"{DATA_PATH}/{PATH}/crops/filtered_low_quality"
        os.makedirs(filtered_dir, exist_ok=True)
        for img_path in list_imgs:
            if img_path not in valid_imgs:
                quality = crop_qualities[img_path]
                name = Path(img_path).stem
                shutil.copy2(img_path, f"{filtered_dir}/{name}_q{quality}.jpg")
    
    list_imgs = valid_imgs
    print(f"✅ {len(list_imgs)} crops passed quality check")
    
    # Classification
    print(f"\n[5/6] Classifying products...")
    
    all_predictions = []
    rejected_predictions = []
    
    for i, IMG_DIR in enumerate(list_imgs):
        if i % 5 == 0 or i == len(list_imgs) - 1:
            print(f"  {i+1}/{len(list_imgs)}...")
        
        try:
            I = Image.open(IMG_DIR).convert('RGB')
            I = enhance_crop_quality(I)
            
            crop_name = Path(IMG_DIR).stem
            quality_score = crop_qualities[IMG_DIR]
            
            # Predict
            product, confidence, metadata = predict_with_disambiguation(
                img2vec, I, model_knn, classes, class_centroids
            )
            
            I.close()
            
            # Get per-class threshold
            threshold = CLASS_THRESHOLDS.get(product, DEFAULT_THRESHOLD)
            
            # Adjust threshold based on quality
            if quality_score < 60:
                threshold += 0.05  # Require higher confidence for low quality crops
            
            if confidence < threshold:
                rejected_predictions.append({
                    'crop': crop_name,
                    'product': product,
                    'conf': confidence,
                    'threshold': threshold,
                    'quality': quality_score,
                    'metadata': metadata
                })
                
                uncertain_dir = f"{DATA_PATH}/{PATH}/crops/uncertain"
                os.makedirs(uncertain_dir, exist_ok=True)
                dest = f"{uncertain_dir}/{crop_name}_{product}_{int(confidence*100)}_q{quality_score}.jpg"
                shutil.copy2(IMG_DIR, dest)
                os.remove(IMG_DIR)
                continue
            
            all_predictions.append({
                'crop': crop_name,
                'product': product,
                'conf': confidence,
                'quality': quality_score,
                'agreement': metadata['agreement'],
                'similar_competitor': metadata['similar_competitor']
            })
            
            dest_dir = f"{DATA_PATH}/{PATH}/crops/{product}"
            os.makedirs(dest_dir, exist_ok=True)
            os.replace(IMG_DIR, f"{dest_dir}/{crop_name}.jpg")
            
            # Write detailed predictions
            with open(f"{DATA_PATH}/{PATH}/predictions_detailed.csv", "a") as f:
                competitor = metadata['similar_competitor'] or 'none'
                f.write(f"{crop_name},{product},{confidence:.2%},{quality_score},{metadata['agreement']:.1%},{competitor}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("[6/6] RESULTS SUMMARY")
    print("="*70)
    
    if all_predictions:
        product_counts = Counter([p['product'] for p in all_predictions])
        
        print(f"\n{'Product':<25} {'Count':>6} {'Avg Conf':>10} {'Avg Quality':>12}")
        print("-"*70)
        
        for product in sorted(product_counts.keys()):
            preds = [p for p in all_predictions if p['product'] == product]
            avg_conf = np.mean([p['conf'] for p in preds])
            avg_quality = np.mean([p['quality'] for p in preds])
            count = len(preds)
            
            print(f"{product:<25} {count:>6} {avg_conf:>9.1%} {avg_quality:>11.0f}/100")
        
        print("-"*70)
        print(f"{'TOTAL ACCEPTED':<25} {len(all_predictions):>6}")
        
        # Warnings for similar products
        similar_warnings = [p for p in all_predictions if p['similar_competitor']]
        if similar_warnings:
            print(f"\n⚠️  {len(similar_warnings)} predictions had similar competitors:")
            for p in similar_warnings[:5]:
                print(f"   {p['crop']}: {p['product']} vs {p['similar_competitor']} ({p['conf']:.1%})")
    
    if rejected_predictions:
        print(f"\n❌ REJECTED: {len(rejected_predictions)} predictions")
        print(f"   Location: data/{PATH}/crops/uncertain/")
        
        # Group by reason
        low_conf = [p for p in rejected_predictions if p['conf'] < p['threshold'] - 0.1]
        borderline = [p for p in rejected_predictions if p not in low_conf]
        
        if low_conf:
            print(f"   Low confidence: {len(low_conf)}")
        if borderline:
            print(f"   Borderline: {len(borderline)} (review these!)")
    
    print("\n" + "="*70)
    print(f"✅ Results: data/{PATH}/predictions_detailed.csv")
    print("="*70)
