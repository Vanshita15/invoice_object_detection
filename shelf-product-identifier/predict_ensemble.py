"""
ENSEMBLE PREDICTION - Uses both DINO2 and ResNet18 for maximum accuracy.
Combines predictions from both models for more robust results.
"""
import os
from ultralytics import YOLO
import glob
import shutil
import numpy as np
from src.img2vec_dino2 import Img2VecDino2
from src.img2vec_resnet18 import Img2VecResnet18
from PIL import Image, ImageEnhance
from collections import Counter
from pathlib import Path
import argparse
import pickle

MODEL_PATH = 'models/best.pt'
DATA_PATH = 'data'
CONFIDENCE_THRESHOLD = 0.70

def load_or_build_dual_kb():
    """Load or build knowledge bases for both models."""
    dino_kb_file = 'data/knowledge_base_dino2.pkl'
    resnet_kb_file = 'data/knowledge_base_resnet18.pkl'
    
    # Try to load existing
    if os.path.exists(dino_kb_file) and os.path.exists(resnet_kb_file):
        print("Loading existing dual knowledge bases...")
        with open(dino_kb_file, 'rb') as f:
            dino_kb = pickle.load(f)
        with open(resnet_kb_file, 'rb') as f:
            resnet_kb = pickle.load(f)
        return dino_kb, resnet_kb
    
    # Build both if not exist
    print("Building dual knowledge bases (DINO2 + ResNet18)...")
    print("This will take a few minutes...")
    
    from sklearn.neighbors import NearestNeighbors
    
    list_imgs = glob.glob(f"{DATA_PATH}/knowledge_base/crops/object/**/*.jpg")
    if not list_imgs:
        print("❌ No images found in knowledge base!")
        return None, None
    
    print(f"Processing {len(list_imgs)} images with both models...")
    
    # Initialize both models
    dino_model = Img2VecDino2()
    resnet_model = Img2VecResnet18()
    
    classes = []
    dino_embeddings = []
    resnet_embeddings = []
    image_paths = []
    
    for i, filename in enumerate(list_imgs):
        if i % 50 == 0:
            print(f"  Processing {i+1}/{len(list_imgs)}...")
        
        try:
            I = Image.open(filename).convert('RGB')
            
            # Get embeddings from both models
            dino_vec = dino_model.getVec(I)
            resnet_vec = resnet_model.getVec(I)
            
            I.close()
            
            folder_name = os.path.basename(os.path.dirname(filename))
            
            classes.append(folder_name)
            dino_embeddings.append(dino_vec)
            resnet_embeddings.append(resnet_vec)
            image_paths.append(filename)
            
        except Exception as e:
            print(f"Error: {filename}: {e}")
            continue
    
    # Normalize embeddings
    dino_embeddings = np.array(dino_embeddings)
    resnet_embeddings = np.array(resnet_embeddings)
    
    dino_embeddings = dino_embeddings / (np.linalg.norm(dino_embeddings, axis=1, keepdims=True) + 1e-8)
    resnet_embeddings = resnet_embeddings / (np.linalg.norm(resnet_embeddings, axis=1, keepdims=True) + 1e-8)
    
    # Build KNN models
    class_counts = Counter(classes)
    optimal_k = min(max(7, min(class_counts.values()) // 3), 15)
    
    dino_knn = NearestNeighbors(metric='cosine', n_neighbors=optimal_k, algorithm='brute')
    dino_knn.fit(dino_embeddings)
    
    resnet_knn = NearestNeighbors(metric='cosine', n_neighbors=optimal_k, algorithm='brute')
    resnet_knn.fit(resnet_embeddings)
    
    # Save both
    dino_kb = {
        'classes': classes,
        'embeddings': dino_embeddings,
        'model_knn': dino_knn,
        'optimal_k': optimal_k
    }
    
    resnet_kb = {
        'classes': classes,
        'embeddings': resnet_embeddings,
        'model_knn': resnet_knn,
        'optimal_k': optimal_k
    }
    
    os.makedirs('data', exist_ok=True)
    with open(dino_kb_file, 'wb') as f:
        pickle.dump(dino_kb, f)
    with open(resnet_kb_file, 'wb') as f:
        pickle.dump(resnet_kb, f)
    
    print(f"✅ Dual knowledge bases saved!")
    print(f"   DINO2: {dino_kb_file}")
    print(f"   ResNet18: {resnet_kb_file}")
    
    return dino_kb, resnet_kb

def predict_single_model(img2vec, image, model_knn, classes):
    """Predict using a single model."""
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
    
    return product, confidence, class_weights

def ensemble_predict(dino_model, resnet_model, image, dino_kb, resnet_kb):
    """
    Ensemble prediction combining DINO2 and ResNet18.
    Returns (product, confidence, details)
    """
    # Get predictions from both models
    dino_pred, dino_conf, dino_weights = predict_single_model(
        dino_model, image, dino_kb['model_knn'], dino_kb['classes']
    )
    
    resnet_pred, resnet_conf, resnet_weights = predict_single_model(
        resnet_model, image, resnet_kb['model_knn'], resnet_kb['classes']
    )
    
    # Combine predictions with weighted voting
    # DINO2 gets 60% weight, ResNet18 gets 40% (DINO2 is generally better)
    combined_weights = {}
    
    for cls, weight in dino_weights.items():
        combined_weights[cls] = weight * 0.6
    
    for cls, weight in resnet_weights.items():
        combined_weights[cls] = combined_weights.get(cls, 0.0) + weight * 0.4
    
    final_product = max(combined_weights.items(), key=lambda x: x[1])[0]
    final_confidence = combined_weights[final_product] / sum(combined_weights.values())
    
    # Boost confidence if both models agree
    if dino_pred == resnet_pred == final_product:
        final_confidence = min(final_confidence * 1.15, 0.98)
    
    details = {
        'dino_pred': dino_pred,
        'dino_conf': dino_conf,
        'resnet_pred': resnet_pred,
        'resnet_conf': resnet_conf,
        'agreement': dino_pred == resnet_pred
    }
    
    return final_product, final_confidence, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ensemble prediction (DINO2 + ResNet18)')
    parser.add_argument('--input', help='Input image path', required=True)
    parser.add_argument('--confidence-threshold', type=float, default=CONFIDENCE_THRESHOLD)
    
    args = parser.parse_args()
    CONFIDENCE_THRESHOLD = args.confidence_threshold
    
    # Load or build dual knowledge bases
    dino_kb, resnet_kb = load_or_build_dual_kb()
    if dino_kb is None:
        exit(1)
    
    # Initialize both models
    print("\nInitializing models...")
    dino_model = Img2VecDino2()
    resnet_model = Img2VecResnet18()
    print("✅ Both models loaded")
    
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
    
    data_dirs = [d for d in os.listdir("data") if d.startswith(PATH)]
    if data_dirs:
        PATH = data_dirs[-1]
    
    list_imgs = glob.glob(f"{DATA_PATH}/{PATH}/crops/object/*.jpg")
    
    if not list_imgs:
        print(f"❌ No crops found!")
        exit(1)
    
    print(f"\n🔍 Classifying {len(list_imgs)} objects with ENSEMBLE...")
    
    all_predictions = []
    rejected = []
    
    for i, IMG_DIR in enumerate(list_imgs):
        if i % 10 == 0:
            print(f"  {i+1}/{len(list_imgs)}...")
        
        try:
            I = Image.open(IMG_DIR).convert('RGB')
            crop_name = Path(IMG_DIR).stem
            
            product, confidence, details = ensemble_predict(
                dino_model, resnet_model, I, dino_kb, resnet_kb
            )
            
            I.close()
            
            if confidence < CONFIDENCE_THRESHOLD:
                rejected.append({
                    'crop': crop_name,
                    'product': product,
                    'conf': confidence,
                    'details': details
                })
                
                uncertain_dir = f"{DATA_PATH}/{PATH}/crops/uncertain"
                os.makedirs(uncertain_dir, exist_ok=True)
                shutil.copy2(IMG_DIR, f"{uncertain_dir}/{crop_name}_{product}_{int(confidence*100)}.jpg")
                os.remove(IMG_DIR)
                continue
            
            all_predictions.append({
                'crop': crop_name,
                'product': product,
                'conf': confidence,
                'agreement': details['agreement']
            })
            
            dest_dir = f"{DATA_PATH}/{PATH}/crops/{product}"
            os.makedirs(dest_dir, exist_ok=True)
            os.replace(IMG_DIR, f"{dest_dir}/{crop_name}.jpg")
            
            with open(f"{DATA_PATH}/{PATH}/predictions_ensemble.csv", "a") as f:
                f.write(f"{crop_name},{product},{confidence:.2%},{details['agreement']}\n")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            continue
    
    # Summary
    print("\n" + "="*70)
    print("ENSEMBLE PREDICTION RESULTS")
    print("="*70)
    
    if all_predictions:
        product_counts = Counter([p['product'] for p in all_predictions])
        agreement_rate = sum([p['agreement'] for p in all_predictions]) / len(all_predictions)
        
        for product, count in sorted(product_counts.items()):
            preds = [p for p in all_predictions if p['product'] == product]
            avg_conf = np.mean([p['conf'] for p in preds])
            print(f"{product:20s}: {count:3d} items (avg: {avg_conf:.1%})")
        
        print(f"\n{'Total accepted':20s}: {len(all_predictions)}")
        print(f"{'Model agreement':20s}: {agreement_rate:.1%}")
        print(f"{'Avg confidence':20s}: {np.mean([p['conf'] for p in all_predictions]):.1%}")
    
    if rejected:
        print(f"\n⚠️  Rejected: {len(rejected)} (< {CONFIDENCE_THRESHOLD:.0%} confidence)")
    
    print("\n" + "="*70)
    print(f"✅ Results: {DATA_PATH}/{PATH}/predictions_ensemble.csv")
    print("="*70)
