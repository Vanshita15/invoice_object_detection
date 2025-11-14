# Complete Knowledge Base Creation & YOLO Training Guide

## Table of Contents
1. [Understanding Your Current Setup](#understanding-your-current-setup)
2. [Knowledge Base Creation](#knowledge-base-creation)
3. [YOLO Model Analysis](#yolo-model-analysis)
4. [Training YOLO from Scratch](#training-yolo-from-scratch)
5. [Best Practices](#best-practices)
6. [Troubleshooting](#troubleshooting)

---

## Understanding Your Current Setup

### Current Knowledge Base Status
Based on your current setup, you have:

```
data/knowledge_base/crops/object/
├── cocacola_can/           (143 images) ✅ GOOD
├── cocacola_pet/           (91 images)  ✅ GOOD  
├── cocacola_zero_can/      (91 images)  ✅ GOOD
├── cream_soda_pet/         (91 images)  ✅ GOOD
├── fanta_pet/              (299 images) 🎯 EXCELLENT
├── minute_maid/            (273 images) 🎯 EXCELLENT
├── multibrand/             (39 images)  🟡 MODERATE
└── sprite_pet/             (78 images)  ✅ GOOD
```

**Total: 1,105 images across 8 categories** - This is actually quite good!

### System Architecture Reminder
```
Input Image → YOLO (finds objects) → Knowledge Base (classifies objects) → Results
```

- **YOLO**: Detects generic "objects" (products on shelves)
- **Knowledge Base**: Classifies detected objects into specific brands

---

## Knowledge Base Creation

### 1. Understanding the Knowledge Base Structure

The knowledge base serves as your **training dataset for product classification**. It should contain:

#### Required Structure:
```
data/knowledge_base/crops/object/
├── [product_category_1]/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── [product_category_2]/
│   ├── image1.jpg
│   └── ...
└── ...
```

#### Naming Conventions:
- **Folder names** = Product categories (e.g., `cocacola_can`, `sprite_pet`)
- **Image names** = Any descriptive name (e.g., `coca_front_1.jpg`)
- **File formats** = `.jpg` or `.png`

### 2. How to Create/Expand Your Knowledge Base

#### Method 1: Manual Collection (Most Accurate)

1. **Take Photos of Products**:
   ```bash
   # Create new category folder
   mkdir "data/knowledge_base/crops/object/new_product_name"
   ```

2. **Photo Guidelines**:
   - **Lighting**: Good, even lighting
   - **Background**: Clean, preferably white/neutral
   - **Angles**: Multiple angles (front, side, slight rotation)
   - **Quality**: Clear, in-focus images
   - **Size**: At least 224x224 pixels
   - **Quantity**: Minimum 20 images per category, ideally 50+

3. **Example Photo Session**:
   ```
   For "pepsi_can" category:
   - pepsi_front_1.jpg
   - pepsi_front_2.jpg (slightly different angle)
   - pepsi_side_1.jpg
   - pepsi_rotated_15deg.jpg
   - pepsi_different_lighting.jpg
   - ... (15-50 more images)
   ```

#### Method 2: Use Existing Detection Results

1. **Run Detection on Shelf Images**:
   ```bash
   python main.py --input "shelf_image.jpg"
   ```

2. **Extract Good Crops**:
   ```bash
   # Crops are saved in data/[image_name]/crops/object/
   # Manually review and move good crops to knowledge base
   ```

3. **Organize Crops**:
   ```python
   # Move correctly identified crops to appropriate folders
   # Example: move crop_001.jpg to data/knowledge_base/crops/object/cocacola_can/
   ```

#### Method 3: Data Augmentation (Expand Existing)

```bash
# Use the augmentation script to multiply your existing data
python improve_accuracy_simple.py --test-image "any_image.jpg"
```

### 3. Knowledge Base Quality Guidelines

#### Image Quality Checklist:
- ✅ **Clear and in-focus**
- ✅ **Good lighting** (not too dark/bright)
- ✅ **Product clearly visible**
- ✅ **Minimal background clutter**
- ✅ **Various angles and conditions**
- ❌ **Avoid blurry images**
- ❌ **Avoid heavily occluded products**
- ❌ **Avoid extreme lighting conditions**

#### Category Balance:
- **Minimum**: 20 images per category
- **Good**: 50+ images per category
- **Excellent**: 100+ images per category
- **Balance**: Similar number of images across categories

---

## YOLO Model Analysis

### 1. Understanding Your Current YOLO Model

Your `models/best.pt` file is likely trained on the **SKU110K dataset**, which contains:
- **1.7 million retail product images**
- **Generic object detection** (finds products, not specific brands)
- **Single class**: "object" (any retail product)

### 2. Verify Your YOLO Model

Run this analysis:
```bash
python analyze_model.py
```

This will tell you:
- What classes your YOLO model can detect
- Whether it's trained for generic or specific detection
- How well it performs on your test images

### 3. Expected YOLO Model Behavior

#### ✅ **Correct Setup** (Generic Object Detection):
```
YOLO Classes: {0: 'object'}
```
- Detects any product as "object"
- Knowledge base handles brand classification
- This is what you want!

#### ⚠️ **Potential Issue** (Specific Product Detection):
```
YOLO Classes: {0: 'coca_cola', 1: 'pepsi', 2: 'sprite', ...}
```
- YOLO tries to classify specific brands
- Conflicts with knowledge base approach
- May need retraining

---

## Training YOLO from Scratch

### When to Retrain YOLO

**Retrain if**:
- Current model has poor detection accuracy
- Model is trained for specific products (conflicts with knowledge base)
- You have a large dataset of annotated shelf images
- You need to detect specific product types (cans vs bottles vs boxes)

**Don't retrain if**:
- Current model detects products well (even if classification is wrong)
- You only have classification issues (knowledge base problem)
- Limited training data (<1000 annotated images)

### 1. Prepare Training Dataset

#### Dataset Structure for YOLO:
```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── val/
│       ├── image1.jpg
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   └── val/
│       ├── image1.txt
│       └── ...
└── data.yaml
```

#### Label Format (YOLO):
Each `.txt` file contains bounding box annotations:
```
# Format: class_id center_x center_y width height (normalized 0-1)
0 0.5 0.3 0.2 0.4
0 0.7 0.6 0.15 0.3
```

#### Create `data.yaml`:
```yaml
# data.yaml
train: dataset/images/train
val: dataset/images/val
nc: 1  # number of classes
names: ['object']  # class names
```

### 2. Annotation Tools

#### Option 1: LabelImg (Recommended)
```bash
pip install labelImg
labelImg
```
- GUI tool for drawing bounding boxes
- Exports YOLO format directly
- Easy to use

#### Option 2: Roboflow (Online)
- Web-based annotation
- Automatic format conversion
- Team collaboration features
- Free tier available

#### Option 3: CVAT (Advanced)
- Professional annotation tool
- Supports multiple formats
- Good for large datasets

### 3. Training Process

#### Step 1: Install Requirements
```bash
pip install ultralytics
```

#### Step 2: Create Training Script
```python
# train_yolo.py
from ultralytics import YOLO

# Load a pretrained model (recommended)
model = YOLO('yolov8m.pt')  # or yolov8n.pt, yolov8s.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(
    data='data.yaml',           # path to dataset YAML
    epochs=100,                 # number of training epochs
    imgsz=640,                  # image size
    batch=16,                   # batch size (adjust based on GPU memory)
    name='product_detection',   # experiment name
    save=True,                  # save checkpoints
    plots=True,                 # save training plots
    device=0,                   # GPU device (0 for first GPU, 'cpu' for CPU)
    workers=4,                  # number of data loading workers
    patience=10,                # early stopping patience
    save_period=10,             # save checkpoint every N epochs
)
```

#### Step 3: Run Training
```bash
python train_yolo.py
```

#### Step 4: Monitor Training
Training outputs will be saved in:
```
runs/detect/product_detection/
├── weights/
│   ├── best.pt      # best model
│   └── last.pt      # last epoch
├── results.png      # training curves
├── confusion_matrix.png
└── ...
```

### 4. Training Parameters Explained

#### Model Sizes:
- **YOLOv8n**: Nano (fastest, least accurate)
- **YOLOv8s**: Small (good balance)
- **YOLOv8m**: Medium (recommended)
- **YOLOv8l**: Large (more accurate, slower)
- **YOLOv8x**: Extra Large (most accurate, slowest)

#### Key Parameters:
- **epochs**: 50-200 (more for complex datasets)
- **batch**: 8-32 (depends on GPU memory)
- **imgsz**: 640 (standard), 1280 (for small objects)
- **patience**: Early stopping (10-20)

### 5. Evaluation and Validation

#### Evaluate Model:
```python
# evaluate_model.py
from ultralytics import YOLO

model = YOLO('runs/detect/product_detection/weights/best.pt')

# Validate on test set
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

#### Test on Images:
```python
# Test on sample images
results = model.predict('test_image.jpg', save=True)
```

---

## Best Practices

### Knowledge Base Best Practices

1. **Quality over Quantity**:
   - 50 high-quality images > 200 poor images
   - Consistent lighting and angles
   - Clear product visibility

2. **Diversity**:
   - Different lighting conditions
   - Various angles and orientations
   - Different backgrounds (within reason)
   - Slight variations in product positioning

3. **Regular Updates**:
   - Add new products as they appear
   - Remove outdated/discontinued products
   - Periodically review and clean data

4. **Validation**:
   - Test classification accuracy regularly
   - Use confusion matrices to identify problems
   - A/B test different knowledge base versions

### YOLO Training Best Practices

1. **Dataset Size**:
   - Minimum: 1,000 images
   - Good: 5,000+ images
   - Excellent: 10,000+ images

2. **Annotation Quality**:
   - Consistent bounding box sizes
   - Include partially visible objects
   - Annotate all objects in image

3. **Data Augmentation**:
   - YOLO includes built-in augmentation
   - Additional augmentation if needed:
     ```python
     # In training script
     model.train(
         data='data.yaml',
         augment=True,  # enable augmentation
         mixup=0.1,     # mixup augmentation
         mosaic=1.0,    # mosaic augmentation
     )
     ```

4. **Hardware Considerations**:
   - **GPU**: Recommended for training
   - **RAM**: 16GB+ for large datasets
   - **Storage**: SSD for faster data loading

---

## Automated Knowledge Base Builder

Let me create a tool to help you build and manage your knowledge base:

```python
# knowledge_base_builder.py
import os
import shutil
from pathlib import Path
from PIL import Image
import argparse

class KnowledgeBaseBuilder:
    def __init__(self, kb_path="data/knowledge_base/crops/object"):
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
    
    def add_category(self, category_name):
        """Create a new product category"""
        category_path = self.kb_path / category_name
        category_path.mkdir(exist_ok=True)
        print(f"✅ Created category: {category_name}")
        return category_path
    
    def add_images(self, category_name, image_paths):
        """Add images to a category"""
        category_path = self.add_category(category_name)
        
        for i, img_path in enumerate(image_paths):
            if Path(img_path).exists():
                # Copy and rename image
                ext = Path(img_path).suffix
                new_name = f"{category_name}_{i+1:03d}{ext}"
                new_path = category_path / new_name
                shutil.copy2(img_path, new_path)
                print(f"  Added: {new_name}")
    
    def validate_images(self):
        """Validate all images in knowledge base"""
        print("🔍 Validating knowledge base images...")
        
        for category_dir in self.kb_path.iterdir():
            if not category_dir.is_dir():
                continue
            
            valid_count = 0
            total_count = 0
            
            for img_file in category_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    total_count += 1
                    try:
                        with Image.open(img_file) as img:
                            # Check if image can be opened and has reasonable size
                            if img.size[0] >= 50 and img.size[1] >= 50:
                                valid_count += 1
                            else:
                                print(f"  ⚠️  Small image: {img_file}")
                    except Exception as e:
                        print(f"  ❌ Corrupt image: {img_file} ({e})")
            
            print(f"  {category_dir.name}: {valid_count}/{total_count} valid images")
    
    def analyze(self):
        """Analyze knowledge base statistics"""
        print("📊 Knowledge Base Analysis:")
        
        categories = {}
        total_images = 0
        
        for category_dir in self.kb_path.iterdir():
            if category_dir.is_dir():
                count = len(list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png")))
                categories[category_dir.name] = count
                total_images += count
        
        print(f"  Total categories: {len(categories)}")
        print(f"  Total images: {total_images}")
        
        for category, count in sorted(categories.items()):
            status = "🔴" if count < 20 else "🟡" if count < 50 else "🟢"
            print(f"  {status} {category}: {count} images")
        
        return categories

if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    builder.analyze()
    builder.validate_images()
```

---

## Troubleshooting

### Common Knowledge Base Issues

#### Issue 1: Poor Classification Accuracy
**Symptoms**: Wrong product predictions
**Solutions**:
- Add more images to problematic categories
- Improve image quality (lighting, focus)
- Remove mislabeled images
- Balance dataset (similar counts per category)

#### Issue 2: New Products Not Recognized
**Symptoms**: Unknown products classified incorrectly
**Solutions**:
- Create new category folder
- Add 20+ images of new product
- Run data augmentation
- Test classification

#### Issue 3: Similar Products Confused
**Symptoms**: Coca-Cola vs Coca-Cola Zero confusion
**Solutions**:
- Add more distinctive images
- Focus on differentiating features
- Increase number of training images
- Use higher quality reference images

### Common YOLO Training Issues

#### Issue 1: Low Detection Accuracy
**Symptoms**: Missing products in shelf images
**Solutions**:
- Lower confidence threshold
- Add more training data
- Improve annotation quality
- Use larger model (YOLOv8m → YOLOv8l)

#### Issue 2: False Positives
**Symptoms**: Detecting non-products as products
**Solutions**:
- Add negative examples to training
- Improve annotation consistency
- Adjust confidence threshold
- Add more diverse backgrounds

#### Issue 3: Training Not Converging
**Symptoms**: Loss not decreasing, poor validation metrics
**Solutions**:
- Check data quality and annotations
- Reduce learning rate
- Increase training epochs
- Use pretrained weights

---

## Summary and Next Steps

### Your Current Status:
✅ **Good knowledge base** (1,105 images across 8 categories)
✅ **Working YOLO model** (detects products)
✅ **Complete pipeline** (detection + classification)

### Recommended Actions:

1. **Immediate** (Today):
   ```bash
   # Analyze your current setup
   python analyze_model.py
   
   # Test current performance
   python improve_accuracy_simple.py --test-image "data/img/testing343.jpg"
   ```

2. **Short-term** (This week):
   - Add more images to `multibrand` category (currently only 39)
   - Test with different shelf images
   - Fine-tune confidence thresholds

3. **Long-term** (If needed):
   - Consider YOLO retraining if detection accuracy is poor
   - Expand to new product categories
   - Implement automated knowledge base updates

### Key Takeaways:
- **Your system is already quite sophisticated**
- **Knowledge base approach is flexible and powerful**
- **YOLO retraining is only needed if detection fails**
- **Focus on knowledge base quality for better accuracy**

The two-stage approach (YOLO + Knowledge Base) is actually more flexible than end-to-end training because you can easily add new products without retraining the entire system!
