"""Quick script to check knowledge base status."""
import pickle
import os
from collections import Counter

KB_FILE = 'data/knowledge_base_embeddings.pkl'

if not os.path.exists(KB_FILE):
    print("❌ Knowledge base not found!")
    print("   Run: python build_knowledge_base.py")
    exit(1)

with open(KB_FILE, 'rb') as f:
    kb_data = pickle.load(f)

classes = kb_data['classes']
embeddings = kb_data['embeddings']
model_type = kb_data.get('model_type', 'unknown')
augmentation_enabled = kb_data.get('augmentation_enabled', False)
class_counts = Counter(classes)

print("="*60)
print("KNOWLEDGE BASE STATUS")
print("="*60)
print(f"Model type: {model_type}")
print(f"Total embeddings: {len(embeddings)}")
print(f"Unique products: {len(class_counts)}")
print(f"Augmentation was enabled: {augmentation_enabled}")
print(f"Embedding dimensions: {embeddings.shape[1]}")

print("\n" + "="*60)
print("PRODUCT DISTRIBUTION")
print("="*60)

for product, count in sorted(class_counts.items()):
    print(f"{product:25s}: {count:4d} samples")

print("\n" + "="*60)
print("ANALYSIS")
print("="*60)

# Calculate expected vs actual
import glob
original_images = glob.glob("data/knowledge_base/crops/object/**/*.jpg")
print(f"Original images in folder: {len(original_images)}")
print(f"Embeddings in knowledge base: {len(embeddings)}")

if augmentation_enabled:
    expected_with_aug = len(original_images) * 4  # 1 original + 3 augmented
    print(f"Expected with augmentation: ~{expected_with_aug}")
    
    if len(embeddings) < len(original_images) * 2:
        print("\n⚠️  WARNING: Augmentation was enabled but embeddings count is low!")
        print("   Recommendation: Rebuild knowledge base")
        print("   Run: python build_knowledge_base.py")
    else:
        print("\n✅ Knowledge base looks good!")
else:
    print("\n⚠️  Augmentation was NOT enabled when building!")
    print("   Recommendation: Rebuild with augmentation for better accuracy")
    print("   Run: python build_knowledge_base.py")

print("="*60)
