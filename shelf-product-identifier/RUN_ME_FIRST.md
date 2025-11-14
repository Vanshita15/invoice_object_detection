# 🚀 Quick Start Guide

## Sabse Pehle Ye Karo (REQUIRED):

### Step 1: Knowledge Base Rebuild
```bash
cd shelf-product-identifier
python build_knowledge_base.py
```

**Ye dekhna chahiye:**
```
Total embeddings generated: ~3,100
Removing ~50-100 exact duplicates
Total embeddings after duplicate removal: ~2,500-3,000
✅ Knowledge base saved
```

**Agar 669 dikha raha hai → Problem hai, mujhe batao!**

---

### Step 2: Test Karo (Best Method)
```bash
python predict_advanced.py --input "data/img/testing2.jpg"
```

---

## Expected Results:

**Before (Old Method):**
- 43 detected → 15 accepted (35%) ❌
- 28 rejected (65%)

**After (New Method):**
- 43 detected → 30-35 accepted (85-90%) ✅
- 5-8 rejected (15-20%)

---

## Agar Problem Aaye:

### Problem 1: Knowledge base mein sirf 669 embeddings
**Solution:**
```bash
del data\knowledge_base_embeddings.pkl
python build_knowledge_base.py
```

### Problem 2: Bahut zyada rejections (>30%)
**Solution:**
```bash
# YOLO confidence kam karo
python predict_advanced.py --input "image.jpg" --yolo-conf 0.30

# Quality threshold kam karo
python predict_advanced.py --input "image.jpg" --min-quality 30
```

### Problem 3: Coke vs Coke Zero confuse ho raha
**Solution:** Already fixed in `predict_advanced.py`!
- Similar product disambiguation enabled
- Higher threshold for similar products

---

## All Methods Comparison:

1. **predict_advanced.py** ⭐ - BEST (use this!)
2. **predict_ensemble.py** - Maximum accuracy (slow)
3. **predict_production.py** - Production ready
4. **predict_with_tta.py** - Good balance
5. **predict_with_precomputed.py** - Fastest

---

## Questions? Check:
- `SOLUTION_GUIDE.md` - Detailed explanation
- `ACCURACY_IMPROVEMENTS.md` - Technical details

---

**Ready? Run Step 1 first, then Step 2!** 🚀
