# 🔥 Overfitting Fix V2 - AGGRESSIVE MODE

**Date**: October 14, 2025  
**Status**: ✅ **READY TO DEPLOY**  
**Version**: V2 - Aggressive Regularization

---

## 📊 Training History

### **V0 - Original (FAILED)**
```
Config:  LR=5e-06, broken scheduler, weight_decay=0.01
Result:  Train=0.369, Eval=5.577 (1409% gap)
Status:  🚨 SEVERE OVERFITTING
```

### **V1 - Initial Fixes (IMPROVED BUT INSUFFICIENT)**
```
Config:  LR=2e-06, fixed scheduler [2020,4040], weight_decay=0.01
Result:  Train=0.422, Eval=5.117 (1114% gap)
Status:  ⚠️ 8% improvement, still overfitting
```

### **V2 - Aggressive Mode (CURRENT)**
```
Config:  LR=1e-06, aggressive scheduler [1010,2020,3030], weight_decay=0.05
Target:  Train=~0.50, Eval=~2.5-3.5 (500-600% gap → acceptable for extended vocab)
Status:  🎯 Expected 35-50% improvement over V1
```

---

## 🔧 V2 Changes Applied

### **1. Learning Rate: 50% Reduction** 🎯
```python
# V1 (Insufficient):
lr=2e-06

# V2 (Aggressive):
lr=1e-06  # 50% reduction for much gentler learning
```

**Why**: Mel spectrograms are overfitting faster than text. Even 2e-06 is too aggressive for the 847 randomly initialized embeddings learning audio patterns.

**Expected Impact**: 
- Slower convergence but better generalization
- Mel loss should improve 30-40%
- Training loss will be slightly higher (~0.50 vs 0.42)

---

### **2. Weight Decay: 5x Increase** 🛡️
```python
# V1 (Insufficient):
weight_decay=0.01

# V2 (Aggressive):
weight_decay=0.05  # 5x increase for stronger L2 regularization
```

**Why**: Need stronger regularization on mel prediction layers to prevent memorization.

**Expected Impact**:
- Prevents large weight updates
- Forces model to find more general patterns
- Eval loss should improve 15-20%

---

### **3. LR Scheduler: More Aggressive Decay** 📉
```python
# V1 (2 reductions over 6 epochs):
milestones=[2020, 4040]  # Epochs 2, 4
gamma=0.5

# V2 (3 reductions over 4 epochs):
milestones=[1010, 2020, 3030]  # Epochs 1, 2, 3
gamma=0.5
```

**LR Schedule:**
| Epoch | V1 LR | V2 LR | Change |
|-------|-------|-------|--------|
| 0 | 2e-06 | **1e-06** | 50% lower |
| 1 | 2e-06 | **5e-07** | 75% lower |
| 2 | 1e-06 | **2.5e-07** | 75% lower |
| 3 | 1e-06 | **1.25e-07** | 87.5% lower |
| 4+ | 5e-07 | **1.25e-07** | 75% lower |

**Why**: Earlier and more frequent reductions prevent late-stage overfitting.

**Expected Impact**:
- Smoother convergence curve
- Less overfitting in later epochs
- Better checkpoint quality throughout training

---

## 📊 Expected Training Behavior

### **Epoch 0 (V2 Aggressive)**
```
Train Loss: 0.48-0.52  (slower start - expected)
Eval Loss:  3.0-3.8    (MUCH better than V1's 5.117)
Gap:        ~600%      (acceptable with extended vocab)
Status:     ✅ Healthy training baseline
```

### **Epoch 1 (LR → 5e-07)**
```
Train Loss: 0.44-0.48  (gradual improvement)
Eval Loss:  2.4-3.0    (20-30% improvement)
Gap:        ~500%      (improving)
Status:     ✅ Good progress
```

### **Epoch 2 (LR → 2.5e-07)**
```
Train Loss: 0.40-0.45  (steady improvement)
Eval Loss:  2.0-2.5    (15-25% improvement)
Gap:        ~400%      (much better)
Status:     ✅ Best checkpoint likely here
```

### **Epoch 3 (LR → 1.25e-07)**
```
Train Loss: 0.38-0.42  (slow refinement)
Eval Loss:  1.8-2.3    (5-15% improvement)
Gap:        ~350-400%  (optimal for extended vocab)
Status:     ✅ Alternative best checkpoint
```

### **Epoch 4+**
```
Train Loss: 0.36-0.40  (diminishing returns)
Eval Loss:  1.8-2.5    (may plateau or increase)
Gap:        Variable
Status:     ⚠️ Monitor closely, may need to stop
```

---

## 🎯 Success Criteria

### **✅ GOOD Training (Stop and Use)**
```
Epoch 2-3: eval_loss = 2.0-2.5
Gap:       400-500%
Status:    Ready for production
```

### **⚠️ ACCEPTABLE (Continue Monitoring)**
```
Epoch 2-3: eval_loss = 2.5-3.5
Gap:       500-700%
Status:    Workable, continue to epoch 4
```

### **🛑 STILL PROBLEMATIC (Further Tuning Needed)**
```
Epoch 2-3: eval_loss > 3.5
Gap:       >700%
Status:    Need even lower LR or investigate data quality
```

---

## 📈 Performance Expectations

### **Comparison Table**

| Metric | V0 (Failed) | V1 (Improved) | V2 (Aggressive) | Target |
|--------|------------|--------------|----------------|--------|
| **Train Loss** | 0.369 | 0.422 | **~0.48** | 0.40-0.50 |
| **Eval Loss (E0)** | 5.577 | 5.117 | **~3.2** | <3.5 |
| **Eval Loss (E2)** | N/A | N/A | **~2.2** | <2.5 |
| **Gap** | 1409% | 1114% | **~560%** | <600% |
| **Improvement** | Baseline | +8% | **+35-50%** | Target |

---

## 🛠️ Technical Details

### **Hyperparameters Summary**

| Parameter | V0 | V1 | V2 | Change |
|-----------|----|----|----|----|
| **Learning Rate** | 5e-06 | 2e-06 | **1e-06** | 80% reduction from V0 |
| **Weight Decay** | 0.01 | 0.01 | **0.05** | 5x increase |
| **LR Milestones** | [900K, 2.7M, 5.4M] | [2020, 4040] | **[1010, 2020, 3030]** | Earlier, more frequent |
| **Gradient Clip** | None | 1.0 | **1.0** | Maintained |
| **Workers** | 8 | 4 | **4** | Maintained |
| **Gamma** | 0.5 | 0.5 | **0.5** | Maintained |

---

## 🚀 Deployment Instructions

### **1. Stop Current Training**
```bash
# On Lightning AI terminal
Ctrl+C
```

### **2. Pull V2 Changes**
```bash
cd /teamspace/studios/this_studio/Amharic_XTTS-V2_TTS
git pull origin main
```

### **3. Verify Changes**
Check that training output shows:
```
🔥 AGGRESSIVE OVERFITTING PREVENTION - V2
 > Learning Rate: 1e-06 (REDUCED: 5e-06 → 2e-06 → 1e-06)
 > Weight Decay: 0.05 (INCREASED for stronger regularization)
```

### **4. Start Fresh Training**
- Same WebUI settings as before
- G2P enabled, rule_based backend
- Start training and monitor closely

### **5. Monitor Eval Loss**
| Checkpoint | Expected Eval Loss | Action |
|------------|-------------------|--------|
| Epoch 0 | 3.0-3.8 | ✅ Continue |
| Epoch 1 | 2.4-3.0 | ✅ Continue |
| Epoch 2 | 2.0-2.5 | ✅ Likely BEST |
| Epoch 3 | 1.8-2.3 | ✅ Alternative BEST |
| Epoch 4 | 1.8-2.5 | ⚠️ Watch for increase |

---

## 💡 Why These Changes Work

### **1. Lower Learning Rate**
- **Problem**: Mel features overfit faster than text
- **Solution**: 1e-06 LR slows mel learning, allows gradual pattern extraction
- **Evidence**: Text loss already good (0.079), mel needs gentle handling

### **2. Higher Weight Decay**
- **Problem**: Large weights in mel prediction layers memorize training data
- **Solution**: 0.05 weight decay strongly penalizes large weights
- **Evidence**: V1 had 0.01, standard for pre-training, but fine-tuning needs more

### **3. Aggressive LR Schedule**
- **Problem**: Constant high LR causes continued overfitting
- **Solution**: Reduce LR every epoch to progressively refine without overfitting
- **Evidence**: Early reduction at epoch 1 prevents late-stage problems

---

## 🎓 Learning from V0 → V1 → V2

### **V0 → V1 (8% improvement)**
**What worked**:
- Fixed broken scheduler ✅
- Reduced LR from 5e-06 to 2e-06 ✅
- Added gradient clipping ✅

**What didn't**:
- LR still too high for mel learning ❌
- Weight decay insufficient for extended vocab ❌
- Scheduler not aggressive enough ❌

### **V1 → V2 (Expected 35-50% improvement)**
**What changed**:
- Halved LR again (2e-06 → 1e-06) ✅
- 5x weight decay increase (0.01 → 0.05) ✅
- Earlier LR reductions (epoch 1 vs epoch 2) ✅

**Why this should work**:
- Addresses mel-specific overfitting directly
- Much stronger regularization throughout training
- Progressive learning rate reduction prevents plateaus

---

## 📋 Quick Decision Guide

### **After Epoch 0:**
```
Eval < 3.5:  ✅ Excellent! Continue
Eval 3.5-4.0: ✅ Good! Continue  
Eval 4.0-4.5: ⚠️ Acceptable, monitor closely
Eval > 4.5:  🛑 Consider reducing LR to 5e-07
```

### **After Epoch 1:**
```
Eval < 2.8:  ✅ Excellent! Continue to find best
Eval 2.8-3.2: ✅ Good progress, continue
Eval 3.2-3.8: ⚠️ Slow improvement, continue to epoch 2
Eval > 3.8:  🛑 Not improving, investigate data
```

### **After Epoch 2:**
```
Eval < 2.5:  ✅ SUCCESS! This is your checkpoint
Eval 2.5-3.0: ✅ Very good! Continue to epoch 3
Eval 3.0-3.5: ⚠️ Usable but not ideal
Eval > 3.5:  🛑 Stop, use epoch 1 checkpoint
```

---

## 🎯 Final Notes

### **Why Extended Vocab Needs Special Treatment**
```
Base vocab:      6,681 tokens (pre-trained, stable)
Extended vocab:  7,528 tokens (847 NEW, random init)

Problem: New embeddings start random, need gentle training
Solution: Very low LR + high weight decay + aggressive schedule
```

### **Expected Timeline**
```
Epoch 0: ~11-12 minutes (1010 steps)
Epoch 1: ~11-12 minutes
Epoch 2: ~11-12 minutes
Epoch 3: ~11-12 minutes
Total:   ~45-50 minutes for 4 epochs
```

### **Checkpoint Strategy**
```
Save:    Every 1000 steps (every epoch)
Monitor: Eval loss after each epoch  
Best:    Likely epoch 2 or 3
Use:     Lowest eval loss checkpoint
```

---

## ✅ Pre-Flight Checklist

Before starting V2 training:

- [ ] Changes committed and pushed to GitHub
- [ ] Changes pulled on Lightning AI
- [ ] Current training stopped
- [ ] Verified V2 banner in training output
- [ ] `TRAINING_MONITOR_QUICK_REFERENCE.md` open
- [ ] Understanding target eval loss < 3.5 after epoch 0
- [ ] Ready to stop if eval > 4.0 after 2 epochs

---

## 🎉 Expected Outcome

**V2 should achieve**:
- ✅ Eval loss ~2.0-2.5 by epoch 2
- ✅ 55-60% reduction from V1 (5.117 → 2.0-2.5)
- ✅ Usable model for Amharic TTS
- ✅ Clear improvement trajectory

**If V2 still shows eval > 3.5 after epoch 2**:
→ Data quality investigation needed (validation set issues)

---

**V2 represents aggressive but well-calibrated hyperparameters specifically for extended vocabulary fine-tuning with 847 new embeddings. This should resolve the overfitting!** 🚀
