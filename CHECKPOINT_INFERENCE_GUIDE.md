# 🎙️ Direct Checkpoint Inference - Quick Guide

## What's New

You can now **test any checkpoint directly** without copying or loading it first! This is perfect for:

- **A/B Testing**: Compare different checkpoints quickly
- **Quality Check**: Test before committing to a checkpoint
- **Experimentation**: Try different training stages

---

## 🚀 Quick Usage

### Step 1: Scan Checkpoints
```
1. Go to Inference tab
2. Scroll to "🔄 Checkpoint Selection (Advanced)"
3. Click "🔍 Scan Checkpoints"
```

### Step 2: Select a Checkpoint
```
Choose any checkpoint from the dropdown
```

### Step 3: Enter Test Text
```
Scroll down to "🎤 Quick Inference Test"
Enter your test text (Amharic or any language)
```

### Step 4: Generate!
```
Click "🎙️ Test Selected Checkpoint"
→ Wait a few seconds
→ Listen to the output!
```

---

## 🎯 Use Cases

### 1. Compare Checkpoints

**Goal**: Find the best sounding checkpoint

**Steps**:
```
1. Scan checkpoints
2. Select checkpoint_3000.pth (Epoch 5)
3. Enter test text: "ሰላም ዓለም"
4. Click "Test Selected Checkpoint"
5. Listen to output
6. Select checkpoint_6000.pth (Epoch 10)
7. Click "Test Selected Checkpoint" again
8. Compare the two outputs!
```

### 2. Quick Quality Check

**Goal**: Verify if early checkpoint sounds better

**Steps**:
```
1. Select best_model_569.pth (Epoch 0) ← Recommended
2. Enter Amharic test sentence
3. Enable "🇪🇹 Use G2P (Amharic)"
4. Test it!
5. If good → Click "✅ Use Selected Checkpoint" to make it permanent
6. If not good → Try another checkpoint
```

### 3. A/B Testing Multiple Checkpoints

**Goal**: Test 3-4 checkpoints with same text

**Workflow**:
```
Keep the same test text
Change checkpoint in dropdown
Click "Test" for each
Download outputs and compare side-by-side
```

---

## ⚙️ Settings

### Language
- Select language from dropdown
- Use `am` or `amh` for Amharic

### G2P (Amharic)
- ✅ **Enable** if your checkpoint was trained with G2P
- ❌ **Disable** if trained without G2P

**How to know?**
- If you trained with "Enable G2P for Training" checked → Enable G2P here
- If unsure → Try both and see which sounds better!

---

## 💡 Tips

### 1. Use Same Test Text
When comparing checkpoints, use the **same test text** for fair comparison:
```
Good test: "ሰላም ዓለም። እንዴት ነህ？"
(Has various phonemes, punctuation, question mark)
```

### 2. Test on New Text
Use text that **wasn't in your training data** to test generalization:
```
✅ Good: New sentences you wrote
❌ Bad: Exact sentences from training dataset
```

### 3. Load Parameters First
Before testing, make sure to:
```
1. Click "📂 Load from Output Folder" (at top of Inference tab)
2. This loads vocab, config, and speaker reference
3. Now you can test any checkpoint!
```

### 4. Don't Overload GPU
Testing loads a full model temporarily:
```
- Each test takes 5-10 seconds
- GPU memory is released after each test
- Wait for one test to finish before starting another
```

---

## 🔄 Workflow Example

### Your Amharic Case

```
GOAL: Find the best checkpoint for Amharic TTS

STEP 1: Scan
  Click "🔍 Scan Checkpoints"
  → See: best_model_569.pth, checkpoint_3000.pth, checkpoint_6000.pth, ...

STEP 2: Prepare Test
  Test Text: "ሰላም። ዛሬ አንድ ጥሩ ቀን ነው።"
  Language: am (or amh)
  G2P: ✅ Enabled

STEP 3: Test Each Checkpoint
  
  Test 1: best_model_569.pth (Epoch 0, Loss: 3.415)
    Click "Test" → Listen → Rate quality: 8/10
  
  Test 2: checkpoint_3000.pth (Epoch 5, Loss: 5.678)
    Click "Test" → Listen → Rate quality: 6/10
  
  Test 3: checkpoint_6000.pth (Epoch 10, Loss: 6.419)
    Click "Test" → Listen → Rate quality: 4/10

STEP 4: Choose Winner
  best_model_569.pth sounds BEST!
  
STEP 5: Make it Permanent
  Select best_model_569.pth
  Click "✅ Use Selected Checkpoint"
  Click "▶️ Step 3 - Load Model"
  
DONE! Now all your generations use the best checkpoint!
```

---

## 📊 Comparison Table

| Feature | Direct Test | Load & Use |
|---------|-------------|------------|
| Speed | 5-10 seconds | ~3 seconds (after loading) |
| Use Case | Compare multiple | Production use |
| GPU Memory | Temporary | Persistent |
| Best For | Testing | Final model |

**Recommendation**: Use **Direct Test** for comparison, then **Load & Use** your favorite!

---

## 🐛 Troubleshooting

### "No speaker reference found"

**Cause**: Haven't loaded model parameters yet

**Fix**:
```
1. Go to top of Inference tab
2. Click "📂 Load from Output Folder"
3. Wait for success message
4. Now try testing checkpoint again
```

### Test takes very long

**Cause**: Loading large checkpoint

**Solution**: Be patient! First test is slow (loading model), subsequent tests are faster.

### Audio output is silent/garbled

**Possible causes**:
1. Checkpoint is heavily overfitted
2. G2P setting doesn't match training
3. Language mismatch

**Fix**:
```
- Try different checkpoint (earlier epoch)
- Toggle G2P on/off
- Check language setting
```

### "Error testing checkpoint"

**Common causes**:
- Corrupted checkpoint file
- Vocab/config mismatch
- Out of GPU memory

**Fix**:
```
1. Check console for detailed error
2. Try a different checkpoint
3. Restart WebUI if needed
```

---

## 🎓 Best Practices

### 1. Always Test Before Committing
```
Don't just trust the eval loss numbers!
Listen to the audio first.
Sometimes a checkpoint with higher loss sounds better.
```

### 2. Use Diverse Test Sentences
```
Test with:
- Short sentences (5-10 words)
- Long sentences (20+ words)
- Questions
- Statements
- Various phonemes
```

### 3. Keep Notes
```
Create a simple table:

Checkpoint         | Epoch | Loss  | Quality | Notes
-------------------|-------|-------|---------|------------------
best_model_569     | 0     | 3.415 | 8/10    | Natural, clear
checkpoint_3000    | 5     | 5.678 | 6/10    | OK but robotic
checkpoint_6000    | 10    | 6.419 | 4/10    | Distorted
```

### 4. Test Multiple Times
```
Same checkpoint can sound different with different text.
Test each checkpoint with 2-3 different sentences.
```

---

## 🎉 Benefits

✅ **Fast Comparison**: Test multiple checkpoints in minutes  
✅ **No Commitment**: Test without changing your loaded model  
✅ **Quality Focused**: Make decisions based on actual audio, not just numbers  
✅ **Iterative**: Quickly find the sweet spot in your training  

---

## 🔗 Related Features

- **Checkpoint Scanning**: Lists all available checkpoints
- **Overfitting Analysis**: Shows which checkpoints likely sound best
- **Use Selected Checkpoint**: Makes your choice permanent
- **Regular Inference**: Generate with currently loaded model

---

## 📚 Documentation

- **Checkpoint Selection Guide**: `CHECKPOINT_SELECTION_GUIDE.md`
- **Quick Start**: `CHECKPOINT_QUICK_START.md`
- **Training Diagnosis**: `TRAINING_DIAGNOSIS_AND_FIX.md`

---

## ✨ Example Session

```
🎯 Goal: Find best Amharic checkpoint

1. Start WebUI
2. Load parameters
3. Scan checkpoints (5 found)
4. Test each with "ሰላም ዓለም"

Results:
  Epoch 0:  ⭐⭐⭐⭐⭐ EXCELLENT
  Epoch 5:  ⭐⭐⭐ Good
  Epoch 10: ⭐⭐ OK
  Epoch 50: ⭐ Poor
  Epoch 91: 💀 Terrible

Winner: Epoch 0 (best_model_569.pth)

5. Use it permanently
6. Generate production audio

Total time: 5 minutes
Result: Perfect Amharic TTS! 🇪🇹✅
```

---

**Feature Added**: October 14, 2025  
**Status**: ✅ **READY TO USE**  
**Next Step**: Try it now in the Inference tab!

Enjoy comparing your checkpoints! 🎙️🎤
