# Quick Start: Training Amharic XTTS with G2P

## TL;DR - Just Do This!

1. **SSH into your Lightning.ai instance**
2. **Pull the latest code:**
   ```bash
   cd ~/Amharic_XTTS-V2_TTS
   git pull origin main
   ```

3. **Access your Gradio WebUI** (the share URL from your running instance)

4. **In the Fine-tuning tab:**
   - Set your training parameters (epochs, batch size, etc.)
   - ✅ **CHECK "Enable G2P for Training"** ← THIS IS THE MAGIC CHECKBOX
   - Click "Step 2 - Train Model"

5. **Watch it work!** 🎉

That's it! The system will:
- Automatically detect if your dataset needs conversion
- Convert Amharic text to phonemes if needed  
- Train successfully without UNK token errors

---

## What Changed?

### Before (Broken)
- Checkbox did nothing
- Training failed with: `AssertionError: assert not torch.any(tokens == 1)`
- Had to manually preprocess everything

### After (Fixed)
- Checkbox actually works!
- Automatic preprocessing on-the-fly
- Training succeeds 

---

## Two Options for Training

### Option 1: Use the Checkbox (Easiest)
Just check "Enable G2P for Training" and let the system handle everything automatically.

**Pros:**
- One click solution
- Works with any dataset state
- Automatic detection

**Cons:**
- Slightly slower first time (does conversion during training load)

### Option 2: Preprocess First (Fastest for repeated training)
```bash
cd ~/Amharic_XTTS-V2_TTS
python3 preprocess_quick.py
```

Then use the preprocessed CSV files and still check the G2P checkbox (it will detect and skip conversion).

**Pros:**
- Faster training start
- Can inspect preprocessed data
- Reuse preprocessed files

**Cons:**
- Extra step
- Need to update CSV paths in WebUI

---

## Troubleshooting in 3 Steps

### 1. Still getting UNK token errors?

Check the logs for:
```
> Amharic G2P mode ENABLED
> Dataset will be checked and converted if needed
```

If you don't see this, **the checkbox isn't working** - report as bug.

### 2. See errors about missing dependencies?

```bash
pip install -r requirements.txt
```

### 3. Training takes forever to start?

First time will be slower (G2P conversion). Subsequent times will be fast if you:
- Use preprocessed CSVs, OR
- Keep using the same dataset

---

## What's Happening Behind the Scenes?

When you check "Enable G2P for Training":

1. **Detection Phase** (< 1 second)
   - Samples your CSV files
   - Checks if text is Amharic script or phonemes

2. **Conversion Phase** (if needed, ~2-5 seconds for small datasets)
   - Loads Amharic G2P tokenizer
   - Converts each Amharic text sample to IPA phonemes
   - Updates language code to 'en' for XTTS tokenizer

3. **Training Phase** (normal)
   - XTTS trains on phoneme representations
   - No UNK tokens!
   - Model learns audio→phoneme mappings

---

## FAQ

**Q: Do I always need to check the G2P checkbox?**
A: For Amharic datasets, YES. Otherwise training will fail with UNK tokens.

**Q: Can I train without preprocessing first?**
A: YES! That's the whole point of this fix. Just check the checkbox.

**Q: What if I already preprocessed my dataset?**
A: Still check the checkbox! It will detect preprocessing and skip conversion.

**Q: Will this slow down training?**
A: No. Conversion happens once during dataset loading, not during training.

**Q: Does this work for other languages?**
A: It's designed for Amharic, but the detection is smart - it won't break other languages.

---

## Success Indicators

Look for these in your training logs:

✅ **Amharic G2P mode ENABLED**
✅ **Dataset check: ... Status: Already preprocessed** OR **Status: Needs G2P conversion**
✅ **G2P preprocessing complete: ✓ Converted: X samples**
✅ **Language code updated: 'amh' → 'en'**
✅ **Training XTTS GPT Encoder** (starts without errors)

---

## Next Steps After Training

1. **Test your model** with Amharic input
2. **Create an inference wrapper** that auto-converts input text
3. **Fine-tune hyperparameters** if needed
4. **Share your results** with the community!

---

**For detailed technical documentation, see: `AMHARIC_G2P_TRAINING_INTEGRATION.md`**

**For offline preprocessing option, see: `AMHARIC_TRAINING_SOLUTION.md`**
