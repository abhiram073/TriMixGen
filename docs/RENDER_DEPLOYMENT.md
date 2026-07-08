# Render Deployment Guide

## 🚨 Memory Error Troubleshooting

### **Problem:**
```
===> Out of memory (used over 512Mi)
```

The free Render tier provides only **512MB RAM**, but your application requires:
- PyTorch: ~1.5GB
- Transformers: ~500MB
- Other dependencies: ~500MB
- **Total: ~2.5GB minimum**

---

## ✅ Solution: Use DEMO_MODE (Recommended for Free Tier)

The application includes a lightweight `DemoGenerator` that doesn't require PyTorch or large model weights.

### **What Changed:**
1. ✅ Created `requirements-prod.txt` with only essential dependencies (~50MB)
2. ✅ Updated `render.yaml` to use production requirements
3. ✅ Set `DEMO_MODE=true` environment variable
4. ✅ Made torch import conditional in `config.py`

### **Deploy with Free Tier:**

1. **Commit changes:**
   ```bash
   git add requirements-prod.txt render.yaml backend/config.py
   git commit -m "chore: optimize for free-tier Render deployment"
   git push
   ```

2. **In Render Dashboard:**
   - Update the build command to: `pip install -r requirements-prod.txt`
   - Add environment variable: `DEMO_MODE=true`
   - Redeploy

3. **Expected behavior:**
   - ✅ Full LID tagging works
   - ✅ CMI calculation works
   - ✅ API responds normally
   - ⚠️ Generation uses heuristic rules (DEMO_MODE), not ML models

---

## 💰 Alternative: Use Paid Tier for Full ML Features

To use actual mT5 + LoRA models in production:

1. **Upgrade Render plan to "Standard"** (2GB RAM - ~$10/month)
   - Edit `render.yaml`: `plan: standard`
   - Set `DEMO_MODE=false`
   - Use regular `requirements.txt`

2. **Or use GitHub's free tier (GitHub Actions + free GPU runners)**

---

## 🔍 Verification

After deployment, test with:
```bash
curl https://your-render-app.onrender.com/api/v1/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello world", "style": "tech"}'
```

---

## 📊 Comparison Table

| Aspect | Free Tier + DEMO | Paid Tier + ML |
|--------|-------------------|-----------------|
| RAM | 512MB ✅ | 2GB+ ✅ |
| Generation Quality | Heuristic (good) | ML Models (excellent) |
| Cost | Free | $10-50/mo |
| Deployment Time | ~2min | ~5min |
| Startup Time | <5s | 30-60s |
| LID Tagging | ✅ Works | ✅ Works |
| CMI Calculation | ✅ Works | ✅ Works |

