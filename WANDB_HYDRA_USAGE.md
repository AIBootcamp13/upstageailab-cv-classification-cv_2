# 🧪 WandB + Hydra Integration Test Commands

# ========================================
# 1. Quick Debug Test (3 epochs only)
# ========================================
python scripts/train.py experiment=quick_debug

# Expected output:
# 🚀 WandB initialized: debug-quick-test-resnet50-basic-b16-s224-(f1_pending)
# 📊 Mode: online, Project: document-classifier
# 🎯 Experiment: debug-quick-test
# 🎨 Augmentation: basic (intensity: 0.5)

# ========================================
# 2. Production Robust Test
# ========================================
python scripts/train.py experiment=production_robust

# Expected output:
# 🚀 WandB initialized: production-robust-v1-resnet50-robust-b32-s224-(f1_pending)
# 📊 Mode: online, Project: document-classifier  
# 🎯 Experiment: production-robust-v1
# 🎨 Augmentation: robust (intensity: 0.85)

# ========================================
# 3. Manual Overrides Test
# ========================================
python scripts/train.py experiment=quick_debug \
    train.batch_size=8 \
    data.augmentation.intensity=0.9 \
    wandb.tags='["manual-test","debug","override"]'

# ========================================
# 4. Offline Mode Test (no internet required)
# ========================================
WANDB_MODE=offline python scripts/train.py experiment=quick_debug

# ========================================
# 5. Disable WandB Test
# ========================================
python scripts/train.py experiment=quick_debug wandb.enabled=false

# Expected output:
# 🚫 WandB logging disabled - Check config.yaml wandb.enabled setting

# ========================================
# 6. EfficientNet Baseline Test  
# ========================================
python scripts/train.py experiment=efficientnet_baseline

# ========================================
# 7. Extreme Robust Test (for challenging test data)
# ========================================
python scripts/train.py experiment=resnet_robust

# Expected output:
# 🚀 WandB initialized: resnet50-robust-extreme-resnet50-robust-b24-s224-(f1_pending)
# 📊 Mode: online, Project: document-classifier
# 🎯 Experiment: resnet50-robust-extreme  
# 🎨 Augmentation: robust (intensity: 0.95)

# ========================================
# 8. Check WandB Dashboard
# ========================================
# After running, check: https://wandb.ai/[your-username]/document-classifier
# Look for:
# - ✅ Run names with experiment info
# - ✅ Real-time loss/accuracy charts  
# - ✅ Augmentation parameters logged
# - ✅ Model artifacts (if enabled)
# - ✅ Sample prediction images
# - ✅ Confusion matrices