# Phase 4: AI Model Power-Up

## 📌 Purpose
Replace basic models (Random Forests) with state-of-the-art  
deep learning models for time-series financial prediction:
- Transformers (TimesNet, Informer, Autoformer)
- Ensemble models

---

## 🎯 Major Deliverables
- Integrate PyTorch modeling module
- Deploy TimesNet or Autoformer models
- Ensemble multiple model predictions (weighted voting or stacking)
- Hyperparameter tuning (GridSearch or Optuna)

---

## 🛠️ Tools / Tech Required
- PyTorch
- TimesNet, Autoformer (open-source models)
- Optuna for tuning (optional)
- Ray Tune (optional for future scaling)

---

## 🗺️ Step-by-Step Plan
1. Create `/src/modeling/deep_models.py`
2. Train TimesNet or Autoformer on existing features
3. Create `/src/modeling/ensemble.py`
4. Blend Random Forest + Transformer outputs
5. Compare ensemble vs single model performance
6. Deploy best-performing ensemble into production

---

## ✅ Success Criteria
- Transformer model trained and evaluated
- Ensemble model outperforms single Random Forest baseline
- Models versioned and tracked in `/models/`

---

## ⚠️ Risks & How to Handle
| Risk | Solution |
|:-----|:---------|
| Overfitting deep models | Regularize, early stopping, dropout layers |
| Slow training | Use GPU runtime if available |
| Managing multiple model versions | Create simple model registry inside `/models/`

---
> Phase 4 is complete when your models are smarter, faster, and more accurate — a true AI trading brain is born.