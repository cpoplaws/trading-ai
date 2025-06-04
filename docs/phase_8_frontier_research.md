# Phase 8: Frontier Research (Quantum ML + Federated Learning)

## 📌 Purpose
Move beyond classical AI — explore and prototype the next generation of trading technologies:
- Quantum Machine Learning (QML) for portfolio optimization
- Federated Learning (FL) for privacy-preserving distributed training

---

## 🎯 Major Deliverables
- Quantum optimization prototype (e.g., portfolio balancing, hedging strategies)
- Federated training demo across simulated distributed nodes
- Research documentation inside `/research/`

---

## 🛠️ Tools / Tech Required
- Qiskit (IBM Quantum SDK)
- PennyLane (Quantum ML library)
- TensorFlow Federated
- Simulated data nodes for FL

---

## 🗺️ Step-by-Step Plan
1. Create `/research/quantum_ml/quantum_portfolio.py`
2. Build and simulate basic quantum optimization on Qiskit
3. Create `/research/federated_learning/federated_trainer.py`
4. Simulate federated model training across 3+ virtual nodes
5. Log results, limitations, and next steps in `/docs/`

---

## ✅ Success Criteria
- Functional quantum optimization example built
- Functional federated learning simulation completed
- Clear documentation of potential vs limitations

---

## ⚠️ Risks & How to Handle
| Risk | Solution |
|:-----|:---------|
| Quantum hardware limitations | Focus on quantum simulation first |
| Federated learning complexity | Start with very small federated datasets |
| Overfocusing on bleeding-edge | Timebox research to avoid slowing Phase 1–6 production builds |

---
> Phase 8 is complete when you have *real*, *working* prototypes in Quantum and Federated ML — and a path forward if/when these become production-ready.