# Phase 3: Build the Intelligence Network

## 📌 Purpose
Expand beyond price data —  
add macroeconomic, news, and sentiment signals into the AI system  
for better regime switching, trend detection, and anomaly responses.

---

## 🎯 Major Deliverables
- Macro data ingestion (Fed rates, CPI, unemployment, etc.)
- News scraping / API integration (e.g., Finviz, NewsAPI)
- Reddit sentiment analysis (Pushshift.io or custom scraper)
- Merge multimodal features into model input pipeline

---

## 🛠️ Tools / Tech Required
- `requests` for API pulls
- `BeautifulSoup` for scraping (optional)
- Pushshift API, NewsAPI keys
- Sentiment analysis (VADER or HuggingFace transformers)

---

## 🗺️ Step-by-Step Plan
1. Build `/src/data_ingestion/macro_data.py`
2. Build `/src/data_ingestion/news_scraper.py`
3. Build `/src/data_ingestion/reddit_sentiment.py`
4. Engineer new features inside `/feature_engineering/`
5. Retrain model with multimodal input
6. Validate improvement in backtest

---

## ✅ Success Criteria
- 3+ external data sources ingested and logged daily
- Sentiment scores cleanly integrated into features
- Model performance improves >5% vs price-only models

---

## ⚠️ Risks & How to Handle
| Risk | Solution |
|:-----|:---------|
| API rate limits | Cache daily pulls, respect rate limits |
| Noisy sentiment signals | Smooth signals with moving averages or weighted scoring |
| Overcomplicating feature set | Perform feature importance analysis before final training |

---
> Phase 3 is complete when the system thinks beyond price — it "listens" to the market's external reality too.