# Smart Grid Forecaster 🔋

A full-stack web application for electricity demand forecasting and decision support. Train models, compare their performance, visualize predictions, and simulate demand scenarios in real-time.

**Try it live:** [https://frontend-lkq5h1yek-sidhantchakus-projects.vercel.app](https://frontend-lkq5h1yek-sidhantchakus-projects.vercel.app)

**Check backend status:** [https://backend-1vql9pg8p-sidhantchakus-projects.vercel.app/status](https://backend-1vql9pg8p-sidhantchakus-projects.vercel.app/status)

![Smart Grid Forecaster screenshot](docs/smart-grid-forecaster.png)

---

## What It Does

This app helps you forecast electricity grid load and make data-driven decisions:

- **Train forecasting models** – Six lightweight models you can train on your data
- **Compare performance** – Side-by-side metrics (MAE, RMSE, R²) to see which model performs best
- **Visualize predictions** – Charts showing actual vs. predicted values for the hybrid model
- **Next-value forecasting** – Feed in a 96-point sequence, get the predicted next load
- **Simulate scenarios** – Scale demand up or down to see how forecasts change
- **Smart recommendations** – Automatically suggests the best model based on R² score
- **Threshold alerts** – Set a threshold and get notified when forecast crosses it

---

## Why This Exists

The original concept was built around TensorFlow/Keras for production forecasting. But for the Vercel-hosted version, I switched to smaller NumPy-based approximations to stay within serverless constraints while keeping the full end-to-end workflow intact. The architecture is intentionally modular, so you can easily swap in heavier models later.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Frontend | React, Vite, Chart.js, Axios |
| Backend | FastAPI, Uvicorn |
| Forecasting | Lightweight NumPy-based models |
| Hosting | Vercel (serverless) |

---

## Project Structure

```
Smart-Grid-Forecaster/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── app.py               # Core forecasting logic
│   ├── models/              # Model implementations
│   ├── routes/              # API endpoints
│   ├── utils/               # Helper functions
│   └── saved_models/        # Pre-trained model files
│
├── frontend/
│   ├── src/                 # React components
│   ├── package.json
│   └── vite.config.js
│
├── docs/
│   └── smart-grid-forecaster.png
│
├── requirements.txt
├── vercel.json
└── README.md
```

---

## Running Locally

### Start the backend

```bash
cd Smart-Grid-Forecaster
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --port 8000
```

The backend will be available at `http://127.0.0.1:8000`.

### Start the frontend

In a new terminal:

```bash
cd Smart-Grid-Forecaster/frontend
npm install
npm run dev
```

Open your browser to `http://localhost:5173` and start using the app.

### Frontend configuration

By default, the frontend tries to reach the backend at `http://127.0.0.1:8000`. If you want to change this, create a file `frontend/.env`:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## API Endpoints

All endpoints are documented below. The backend provides a RESTful API for the frontend and any external tools.

| HTTP Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Returns current training state and model readiness |
| `POST` | `/train` | Trains all forecasting models on provided data |
| `GET` | `/metrics` | Returns MAE, RMSE, and R² for each trained model |
| `GET` | `/predictions` | Returns actual vs predicted values for the hybrid model |
| `POST` | `/predict` | Predicts the next value from a 96-point sequence |
| `POST` | `/simulate` | Compares base forecast with scaled demand scenario |
| `GET` | `/recommendation` | Returns the best-performing model (by R²) |
| `POST` | `/alert` | Checks if forecast exceeds a user-defined threshold |

---

## How the Models Work

The backend includes six lightweight forecasting approaches:

1. **Moving Average** – Simple baseline
2. **Exponential Smoothing** – Weights recent values more heavily
3. **Linear Regression** – Fits a trend line
4. **Random Forest** – Ensemble of decision trees
5. **Hybrid Model** – Weighted combination of the above

Each model is trained independently, then compared. The hybrid model typically performs best because it combines strengths of multiple approaches.

---

## Workflow

A typical session looks like:

1. **Upload or generate data** – A time series of electricity load values
2. **Train models** – Call `/train` to train all six models
3. **Review metrics** – Call `/metrics` to see MAE, RMSE, R² for each
4. **Check recommendation** – Call `/recommendation` to see which model Wins
5. **Make predictions** – Call `/predict` with a 96-value sequence
6. **Simulate scenarios** – Call `/simulate` to see what-if analyses
7. **Set alerts** – Call `/alert` to check forecasts against thresholds

---

## Deployment to Vercel

The repo is production-ready for Vercel:

### 1. Push to GitHub

Ensure your repo contains:
- `api/index.py` – Vercel Python function that runs FastAPI
- `public/index.html` – Static frontend files
- `vercel.json` – Vercel configuration
- `requirements.txt` – Python dependencies

### 2. Import into Vercel

- Go to [vercel.com](https://vercel.com)
- Create a new project from your GitHub repo
- Keep the project root as the repository root

### 3. Add environment variables (if needed)

Most features work out-of-the-box. If you customize the backend, add any required env vars in Vercel project settings.

### 4. Deploy

Push to `main` and Vercel automatically deploys. Visit your unique URL to see the live app.

---

## Design Philosophy

- **Modular architecture** – Easy to add or swap models without touching the UI
- **Lightweight models** – No heavy dependencies; works on serverless platforms
- **Real-time feedback** – See predictions instantly without long training waits
- **Visual clarity** – Charts make trends and errors obvious at a glance
- **Future-proof** – The codebase is structured so you can plug in TensorFlow/Keras models later if you move to a persistent backend

---

## Performance Notes

- Model training is fast (< 1 second) due to lightweight approximations
- Predictions are instant
- Frontend updates are responsive thanks to Vite
- Charts re-render smoothly even with large sequences

For production with millions of load forecasts, consider:
- A persistent backend server instead of serverless
- Heavier ML models (LSTM, Prophet, etc.)
- Caching and database layer
- Distributed training if data grows

---

## Next Steps

- Extend with ARIMA or SARIMA models
- Add upload functionality for CSV data
- Implement confidence intervals on predictions
- Build a dashboard for historical performance
- Add real grid data connectors
- Create alert webhooks (email, Slack)

---

## Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React + Vite](https://vitejs.dev/)
- [Chart.js](https://www.chartjs.org/)
- Time series forecasting best practices
