# Smart Grid Forecast (SGF)
Web App Link - [Click Here](https://frontend-lkq5h1yek-sidhantchakus-projects.vercel.app/)
A hybrid deep learning system for electricity load forecasting, built as a full-stack web application. Train six time-series models in your browser, compare their accuracy on the ETTh1 benchmark, and get single-step predictions from a 96-point input sequence — no Python knowledge required to use it.

The core model fuses two architectures that capture complementary temporal structure: **xLSTMTime** (recurrent + temporal convolution + multi-head attention) handles fine-grained step-to-step dynamics, while **PatchMixer** (patch-based MLP-Mixer) reads broad daily-cycle patterns. A learned fusion layer combines both. Pre-processing uses **Federated Normalisation (FedNorm)**, which normalises training shards independently to reduce distribution shift between training and test periods — a single preprocessing trick that adds ~0.012 R² for free.

On the ETTh1 test set, the hybrid achieves **R² = 0.9094**, outperforming five baselines including LSTM + Attention, N-BEATS, and Prophet.

---

## What's inside

```
Smart-Grid-Forecaster/
├── backend/
│   ├── main.py              # FastAPI app — endpoints, CORS, background training thread
│   ├── trainer.py           # Training orchestration for all six models
│   ├── models/
│   │   ├── hybrid_model.py  # xLSTMTime + PatchMixer fusion
│   │   ├── xlstmtime_model.py
│   │   ├── patchmixer_model.py
│   │   ├── lstm_model.py    # LSTM + Bahdanau attention baseline
│   │   └── other_models.py  # N-BEATS and Prophet
│   └── utils/
│       ├── preprocessing.py # FedNorm, MinMaxScaler, sliding window
│       └── metrics.py       # MAE, MSE, RMSE, R²
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   └── components/      # TrainPanel, MetricsTable, ForecastChart, ModelCompareChart
│   ├── package.json
│   └── vercel.json
├── requirements.txt
├── render.yaml              # One-click Render deployment config
└── README.md
```

The backend and frontend are fully decoupled — they communicate only over HTTP JSON, so you can swap either side independently.

---

## Running locally

You'll need Python 3.10+ and Node.js 18+. No GPU required.

**Backend**

```bash
# From the repo root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn backend.main:app --reload --port 8000
```

The API will be live at `http://localhost:8000`. Visit `/docs` for the auto-generated Swagger UI.

**Frontend**

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. That's it — hit **Train All Models** and wait about 10–15 minutes on a standard CPU machine.

If you have your own dataset, upload a CSV with a `Date` column (ISO 8601) and a numeric `Load` column before training. The preprocessing pipeline will handle everything else.

---

## API reference

| Method | Endpoint | What it does |
|--------|----------|--------------|
| `GET` | `/status` | Returns current training state: `idle`, `running`, `done`, or `error` |
| `POST` | `/train` | Starts training all six models in a background thread; returns immediately |
| `GET` | `/metrics` | Returns MAE, MSE, RMSE, and R² for every trained model |
| `GET` | `/predictions` | Returns actual vs. predicted arrays for the hybrid model's test set |
| `POST` | `/predict` | Accepts `{"sequence": [float × 96], "model": "Hybrid"}`, returns `{"prediction": float}` |

Poll `/status` every few seconds while training is running — the frontend does this automatically every 3 seconds.

---

## Models

Six architectures are trained and compared in a single run:

| Model | Description | ETTh1 R² |
|-------|-------------|-----------|
| **SGF Hybrid** | xLSTMTime + PatchMixer + FedNorm | **0.9094** |
| xLSTMTime | Stacked LSTM + Conv1D + Multi-head attention | 0.8814 |
| LSTM + Attention | Two-layer LSTM with Bahdanau attention | 0.8738 |
| PatchMixer | Patch-based MLP-Mixer | 0.7209 |
| N-BEATS | Neural basis expansion, 4 blocks | 0.7239 |
| Prophet | Additive decomposition (Meta) | 0.6561 |

All neural models share the same training config: Adam `lr=1e-3`, MSE loss, batch size 32, max 50 epochs, early stopping with patience 10. Random seed is fixed at 42 for reproducibility.

---

## Dataset

The project uses **ETTh1** (Electricity Transformer Temperature, hourly), a public benchmark from Zhou et al. (Informer, AAAI 2021). It contains 17,420 hourly readings from a Chinese grid substation across 20 months (July 2016 – June 2018), with oil temperature (OT) as the forecast target.

The preprocessing pipeline applies FedNorm → MinMaxScaler (fit on training split only) → sliding window of length 96. The train/val/test split is 70/10/20 in temporal order — no shuffling, which matters for a fair evaluation.

If the dataset file isn't found at startup, the backend falls back to a synthetic load series so you can still explore the interface.

---

## Deploying to Render

The included `render.yaml` deploys both services in one step:

1. Push the repo to GitHub.
2. In Render, click **New → Blueprint** and connect the repository.
3. Render reads `render.yaml` and sets up both services automatically.

The two services that get created are `smart-grid-forecasting-api` (FastAPI) and `smart-grid-forecasting-ui` (Vite static). Both run on Render's free tier.

If Render assigns different service names from what's in the config, update two environment variables in the Render dashboard:
- On the backend: `ALLOWED_ORIGINS`
- On the frontend: `VITE_API_BASE_URL`

---

## Deploying the frontend to Vercel

If you'd rather host the frontend on Vercel (faster cold starts for static sites) and the backend on Render:

**Vercel settings:**
- Root directory: `frontend`
- Framework preset: `Vite`
- Build command: `npm install && npm run build`
- Output directory: `dist`
- Environment variable: `VITE_API_BASE_URL=https://your-render-backend-url`

The included `frontend/vercel.json` handles client-side routing so page refreshes don't 404.

For local development with this setup, create `frontend/.env`:

```
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## How FedNorm works

Standard preprocessing fits a single MinMaxScaler on the entire training set. The problem is that when training and test periods cover different seasons or operating conditions, the scaler encodes the training distribution — and the model never sees what "out-of-distribution scale" looks like.

FedNorm splits the training array into four equal shards and normalises each shard to zero mean and unit variance independently, then recombines them before passing to MinMaxScaler. The model trains on examples with four slightly different normalised distributions instead of one, which makes its representations more scale-invariant. The ablation in the thesis confirms this adds ~0.012 R² consistently across runs — not huge, but entirely free at inference time.

```python
def fednorm(X: np.ndarray, n_clients: int = 4) -> np.ndarray:
    n, chunk = len(X), len(X) // n_clients
    parts = []
    for i in range(n_clients):
        start = i * chunk
        end = (i + 1) * chunk if i < n_clients - 1 else n
        part = X[start:end]
        parts.append((part - part.mean()) / (part.std() + 1e-8))
    return np.concatenate(parts, axis=0)
```

---

## Tech stack

- **Backend:** FastAPI, TensorFlow/Keras, Prophet (Meta), scikit-learn, NumPy, Pandas, Uvicorn
- **Frontend:** React, Vite, Chart.js (via react-chartjs-2), Axios
- **Deployment:** Render (backend + frontend), Vercel (frontend alternative)

---

## Acknowledgements

The ETTh1 dataset was released by Zhou et al. alongside the [Informer paper](https://arxiv.org/abs/2012.07436) (AAAI 2021) and has since become a standard benchmark for time-series forecasting. FedNorm is inspired by [FedBN](https://arxiv.org/abs/2102.07623) (Li et al., ICLR 2021). The PatchMixer branch draws on ideas from [PatchTST](https://arxiv.org/abs/2211.14730) (Nie et al., ICLR 2023) and [MLP-Mixer](https://arxiv.org/abs/2105.01601) (Tolstikhin et al., NeurIPS 2021).

---

## License

MIT — do whatever you want with it.
