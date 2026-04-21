# Smart Grid Forecasting Dashboard

An end-to-end forecasting demo built with FastAPI, React, TensorFlow/Keras, and Chart.js. The app trains six lightweight time-series models, compares their metrics, visualizes hybrid predictions, and predicts the next value from a 96-point input sequence.

## Project Structure

```text
smart-grid-forecasting-dashboard/
  backend/
    main.py
    models/
    routes/
    utils/
    saved_models/
  frontend/
    src/
    package.json
  requirements.txt
  README.md
```

## Backend Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the API server:

```bash
uvicorn backend.main:app --reload --port 8000
```

The backend will automatically use `REV-2/ETTh1 final.csv` if it exists in the parent workspace. If not, it falls back to a synthetic load series.

## Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the Vite dev server:

```bash
npm run dev
```

Open the frontend at `http://localhost:5173`.

## Deployment Setup

This project is prepared for deployment on Render using the included [render.yaml](C:\Users\Admin\Desktop\CAPSTONE PROJ\smart-grid-forecasting-dashboard\render.yaml).

### What gets deployed

- `smart-grid-forecasting-api`: FastAPI backend
- `smart-grid-forecasting-ui`: Vite static frontend

### Before deploy

1. Push this folder to a GitHub repository.
2. In Render, choose `New +` -> `Blueprint`.
3. Connect the GitHub repo that contains this project.
4. Render will detect `render.yaml` and prepare both services.

### Important Render settings

After Render creates the services, confirm these URLs:

- Frontend URL: `https://smart-grid-forecasting-ui.onrender.com`
- Backend URL: `https://smart-grid-forecasting-api.onrender.com`

If Render assigns different names, update these environment variables in Render:

- Backend `ALLOWED_ORIGINS`
- Frontend `VITE_API_BASE_URL`

## Vercel Frontend Deployment

The React frontend is ready to deploy to Vercel from the [frontend](C:\Users\Admin\Desktop\CAPSTONE PROJ\smart-grid-forecasting-dashboard\frontend) directory.

### Recommended architecture

- Deploy frontend on Vercel
- Deploy backend on Render or Railway

This recommendation matters because the FastAPI backend currently:

- trains multiple TensorFlow models on demand
- writes `.keras` files and JSON artifacts to local disk
- relies on longer-running CPU work than Vercel serverless is ideal for

### Vercel settings

- Root Directory: `frontend`
- Framework Preset: `Vite`
- Build Command: `npm install && npm run build`
- Output Directory: `dist`
- Environment Variable: `VITE_API_BASE_URL=https://your-backend-url`

The included [frontend/vercel.json](C:\Users\Admin\Desktop\CAPSTONE PROJ\smart-grid-forecasting-dashboard\frontend\vercel.json) keeps client-side routes working after deployment.

### Local environment variable

For local frontend development, you can optionally create `frontend/.env` from [frontend/.env.example](C:\Users\Admin\Desktop\CAPSTONE PROJ\smart-grid-forecasting-dashboard\frontend\.env.example):

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## API Overview

- `GET /status` returns the training state.
- `POST /train` starts background model training.
- `GET /metrics` returns MAE, RMSE, and R2 for all trained models.
- `GET /predictions` returns actual vs predicted values for the hybrid model.
- `POST /predict` predicts the next value from a 96-point sequence.
