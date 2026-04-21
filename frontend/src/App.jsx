import { useEffect, useState } from "react";
import axios from "axios";
import {
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Filler,
  Legend,
  LineElement,
  LinearScale,
  PointElement,
  Title,
  Tooltip,
} from "chart.js";
import { Bar, Line } from "react-chartjs-2";

import "./styles.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000",
});

const modelOptions = [
  { value: "hybrid", label: "Hybrid" },
  { value: "lstm", label: "LSTM" },
  { value: "mlp", label: "MLP" },
  { value: "attention", label: "LSTM + Attention" },
  { value: "nbeats", label: "N-BEATS" },
  { value: "baseline", label: "Baseline" },
];

const defaultSequence = Array.from({ length: 96 }, (_, index) =>
  Number((4.8 + Math.sin(index / 8) * 0.7 + Math.cos(index / 13) * 0.35).toFixed(4))
);

function App() {
  const [status, setStatus] = useState("idle");
  const [statusMeta, setStatusMeta] = useState({});
  const [metrics, setMetrics] = useState({});
  const [predictions, setPredictions] = useState({ actual: [], predicted: [] });
  const [recommendation, setRecommendation] = useState(null);
  const [sequenceInput, setSequenceInput] = useState(defaultSequence.join(", "));
  const [selectedModel, setSelectedModel] = useState("hybrid");
  const [predictionResult, setPredictionResult] = useState(null);
  const [scaleFactor, setScaleFactor] = useState("1.10");
  const [simulationResult, setSimulationResult] = useState(null);
  const [threshold, setThreshold] = useState("6.0");
  const [alertResult, setAlertResult] = useState(null);
  const [dashboardError, setDashboardError] = useState("");
  const [predictError, setPredictError] = useState("");
  const [simulateError, setSimulateError] = useState("");
  const [alertError, setAlertError] = useState("");
  const [isSubmittingPrediction, setIsSubmittingPrediction] = useState(false);
  const [isSubmittingSimulation, setIsSubmittingSimulation] = useState(false);
  const [isSubmittingAlert, setIsSubmittingAlert] = useState(false);

  const isTraining = status === "training";

  const fetchStatus = async () => {
    const { data } = await api.get("/status");
    setStatus(data.status);
    setStatusMeta(data);
    return data;
  };

  const fetchMetrics = async () => {
    try {
      const { data } = await api.get("/metrics");
      setMetrics(data.models ?? {});
    } catch (error) {
      if (error.response?.status !== 404) {
        throw error;
      }
    }
  };

  const fetchPredictions = async () => {
    try {
      const { data } = await api.get("/predictions");
      setPredictions({
        actual: data.actual ?? [],
        predicted: data.predicted ?? [],
      });
    } catch (error) {
      if (error.response?.status !== 404) {
        throw error;
      }
    }
  };

  const fetchRecommendation = async () => {
    try {
      const { data } = await api.get("/recommendation");
      setRecommendation(data);
    } catch (error) {
      if (error.response?.status !== 404) {
        throw error;
      }
    }
  };

  const refreshWorkspace = async () => {
    try {
      await Promise.all([fetchStatus(), fetchMetrics(), fetchPredictions(), fetchRecommendation()]);
      setDashboardError("");
    } catch (error) {
      setDashboardError(error.response?.data?.detail || "Unable to reach the backend.");
    }
  };

  useEffect(() => {
    refreshWorkspace();
  }, []);

  useEffect(() => {
    if (!isTraining) {
      return undefined;
    }

    const intervalId = window.setInterval(async () => {
      try {
        const latestStatus = await fetchStatus();
        if (latestStatus.status === "completed") {
          await Promise.all([fetchMetrics(), fetchPredictions(), fetchRecommendation()]);
        }
      } catch (error) {
        setDashboardError(error.response?.data?.detail || "Polling failed.");
      }
    }, 2000);

    return () => window.clearInterval(intervalId);
  }, [isTraining]);

  const modelRows = Object.entries(metrics)
    .map(([key, values]) => ({
      key,
      label: modelOptions.find((option) => option.value === key)?.label ?? key,
      ...values,
    }))
    .sort((a, b) => b.r2 - a.r2);

  const bestModel = recommendation?.best_model ?? modelRows[0]?.key ?? null;
  const topR2 = modelRows[0]?.r2;
  const activePrediction = simulationResult?.simulated_prediction ?? predictionResult?.value ?? null;
  const scenarioDirection =
    simulationResult == null ? "No scenario yet" : simulationResult.delta > 0 ? "Upward pressure" : simulationResult.delta < 0 ? "Relief signal" : "Flat response";
  const riskTone = alertResult?.alert ? "Critical watch" : activePrediction != null ? "Within envelope" : "No signal";

  const r2ChartData = {
    labels: modelRows.map((row) => row.label),
    datasets: [
      {
        label: "R2 Score",
        data: modelRows.map((row) => row.r2),
        backgroundColor: modelRows.map((row) => (row.key === bestModel ? "#0f766e" : "#94a3b8")),
        borderRadius: 12,
      },
    ],
  };

  const forecastChartData = {
    labels: predictions.actual.map((_, index) => `T${index + 1}`),
    datasets: [
      {
        label: "Actual",
        data: predictions.actual,
        borderColor: "#111827",
        backgroundColor: "rgba(17, 24, 39, 0.08)",
        tension: 0.3,
        fill: false,
      },
      {
        label: "Hybrid Forecast",
        data: predictions.predicted,
        borderColor: "#0f766e",
        backgroundColor: "rgba(15, 118, 110, 0.14)",
        tension: 0.3,
        fill: true,
      },
    ],
  };

  const parseSequence = () =>
    sequenceInput
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean)
      .map((value) => Number(value));

  const validateSequence = () => {
    const sequence = parseSequence();
    if (sequence.length !== 96 || sequence.some((value) => Number.isNaN(value))) {
      throw new Error("Please provide exactly 96 comma-separated numeric values.");
    }
    return sequence;
  };

  const handleTrain = async () => {
    try {
      setDashboardError("");
      setStatus("training");
      await api.post("/train");
      setPredictionResult(null);
      setSimulationResult(null);
      setAlertResult(null);
    } catch (error) {
      setDashboardError(error.response?.data?.detail || "Failed to start training.");
      setStatus("idle");
    }
  };

  const handleLoadSample = () => {
    setSequenceInput(defaultSequence.join(", "));
    setPredictError("");
    setSimulateError("");
  };

  const handlePredict = async () => {
    setIsSubmittingPrediction(true);
    setPredictError("");

    try {
      const sequence = validateSequence();
      const { data } = await api.post("/predict", {
        sequence,
        model: selectedModel,
      });

      setPredictionResult({
        model: data.model,
        value: data.prediction,
      });
    } catch (error) {
      setPredictionResult(null);
      setPredictError(error.response?.data?.detail || error.message || "Prediction failed.");
    } finally {
      setIsSubmittingPrediction(false);
    }
  };

  const handleSimulate = async () => {
    setIsSubmittingSimulation(true);
    setSimulateError("");

    try {
      const sequence = validateSequence();
      const numericScale = Number(scaleFactor);

      if (Number.isNaN(numericScale) || numericScale <= 0) {
        throw new Error("Scale factor must be a positive number.");
      }

      const { data } = await api.post("/simulate", {
        sequence,
        scale_factor: numericScale,
      });
      setSimulationResult(data);
    } catch (error) {
      setSimulationResult(null);
      setSimulateError(error.response?.data?.detail || error.message || "Simulation failed.");
    } finally {
      setIsSubmittingSimulation(false);
    }
  };

  const handleAlertCheck = async (predictionOverride) => {
    setIsSubmittingAlert(true);
    setAlertError("");

    try {
      const numericThreshold = Number(threshold);
      const numericPrediction =
        predictionOverride ?? simulationResult?.simulated_prediction ?? predictionResult?.value;

      if (Number.isNaN(numericThreshold)) {
        throw new Error("Threshold must be a number.");
      }

      if (numericPrediction == null) {
        throw new Error("Run a prediction or simulation before checking alerts.");
      }

      const { data } = await api.post("/alert", {
        prediction: numericPrediction,
        threshold: numericThreshold,
      });
      setAlertResult(data);
    } catch (error) {
      setAlertResult(null);
      setAlertError(error.response?.data?.detail || error.message || "Alert evaluation failed.");
    } finally {
      setIsSubmittingAlert(false);
    }
  };

  const operationalPosture =
    alertResult?.alert
      ? "Escalated"
      : predictionResult || simulationResult
        ? "Stable"
        : "Awaiting forecast";
  const topModelLabel = bestModel ? modelOptions.find((item) => item.value === bestModel)?.label : "--";

  return (
    <div className="page-shell">
      <header className="hero-panel">
        <div className="hero-copy-block">
          <p className="eyebrow">AI-Based Smart Grid Forecasting & Decision Support System</p>
          <h1>Anticipate load shifts, stress-test scenarios, and guide smarter grid operations.</h1>
          <p className="hero-copy">
            A decision-support workspace for forecasting demand, comparing model behavior, simulating
            stress conditions, and translating projected load into actionable operating guidance.
          </p>
          <p className="hero-subcopy">
            Use the workbench to move from model readiness to operational posture in one flow:
            train, compare, simulate, evaluate, and respond.
          </p>

          <div className="signal-strip">
            <div className="signal-chip">
              <span className="signal-label">Lead engine</span>
              <strong>{topModelLabel}</strong>
            </div>
            <div className="signal-chip">
              <span className="signal-label">Scenario trend</span>
              <strong>{scenarioDirection}</strong>
            </div>
            <div className="signal-chip">
              <span className="signal-label">Risk state</span>
              <strong>{riskTone}</strong>
            </div>
          </div>

          <div className="hero-highlights">
            <div className="mini-card">
              <span className="mini-label">Forecasting State</span>
              <strong>{status.toUpperCase()}</strong>
            </div>
            <div className="mini-card">
              <span className="mini-label">Primary Model</span>
              <strong>{bestModel ? modelOptions.find((item) => item.value === bestModel)?.label : "--"}</strong>
            </div>
            <div className="mini-card">
              <span className="mini-label">Grid Posture</span>
              <strong>{operationalPosture}</strong>
            </div>
          </div>

          <div className="hero-visual">
            <div className="visual-card primary">
              <span className="visual-label">Forecast Control Loop</span>
              <div className="flow-row">
                <span>Train</span>
                <span>Compare</span>
                <span>Simulate</span>
                <span>Act</span>
              </div>
            </div>
            <div className="visual-card">
              <span className="visual-label">Best R2</span>
              <strong>{topR2 ?? "--"}</strong>
            </div>
            <div className="visual-card">
              <span className="visual-label">Active forecast</span>
              <strong>{activePrediction ?? "--"}</strong>
            </div>
          </div>
        </div>

        <aside className="mission-panel">
          <div className="section-heading compact">
            <div>
              <p className="eyebrow">Control Center</p>
              <h2>System readiness</h2>
            </div>
          </div>

          <button className="primary-button" onClick={handleTrain} disabled={isTraining}>
            {isTraining ? "Training..." : "Train Models"}
          </button>
          <button className="ghost-button" onClick={refreshWorkspace}>
            Refresh Workspace
          </button>

          <div className={`status-badge status-${status}`}>{status.toUpperCase()}</div>
          {isTraining && <div className="spinner" aria-label="Training in progress" />}
          {statusMeta.error && <p className="error-text">Training error: {statusMeta.error}</p>}
          {dashboardError && <p className="error-text">{dashboardError}</p>}

          <div className="bullet-metrics">
            <div>
              <span>Forecast engines</span>
              <strong>{modelRows.length || 6}</strong>
            </div>
            <div>
              <span>Observation window</span>
              <strong>96</strong>
            </div>
            <div>
              <span>Scenario planning</span>
              <strong>Enabled</strong>
            </div>
          </div>
        </aside>
      </header>

      <main className="workspace-stack">
        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Decision Brief</p>
              <h2>Executive model brief</h2>
            </div>
          </div>

          <div className="brief-grid">
            <article className="insight-card feature">
              <span className="tag">Recommended Engine</span>
              <h3>
                {recommendation
                  ? modelOptions.find((item) => item.value === recommendation.best_model)?.label
                  : "No recommendation yet"}
              </h3>
              <p>{recommendation?.reason || "Train the models to generate a recommendation."}</p>
            </article>

            <article className="insight-card">
              <span className="tag">Forecast Coverage</span>
              <h3>{predictions.actual.length > 0 ? "Evaluation trace ready" : "Awaiting benchmark trace"}</h3>
              <p>
                {predictions.actual.length > 0
                  ? `Hybrid forecast contains ${predictions.actual.length} validation points for operational review and chart-based inspection.`
                  : "No prediction trace is available yet."}
              </p>
            </article>

            <article className="insight-card">
              <span className="tag">Operational Risk</span>
              <h3>{alertResult?.message || "Threshold monitor online"}</h3>
              <p>
                {alertResult?.recommendation ||
                  "Set a threshold and evaluate a prediction to receive an operating recommendation."}
              </p>
            </article>
          </div>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Model Comparison</p>
              <h2>Benchmark leaderboard</h2>
            </div>
          </div>

          <div className="comparison-layout">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>R2</th>
                  </tr>
                </thead>
                <tbody>
                  {modelRows.length === 0 ? (
                    <tr>
                      <td colSpan="4" className="empty-state">
                        Train models to populate the benchmark table.
                      </td>
                    </tr>
                  ) : (
                    modelRows.map((row) => (
                      <tr key={row.key} className={row.key === bestModel ? "best-row" : ""}>
                        <td>
                          {row.label}
                          {row.key === bestModel ? <span className="best-pill">Best</span> : null}
                        </td>
                        <td>{row.mae}</td>
                        <td>{row.rmse}</td>
                        <td>{row.r2}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="chart-box">
              <Bar
                data={r2ChartData}
                options={{
                  responsive: true,
                  maintainAspectRatio: false,
                  plugins: { legend: { display: false } },
                }}
              />
            </div>
          </div>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Forecast Visualization</p>
              <h2>Hybrid forecast trace</h2>
            </div>
          </div>

          <div className="chart-box large">
            <Line
              data={forecastChartData}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: "index", intersect: false },
                plugins: { legend: { position: "top" } },
                scales: {
                  x: { ticks: { maxTicksLimit: 14 } },
                },
              }}
            />
          </div>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Forecast Workbench</p>
              <h2>Operational scenario workbench</h2>
            </div>
          </div>

          <div className="workbench-layout">
            <div className="sequence-panel">
              <label htmlFor="sequence" className="input-label">
                96-point demand sequence
              </label>
              <textarea
                id="sequence"
                value={sequenceInput}
                onChange={(event) => setSequenceInput(event.target.value)}
                placeholder="Enter 96 comma-separated load readings"
                rows={10}
              />
              <div className="action-row">
                <button className="ghost-button" onClick={handleLoadSample}>
                  Load Sample Sequence
                </button>
              </div>
            </div>

            <div className="tool-stack">
              <div className="tool-card">
                <h3>Next-Step Forecast</h3>
                <div className="chip-row">
                  <span className="console-chip neutral">Primary action</span>
                  <span className="console-chip">Model-driven</span>
                </div>
                <label htmlFor="model-select" className="input-label">
                  Model selection
                </label>
                <select
                  id="model-select"
                  value={selectedModel}
                  onChange={(event) => setSelectedModel(event.target.value)}
                >
                  {modelOptions.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
                <button className="primary-button" onClick={handlePredict} disabled={isSubmittingPrediction}>
                  {isSubmittingPrediction ? "Forecasting..." : "Generate Forecast"}
                </button>
                <div className="result-panel">
                  <span className="result-label">Projected next load</span>
                  <strong>{predictionResult?.value ?? "--"}</strong>
                  <div className="micro-stat-row">
                    <span>{predictionResult?.model ? `Using ${predictionResult.model}` : "Choose a model"}</span>
                    <span>{topR2 != null ? `Best R2 ${topR2}` : "Awaiting training"}</span>
                  </div>
                </div>
                {predictError && <p className="error-text">{predictError}</p>}
              </div>

              <div className="tool-card">
                <h3>Scenario Simulation</h3>
                <div className="chip-row">
                  <span className="console-chip neutral">Stress test</span>
                  <span className="console-chip">What-if planning</span>
                </div>
                <label htmlFor="scale-factor" className="input-label">
                  Demand multiplier
                </label>
                <input
                  id="scale-factor"
                  className="text-input"
                  value={scaleFactor}
                  onChange={(event) => setScaleFactor(event.target.value)}
                  placeholder="1.10"
                />
                <button className="primary-button" onClick={handleSimulate} disabled={isSubmittingSimulation}>
                  {isSubmittingSimulation ? "Running..." : "Run Scenario"}
                </button>
                <div className="scenario-results">
                  <div>
                    <span>Base case</span>
                    <strong>{simulationResult?.original_prediction ?? "--"}</strong>
                  </div>
                  <div>
                    <span>Scenario case</span>
                    <strong>{simulationResult?.simulated_prediction ?? "--"}</strong>
                  </div>
                </div>
                <p className="support-copy">
                  {simulationResult?.insight || "Model how amplified or reduced demand could shift the next-step forecast."}
                </p>
                {simulateError && <p className="error-text">{simulateError}</p>}
              </div>

              <div className="tool-card">
                <h3>Risk & Alert Monitor</h3>
                <div className="chip-row">
                  <span className={`console-chip ${alertResult?.alert ? "danger" : "safe"}`}>
                    {alertResult?.alert ? "Threshold breach" : "Threshold watch"}
                  </span>
                  <span className="console-chip">Operator cue</span>
                </div>
                <label htmlFor="threshold" className="input-label">
                  Operating threshold
                </label>
                <input
                  id="threshold"
                  className="text-input"
                  value={threshold}
                  onChange={(event) => setThreshold(event.target.value)}
                  placeholder="6.0"
                />
                <button className="primary-button" onClick={() => handleAlertCheck()} disabled={isSubmittingAlert}>
                  {isSubmittingAlert ? "Evaluating..." : "Assess Risk"}
                </button>
                <div className={`alert-panel ${alertResult?.alert ? "alert-high" : "alert-normal"}`}>
                  <span className="result-label">Risk status</span>
                  <strong>{alertResult?.message || "Waiting for evaluation"}</strong>
                  <p>{alertResult?.recommendation || "Use a prediction or simulation result to evaluate the threshold."}</p>
                  <div className="micro-stat-row">
                    <span>{alertResult ? `Margin ${alertResult.margin}` : "No margin computed"}</span>
                    <span>{alertResult ? `Severity ${alertResult.severity}` : "No severity yet"}</span>
                  </div>
                </div>
                {alertError && <p className="error-text">{alertError}</p>}
              </div>
            </div>
          </div>
        </section>

        <section className="card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Operational Guidance</p>
              <h2>Action summary</h2>
            </div>
          </div>

          <div className="brief-grid">
            <article className="insight-card">
              <span className="tag">Lead Engine</span>
              <h3>{bestModel ? modelOptions.find((item) => item.value === bestModel)?.label : "Pending"}</h3>
              <p>
                {recommendation?.reason ||
                  "Once training completes, the best-performing model will be highlighted here for operational use."}
              </p>
            </article>

            <article className="insight-card">
              <span className="tag">Scenario Shift</span>
              <h3>{simulationResult ? simulationResult.delta : "--"}</h3>
              <p>
                {simulationResult
                  ? `A demand multiplier of ${simulationResult.scale_factor} shifted the hybrid outlook by ${simulationResult.delta}.`
                  : "Run a simulation to understand how demand scaling shifts the forecast."}
              </p>
            </article>

            <article className="insight-card">
              <span className="tag">Decision Signal</span>
              <h3>{activePrediction != null ? activePrediction : "--"}</h3>
              <p>
                {alertResult
                  ? `Current forecast margin versus threshold is ${alertResult.margin}, with posture marked as ${alertResult.severity}.`
                  : "Evaluate an alert to understand whether the projected load breaches your operating threshold."}
              </p>
            </article>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
