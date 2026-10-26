/* =========================================================
   NAB ANOMALY DETECTION MONITORING
   Clean Frontend Controller
========================================================= */

let liveChart = null;
let monitoringTimer = null;
let isMonitoring = false;
let isPaused = false;
let currentObservation = 0;
let anomalyCount = 0;
let anomalyRecords = [];

const algorithmDefaults = {
    "Isolation Forest": { contamination: 0.005, n_estimators: 100, max_samples: 0.8 },
    "LOF": { contamination: 0.005, n_neighbors: 50 },
    "One-Class SVM": { nu: 0.005, gamma: 0.01 },
    "DBSCAN": { eps: 0.7, min_samples: 5 },
    "LSTM Autoencoder": { epochs: 20, batch_size: 64, latent_dim: 16 },
    "Half-Space Trees": { window_size: 250, n_trees: 25 }
};

function updateAlgorithmParams() {
    const algo = document.getElementById("algorithm-select")?.value;
    const container = document.getElementById("param-fields");
    if (!container) return;

    const defaults = algorithmDefaults[algo] || {};
    container.innerHTML = "";
    for (const [key, val] of Object.entries(defaults)) {
        // Contamination ya nu jaise parameters ke liye max limit set karna taaki error na aaye
        let maxAttr = "";
        if (key === "contamination" || key === "nu") {
            maxAttr = `max="0.5" min="0.001"`;
        }

        container.innerHTML += `
            <div>
                <label style="font-size: 11px; color: var(--muted); display: block; margin-bottom: 3px;">${key}</label>
                <input type="number" step="any" ${maxAttr} id="param-${key}" value="${val}" style="width: 100%; padding: 6px; border-radius: 6px; border: 1px solid var(--border); font-size: 12px;">
            </div>
        `;
    }
}

function playAnomalySound() {
    const sound = document.getElementById("anomaly-sound");
    if (!sound) return;
    sound.currentTime = 0;
    sound.play().catch(error => console.log("Sound play error:", error));
}

async function startMonitoring() {
    const datasetSelect = document.getElementById("dataset-select");
    const algorithmSelect = document.getElementById("algorithm-select");

    const selectedDataset = datasetSelect ? datasetSelect.value : "";
    const selectedAlgorithm = algorithmSelect ? algorithmSelect.value : "Isolation Forest";

    const defaults = algorithmDefaults[selectedAlgorithm] || {};
    const params = {};
    for (const key of Object.keys(defaults)) {
        const input = document.getElementById(`param-${key}`);
        if (input) params[key] = Number(input.value);
    }

    if (!selectedDataset) {
        alert("Please select a NAB dataset first.");
        return;
    }

    try {
        const response = await fetch("/api/start-monitoring", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dataset: selectedDataset, algorithm: selectedAlgorithm, params: params })
        });

        const data = await response.json();
        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not start monitoring.");
        }

        localStorage.setItem("monitoringActive", "true");
        localStorage.setItem("activeDataset", selectedDataset);
        localStorage.setItem("activeAlgorithm", selectedAlgorithm);

        clearMonitoringTimer();
        isMonitoring = true;
        isPaused = false;
        currentObservation = 0;
        anomalyCount = 0;
        anomalyRecords = [];

        resetDashboard();
        updateMonitoringButtons();
        showLiveMonitor();
        initializeLiveChart();

        updateElement("current-dataset", `${data.dataset} (${selectedAlgorithm})`);
        updateMonitorTitle("Monitoring in progress");
        updateMonitorDescription(`Analyzing using ${selectedAlgorithm} with experimental parameters.`);

        runMonitoringLoop();

    } catch (error) {
        console.error("Could not start monitoring:", error);
        alert("Could not start monitoring: " + error.message);
    }
}

function togglePause() {
    if (!isMonitoring) return;

    if (!isPaused) {
        isPaused = true;
        clearMonitoringTimer();
        updateMonitoringButtons();
        updateMonitorTitle("Monitoring paused");
        updateMonitorDescription("Monitoring is paused. Resume to continue.");
        return;
    }

    isPaused = false;
    updateMonitoringButtons();
    updateMonitorTitle("Monitoring in progress");
    updateMonitorDescription("The system is continuously analysing the selected NAB dataset.");
    runMonitoringLoop();
}

function stopMonitoring() {
    localStorage.removeItem("monitoringActive");
    localStorage.removeItem("activeDataset");
    localStorage.removeItem("activeAlgorithm");

    clearMonitoringTimer();
    isMonitoring = false;
    isPaused = false;

    updateMonitoringButtons();
    updateMonitorTitle("Monitoring stopped");
    updateMonitorDescription("The monitoring session has ended.");
}

function clearMonitoringTimer() {
    if (monitoringTimer) {
        clearTimeout(monitoringTimer);
        monitoringTimer = null;
    }
}

async function runMonitoringLoop() {
    clearMonitoringTimer();
    if (!isMonitoring || isPaused) return;

    await processNextObservation();

    if (!isMonitoring || isPaused) return;

    // Delay ko 100ms se hata kar 10ms kar diya hai taaki yeh bilkul tez (fast) chale
    // aur bina kisi lag ke 2 minute ke andar poora process complete ho jaye.
    monitoringTimer = setTimeout(runMonitoringLoop, 25);
}

async function processNextObservation() {
    if (!isMonitoring || isPaused) return;

    try {
        const response = await fetch("/api/next-observation");
        const data = await response.json();

        if (!response.ok || !data.success) {
            throw new Error(data.error || "Could not retrieve observation.");
        }

        if (data.completed) {
            stopMonitoring();
            updateMonitorTitle("Monitoring completed");
            updateMonitorDescription("The selected NAB dataset has been completely analysed.");
            return;
        }

        currentObservation = data.observation;
        updateObservationDisplay(data.value, data.timestamp, data.anomaly);

        const progressFill = document.getElementById("progress-fill");
        const progressText = document.getElementById("progress-text");
        const progress = Number(data.progress) || 0;

        if (progressFill) progressFill.style.width = progress + "%";
        if (progressText) progressText.textContent = progress.toFixed(1) + "%";

        if (data.anomaly) registerBackendAnomaly(data);

        if (progress >= 100) {
            stopMonitoring();
            updateMonitorTitle("Monitoring completed");
            updateMonitorDescription("The selected NAB dataset has been completely analysed.");
        }

    } catch (error) {
        console.error("Monitoring error:", error);
        stopMonitoring();
    }
}

// Har model/algorithm ke apne min aur max scores store karne ke liye object
// Saare scores ka array ya min/max pehle se track karne ke liye
let allScores = [];

function registerBackendAnomaly(data) {
    anomalyCount++;

    const score = Number(data.anomaly_score) || 0;
    allScores.push(score);

    // Backend ya CSV ki hardcoded severity ko IGNORE kar rahe hain
    // taaki hum score ke base par dynamic severity nikal sakein.
    let severity = "";

    // Min aur Max nikal lein saare loaded scores mein se
    const minScore = Math.min(...allScores);
    const maxScore = Math.max(...allScores);

    let normalizedScore = 0.5;
    if (maxScore > minScore) {
        normalizedScore = (score - minScore) / (maxScore - minScore);
    }

    // Ab scores ke basis par properly distribute hoga
    if (normalizedScore > 0.75) severity = "High";
    else if (normalizedScore > 0.35) severity = "Medium";
    else severity = "Low";

    const anomaly = {
        number: anomalyCount,
        observation: data.observation,
        timestamp: data.timestamp,
        value: Number(data.value).toFixed(2),
        score: score.toFixed(3),
        severity: severity
    };

    anomalyRecords.push(anomaly);
    updateAnomalyCounters();
    playAnomalySound();
}

function initializeLiveChart() {
    const canvas = document.getElementById("live-chart");
    if (!canvas) return;

    if (liveChart) {
        liveChart.destroy();
        liveChart = null;
    }

    liveChart = new Chart(canvas, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                { label: "Signal", data: [], borderWidth: 2, tension: 0.35, pointRadius: 0, fill: false },
                { label: "Anomalies", data: [], borderWidth: 0, pointRadius: 6, pointHoverRadius: 8, showLine: false }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: { legend: { display: true } },
            scales: {
                x: { title: { display: true, text: "Observation" } },
                y: { title: { display: true, text: "Signal Value" } }
            }
        }
    });
}

function updateLiveChart(observation, timestamp, signalValue, anomaly) {
    if (!liveChart) return;
    const MAX_VISIBLE_POINTS = 20;

    liveChart.data.labels.push(timestamp);
    liveChart.data.datasets[0].data.push(Number(signalValue));
    liveChart.data.datasets[1].data.push(anomaly ? Number(signalValue) : null);

    if (liveChart.data.labels.length > MAX_VISIBLE_POINTS) {
        liveChart.data.labels.shift();
        liveChart.data.datasets[0].data.shift();
        liveChart.data.datasets[1].data.shift();
    }
    liveChart.update("none");
}

function updateObservationDisplay(value, timestamp, anomaly) {
    updateElement("current-observation", currentObservation);
    updateElement("observation-count", currentObservation);
    updateLiveChart(currentObservation, timestamp, value, anomaly);
}

function updateAnomalyCounters() {
    updateElement("anomaly-count", anomalyCount);
    updateElement("total-anomaly-count", anomalyCount);
}

function resetDashboard() {
    updateElement("observation-count", 0);
    updateElement("current-observation", 0);
    updateElement("anomaly-count", 0);
    updateElement("total-anomaly-count", 0);
    updateElement("progress-text", "0%");

    const progress = document.getElementById("progress-fill");
    if (progress) progress.style.width = "0%";
}

function updateMonitoringButtons() {
    const startButton = document.getElementById("start-button");
    const pauseButton = document.getElementById("pause-button");
    const stopButton = document.getElementById("stop-button");
    if (!startButton || !pauseButton || !stopButton) return;

    if (!isMonitoring) {
        startButton.classList.remove("hidden");
        pauseButton.classList.add("hidden");
        stopButton.classList.add("hidden");
        return;
    }

    startButton.classList.add("hidden");
    stopButton.classList.remove("hidden");
    pauseButton.classList.remove("hidden");
    pauseButton.textContent = isPaused ? "▶ Resume Monitoring" : "⏸ Pause Monitoring";
}

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) element.textContent = value;
}

function updateMonitorTitle(text) { updateElement("monitor-title", text); }
function updateMonitorDescription(text) { updateElement("monitor-description", text); }

function showLiveMonitor() {
    document.getElementById("live-monitor")?.classList.remove("hidden");
}

async function loadDatasets() {
    const select = document.getElementById("dataset-select");
    if (!select) return;

    try {
        const response = await fetch("/api/datasets");
        const data = await response.json();

        if (data && data.datasets) {
            select.innerHTML = '<option value="">Select a NAB dataset...</option>';
            data.datasets.forEach(ds => {
                const opt = document.createElement("option");
                const dsValue = (typeof ds === 'object' && ds !== null)
                    ? (ds.name || ds.filename || ds.id || Object.values(ds)[0])
                    : ds;

                opt.value = dsValue;
                opt.textContent = dsValue;
                select.appendChild(opt);
            });
        }
    } catch (error) {
        console.error("Error loading datasets:", error);
        select.innerHTML = '<option value="">Failed to load datasets</option>';
    }
}

document.addEventListener("DOMContentLoaded", () => {
    loadDatasets();
    updateAlgorithmParams();

    document.getElementById("algorithm-select")?.addEventListener("change", () => {
        updateAlgorithmParams();
        const selectedAlgo = document.getElementById("algorithm-select").value;
        localStorage.setItem("activeAlgorithm", selectedAlgo);
    });

    const isActive = localStorage.getItem("monitoringActive");
    if (isActive === "true" && document.getElementById("dataset-select")) {
        isMonitoring = true;
        isPaused = false;

        const dataset = localStorage.getItem("activeDataset") || "";
        const algorithm = localStorage.getItem("activeAlgorithm") || "Isolation Forest";

        showLiveMonitor();
        initializeLiveChart();

        updateElement("current-dataset", `${dataset} (${algorithm})`);
        updateMonitorTitle("Monitoring session active");
        updateMonitorDescription(`Resumed session using ${algorithm}.`);

        updateMonitoringButtons();
        runMonitoringLoop();
    }
});