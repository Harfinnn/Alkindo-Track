
import { initStream } from "./stream.js";
import { initSidebar } from "./sidebar.js";
import { fetchLogs, fetchStatus } from "./api.js";

import { renderDashboard } from "./dashboard.js";
import { renderCameras } from "./camera.js";

/* =========================================
   CONFIGURATION
========================================= */

const MONITORING_POLL_INTERVAL = 2000;
const LOGS_POLL_INTERVAL = 5000;
const MAX_LOGS = 50;

let monitoringInterval = null;
let logsInterval = null;

let isMonitoringPolling = false;
let isLogsPolling = false;

/* =========================================
   LOAD COMPONENT
========================================= */

async function loadComponent(elementId, url) {
  const element = document.getElementById(elementId);

  if (!element) {
    console.warn(
      `[GateWatch] Element #${elementId} tidak ditemukan.`
    );
    return;
  }

  try {
    const response = await fetch(url, {
      cache: "no-cache"
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    element.innerHTML = await response.text();

  } catch (error) {
    console.error(
      `[GateWatch] Gagal memuat ${url}:`,
      error
    );
  }
}

/* =========================================
   MONITORING DATA
   - OCCUPANCY
   - DAILY COUNTER
   - CAMERA STATUS
========================================= */

async function loadMonitoringData() {
  if (isMonitoringPolling) return;

  isMonitoringPolling = true;

  try {
    const data = await fetchStatus();

    // 1. Render okupansi dan hitungan hari ini
    renderDashboard(data);

    // 2. Render status seluruh kamera
    renderCameras(data.cameras || []);

    console.log(
      "[GateWatch] Monitoring data updated."
    );

  } catch (error) {
    console.error(
      "[GateWatch] Monitoring data failed:",
      error
    );

  } finally {
    isMonitoringPolling = false;
  }
}

/* =========================================
   ACTIVITY LOGS HELPERS
========================================= */

function cap(value) {
  const text = String(value ?? "-");

  return text.charAt(0) + text.slice(1).toLowerCase();
}

function createCell(text, className = "") {
  const cell = document.createElement("td");

  if (className) {
    cell.className = className;
  }

  cell.textContent = String(text ?? "-");

  return cell;
}

/* =========================================
   RENDER ACTIVITY LOGS
========================================= */

function renderLogs(list = []) {
  const tableBody = document.getElementById("logs");
  const emptyState = document.getElementById("logEmpty");
  const logCount = document.getElementById("logCount");
  const lastUpdated = document.getElementById("lastUpdated");

  if (!tableBody) return;

  tableBody.replaceChildren();

  const logs = Array.isArray(list)
    ? list.slice(0, MAX_LOGS)
    : [];

  if (emptyState) {
    emptyState.hidden = logs.length > 0;
  }

  if (logCount) {
    logCount.textContent = `${logs.length} kejadian`;
  }

  logs.forEach((event) => {
    const row = document.createElement("tr");

    const direction =
      String(event.direction ?? "").toLowerCase() === "in"
        ? "in"
        : "out";

    const time = String(event.time ?? "-");

    const timeLabel = time.includes("T")
      ? time.split("T")[1]?.split(".")[0] || time
      : time;

    row.appendChild(
      createCell(timeLabel)
    );

    row.appendChild(
      createCell(event.camera)
    );

    row.appendChild(
      createCell(event.gate)
    );

    row.appendChild(
      createCell(cap(event.category))
    );

    row.appendChild(
      createCell(
        direction === "in" ? "MASUK" : "KELUAR",
        `tag ${direction}`
      )
    );

    row.appendChild(
      createCell(event.track_id)
    );

    tableBody.appendChild(row);
  });

  if (lastUpdated) {
    lastUpdated.textContent =
      `Diperbarui ${new Date().toLocaleTimeString("id-ID")}`;
  }
}

/* =========================================
   FETCH ACTIVITY LOGS
========================================= */

async function loadLogs() {
  if (isLogsPolling) return;

  isLogsPolling = true;

  const refreshButton =
    document.getElementById("refreshLogs");

  try {
    if (refreshButton) {
      refreshButton.disabled = true;
    }

    const data = await fetchLogs();

    const logs = Array.isArray(data)
      ? data
      : data?.logs || [];

    renderLogs(logs);

  } catch (error) {
    console.error(
      "[GateWatch] Failed to fetch logs:",
      error
    );

  } finally {
    isLogsPolling = false;

    if (refreshButton) {
      refreshButton.disabled = false;
    }
  }
}

/* =========================================
   POLLING MANAGEMENT
========================================= */

function startMonitoringPolling() {
  stopMonitoringPolling();

  // Fetch pertama kali
  loadMonitoringData();

  // Update otomatis
  monitoringInterval = window.setInterval(
    loadMonitoringData,
    MONITORING_POLL_INTERVAL
  );
}

function stopMonitoringPolling() {
  if (monitoringInterval !== null) {
    window.clearInterval(monitoringInterval);
    monitoringInterval = null;
  }
}

function startLogsPolling() {
  stopLogsPolling();

  // Fetch pertama kali
  loadLogs();

  // Update otomatis
  logsInterval = window.setInterval(
    loadLogs,
    LOGS_POLL_INTERVAL
  );
}

function stopLogsPolling() {
  if (logsInterval !== null) {
    window.clearInterval(logsInterval);
    logsInterval = null;
  }
}

/* =========================================
   INITIALIZATION
========================================= */

async function init() {
  console.log(
    "[GateWatch] Initializing Monitoring..."
  );

  // 1. Load sidebar
  await loadComponent(
    "sidebar-container",
    "components/sidebar.html"
  );

  // 2. Initialize sidebar
  initSidebar();

  // 3. Initialize live CCTV
  initStream();

  // 4. Initialize manual refresh logs
  const refreshButton =
    document.getElementById("refreshLogs");

  refreshButton?.addEventListener(
    "click",
    loadLogs
  );

  // 5. Load monitoring data
  startMonitoringPolling();

  // 6. Load activity logs
  startLogsPolling();

  console.log(
    "[GateWatch] Monitoring initialized successfully."
  );
}

/* =========================================
   START APPLICATION
========================================= */

init();

/* =========================================
   CLEANUP
========================================= */

window.addEventListener("beforeunload", () => {
  stopMonitoringPolling();
  stopLogsPolling();
});
