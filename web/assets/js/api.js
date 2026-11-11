
/* API layer: all backend endpoints stay centralized here. */

const API_BASE_URL = "http://localhost:8000";
const VIEWER_BASE_URL = "http://localhost:8080";

const API = {
  status: `${API_BASE_URL}/api/status`,
  logs: `${API_BASE_URL}/api/logs?limit=50`,
};

async function getJson(url) {
  const response = await fetch(url, {
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(
      `HTTP ${response.status} saat mengambil ${url}`
    );
  }

  return response.json();
}

export function fetchStatus() {
  return getJson(API.status);
}

export function fetchLogs() {
  return getJson(API.logs);
}

export function getStreamUrl() {
  return `${VIEWER_BASE_URL}/stream?t=${Date.now()}`;
}

export async function loadComponent(containerId, filePath) {
  const container = document.getElementById(containerId);

  if (!container) {
    throw new Error(
      `Container #${containerId} tidak ditemukan`
    );
  }

  const response = await fetch(filePath, {
    cache: "no-cache"
  });

  if (!response.ok) {
    throw new Error(
      `Gagal memuat ${filePath}: HTTP ${response.status}`
    );
  }

  container.innerHTML = await response.text();
}
