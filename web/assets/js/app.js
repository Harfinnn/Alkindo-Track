
import { fetchStatus, loadComponent } from "./api.js";
import { initSidebar } from "./sidebar.js";
import { initHeader } from "./header.js";

let pollingInterval = null;
let isPolling = false;

function updateDashboardConnection(connected) {
  const status = document.getElementById("dashboardConnection");
  const dot = document.querySelector(".system-dot");

  if (status) {
    status.textContent = connected
      ? "Sistem Terhubung"
      : "Koneksi Terputus";
  }

  if (dot) {
    dot.style.backgroundColor = connected
      ? "#10b981"
      : "#ef4444";
  }
}

async function pollStatus(connection) {
  if (isPolling) return;

  isPolling = true;

  try {
    await fetchStatus();

    connection?.setConnection(true);
    updateDashboardConnection(true);

  } catch (error) {
    console.error(
      "[GateWatch] Status polling failed:",
      error
    );

    connection?.setConnection(false);
    updateDashboardConnection(false);

  } finally {
    isPolling = false;
  }
}

function initPolling(connection) {
  if (pollingInterval !== null) {
    clearInterval(pollingInterval);
  }

  // Fetch pertama kali.
  pollStatus(connection);

  // Polling setiap 2 detik.
  pollingInterval = window.setInterval(() => {
    pollStatus(connection);
  }, 2000);
}

async function initialize() {
  try {
    // Load komponen terlebih dahulu.
    await Promise.all([
      loadComponent(
        "sidebar-container",
        "components/sidebar.html"
      ),

      loadComponent(
        "header-container",
        "components/header.html"
      )
    ]);

    // Inisialisasi navigasi setelah HTML tersedia.
    initSidebar();

    // Inisialisasi header.
    const connection = initHeader();

    // Dashboard hanya mengambil status sistem.
    // Tidak menjalankan stream CCTV.
    // Tidak mengambil Activity Logs.

    initPolling(connection);

  } catch (error) {
    console.error(
      "[GateWatch] Initialization failed:",
      error
    );

    const notice = document.createElement("div");

    notice.className = "app-error";

    notice.textContent =
      "Gagal memuat komponen dashboard. Periksa server dan path components/.";

    document.body.prepend(notice);
  }
}

initialize();
