
import { showNotification, showConfirmation } from "./notification.js";

const TRACKING_API = "/api/control";
const FACE_API = "/api/face/control";

// =====================================================
// TRACKING ELEMENTS
// =====================================================

const trackingStartBtn = document.getElementById("start-engine");
const trackingStopBtn = document.getElementById("stop-engine");

const trackingBadge = document.getElementById("engine-status");
const trackingTitle = document.getElementById("engine-title");
const trackingDescription = document.getElementById("engine-description");
const trackingPid = document.getElementById("engine-pid");

// =====================================================
// FACE RECOGNITION ELEMENTS
// =====================================================

const faceStartBtn = document.getElementById("start-face-engine");
const faceStopBtn = document.getElementById("stop-face-engine");

const faceBadge = document.getElementById("face-engine-status");
const faceTitle = document.getElementById("face-engine-title");
const faceDescription = document.getElementById("face-engine-description");
const facePid = document.getElementById("face-engine-pid");

const faceCameraIndicator = document.getElementById("face-camera-indicator");
const faceCameraStatus = document.getElementById("face-camera-status");

// =====================================================
// ENGINE STATE
// =====================================================

let trackingBusy = false;
let faceBusy = false;

// =====================================================
// API HELPER
// =====================================================

async function requestJSON(url, options = {}) {
    const response = await fetch(url, {
        cache: "no-store",
        ...options
    });

    let data;

    try {
        data = await response.json();
    } catch {
        throw new Error(`Respons server tidak valid: ${response.status}`);
    }

    if (!response.ok) {
        throw new Error(
            data.message ||
            data.error ||
            `Request gagal (${response.status})`
        );
    }

    return data;
}

// =====================================================
// 1. TRACKING STATUS
// =====================================================

function renderTrackingStatus(data) {
    if (!trackingBadge) return;

    const running = data.running === true;

    trackingBadge.textContent = running ? "RUNNING" : "STOPPED";
    trackingBadge.className = running
        ? "status-badge online"
        : "status-badge offline";

    trackingTitle.textContent = running
        ? "Tracking Aktif"
        : "Tracking Offline";

    trackingDescription.textContent = running
        ? "AI People & Vehicle Tracking sedang berjalan."
        : "Sistem AI People & Vehicle Tracking belum dijalankan.";

    trackingPid.textContent = running
        ? `Process ID: ${data.pid}`
        : "Process ID: -";

    trackingStartBtn.disabled = running || trackingBusy;
    trackingStopBtn.disabled = !running || trackingBusy;

    trackingStartBtn.textContent = trackingBusy
        ? "Processing..."
        : "▶ Start Tracking";

    trackingStopBtn.textContent = trackingBusy
        ? "Processing..."
        : "■ Stop Tracking";
}

async function updateTrackingStatus() {
    try {
        const data = await requestJSON(`${TRACKING_API}/status`);
        renderTrackingStatus(data);

    } catch (error) {
        console.error("[GateWatch] Tracking status:", error);

        if (trackingTitle) {
            trackingTitle.textContent = "Control Center Offline";
            trackingDescription.textContent =
                "Tidak dapat mengambil status Tracking.";
        }
    }
}

// =====================================================
// 2. FACE RECOGNITION STATUS
// =====================================================

function renderFaceProcessStatus(data) {
    if (!faceBadge) return;

    const running = data.running === true;

    faceBadge.textContent = running ? "RUNNING" : "STOPPED";
    faceBadge.className = running
        ? "status-badge online"
        : "status-badge offline";

    faceTitle.textContent = running
        ? "Face Recognition Aktif"
        : "Face Recognition Offline";

    faceDescription.textContent = running
        ? "AI Face Detection & Recognition sedang berjalan."
        : "Engine Face Recognition belum dijalankan.";

    facePid.textContent = running
        ? `Process ID: ${data.pid}`
        : "Process ID: -";

    faceStartBtn.disabled = running || faceBusy;
    faceStopBtn.disabled = !running || faceBusy;

    faceStartBtn.textContent = faceBusy
        ? "Processing..."
        : "▶ Start Face Recognition";

    faceStopBtn.textContent = faceBusy
        ? "Processing..."
        : "■ Stop Face Recognition";
}

function renderFaceCameraStatus(data) {
    if (!faceCameraIndicator || !faceCameraStatus) return;

    const connected = data.camera_connected === true;
    const running = data.running === true;

    faceCameraIndicator.className = connected
        ? "face-indicator online"
        : "face-indicator offline";

    if (connected) {
        faceCameraStatus.textContent =
            `Camera Recognition Online · ${data.fps ?? 0} FPS`;
    } else if (running) {
        faceCameraStatus.textContent =
            data.camera_error || "Engine aktif, menunggu koneksi CCTV...";
    } else {
        faceCameraStatus.textContent =
            "Camera Recognition Offline";
    }
}

async function updateFaceStatus() {
    try {
        const data = await requestJSON("/api/face/status");

        renderFaceProcessStatus(data);
        renderFaceCameraStatus(data);

    } catch (error) {
        console.error(
            "[GateWatch] Face Recognition status:",
            error
        );

        if (faceTitle) {
            faceTitle.textContent = "Face Service Unavailable";
            faceDescription.textContent =
                "Tidak dapat mengambil status Face Recognition.";
        }

        if (faceCameraIndicator) {
            faceCameraIndicator.className = "face-indicator offline";
        }

        if (faceCameraStatus) {
            faceCameraStatus.textContent =
                "Status kamera tidak tersedia";
        }
    }
}

// =====================================================
// 3. TRACKING CONTROL
// =====================================================

async function controlTracking(action) {
    if (trackingBusy) return;

    trackingBusy = true;

    trackingStartBtn.disabled = true;
    trackingStopBtn.disabled = true;

    try {
        const result = await requestJSON(
            `${TRACKING_API}/${action}`,
            { method: "POST" }
        );

        showNotification(
            result.message ||
            (action === "start"
                ? "AI People & Vehicle Tracking berhasil dijalankan."
                : "AI People & Vehicle Tracking berhasil dihentikan."),
            "success"
        );

        console.log("[GateWatch] Tracking:", result.message);

        await updateTrackingStatus();

    } catch (error) {
        console.error("[GateWatch] Tracking control:", error);

        showNotification(error.message, "error", 5000);

    } finally {
        trackingBusy = false;
        await updateTrackingStatus();
    }
}

// =====================================================
// 4. FACE RECOGNITION CONTROL
// =====================================================

async function controlFaceRecognition(action) {
    if (faceBusy) return;

    faceBusy = true;

    faceStartBtn.disabled = true;
    faceStopBtn.disabled = true;

    try {
        const result = await requestJSON(
            `${FACE_API}/${action}`,
            { method: "POST" }
        );

        showNotification(
            result.message ||
            (action === "start"
                ? "Face Recognition berhasil dijalankan."
                : "Face Recognition berhasil dihentikan."),
            "success"
        );

        console.log("[GateWatch] Face Recognition:", result.message);

        await updateFaceStatus();

    } catch (error) {
        console.error("[GateWatch] Face Recognition control:", error);

        showNotification(error.message, "error", 5000);

    } finally {
        faceBusy = false;
        await updateFaceStatus();
    }
}

// =====================================================
// 5. EVENT LISTENERS
// =====================================================

// Tracking Start
trackingStartBtn?.addEventListener("click", () => {
    controlTracking("start");
});

// Tracking Stop
trackingStopBtn?.addEventListener("click", async () => {
    const confirmed = await showConfirmation({
        title: "Stop People & Vehicle Tracking?",
        message: "Apakah Anda yakin ingin menghentikan sistem AI Tracking?",
        confirmText: "Ya, Stop Tracking",
        cancelText: "Batal",
        type: "warning"
    });

    if (confirmed) {
        controlTracking("stop");
    }
});

// Face Recognition Start
faceStartBtn?.addEventListener("click", () => {
    controlFaceRecognition("start");
});

// Face Recognition Stop
faceStopBtn?.addEventListener("click", async () => {
    const confirmed = await showConfirmation({
        title: "Stop Face Recognition?",
        message: "Apakah Anda yakin ingin menghentikan sistem Face Recognition?",
        confirmText: "Ya, Stop Engine",
        cancelText: "Batal",
        type: "warning"
    });

    if (confirmed) {
        controlFaceRecognition("stop");
    }
});

// =====================================================
// 6. INITIALIZATION & POLLING
// =====================================================

async function updateAllEngineStatus() {
    await Promise.all([
        updateTrackingStatus(),
        updateFaceStatus()
    ]);
}

updateAllEngineStatus();

setInterval(updateAllEngineStatus, 2000);
