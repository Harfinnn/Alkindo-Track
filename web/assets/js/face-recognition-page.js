
import { loadComponent } from "./api.js";
import { initSidebar } from "./sidebar.js";

// =====================================================
// CONFIGURATION
// =====================================================

const STATUS_API = "/api/face/status";

const STREAM_URL = "http://localhost:8081/stream";

const POLLING_INTERVAL = 2000;
const MAX_ACTIVITY = 20;

// =====================================================
// ELEMENTS
// =====================================================

const el = {
    engineStatus: document.getElementById("recognition-engine-status"),
    engineDescription: document.getElementById("recognition-engine-description"),
    engineBadge: document.getElementById("recognition-badge"),

    detected: document.getElementById("total-detected"),
    recognized: document.getElementById("total-recognized"),
    unknown: document.getElementById("total-unknown"),

    stream: document.getElementById("stream"),
    well: document.getElementById("well"),
    message: document.getElementById("msg"),
    streamStatus: document.getElementById("streamStatus"),

    latestAvatar: document.getElementById("latest-face-avatar"),
    latestName: document.getElementById("latest-face-name"),
    latestDetail: document.getElementById("latest-face-detail"),
    latestConfidence: document.getElementById("latest-face-confidence"),

    activity: document.getElementById("recognition-activity"),
    activityCount: document.getElementById("activity-count"),

    detectionModel: document.getElementById("face-detection-model"),
    recognitionModel: document.getElementById("face-recognition-model"),
    backend: document.getElementById("face-backend"),
    device: document.getElementById("face-device")
};

// =====================================================
// STATE
// =====================================================

let streamActive = false;
let streamRetryTimer = null;
let polling = false;

let activityHistory = [];
const activityKeys = new Set();

// =====================================================
// HELPER
// =====================================================

function setText(element, value) {
    if (element) {
        element.textContent = value;
    }
}

function formatPercent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function getTime() {
    return new Date().toLocaleTimeString("id-ID", {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit"
    });
}

async function fetchStatus() {
    const response = await fetch(STATUS_API, {
        cache: "no-store"
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
}

// =====================================================
// 1. ENGINE STATUS
// =====================================================

function renderEngine(data) {
    const running = data.running === true;

    setText(
        el.engineStatus,
        running
            ? "Face Recognition Active"
            : "Waiting for Engine"
    );

    setText(
        el.engineDescription,
        running
            ? `Engine berjalan dengan PID ${data.pid || "-"}`
            : "Menunggu sistem Face Recognition dijalankan."
    );

    if (el.engineBadge) {
        el.engineBadge.className =
            `engine-status-badge ${running ? "online" : "offline"}`;

        const badgeText = running ? "ACTIVE" : "INACTIVE";

        el.engineBadge.lastChild.textContent = badgeText;
    }

    if (el.well) {
        el.well.classList.toggle(
            "down",
            !data.camera_connected
        );
    }
}

// =====================================================
// 2. FACE STATISTICS
// =====================================================

function renderStatistics(data) {
    const faces = Array.isArray(data.faces)
        ? data.faces
        : [];

    const detected = faces.length;

    const recognized = faces.filter(
        face => Boolean(face.name)
    ).length;

    const unknown = detected - recognized;

    setText(el.detected, detected);
    setText(el.recognized, recognized);
    setText(el.unknown, unknown);
}

// =====================================================
// 3. CAMERA STREAM
// =====================================================

function showMessage(message) {
    if (!el.message) return;

    el.message.textContent = message;
    el.message.style.display = "flex";
}

function hideMessage() {
    if (el.message) {
        el.message.style.display = "none";
    }
}

function stopStream() {
    streamActive = false;

    if (streamRetryTimer) {
        clearTimeout(streamRetryTimer);
        streamRetryTimer = null;
    }

    if (el.stream) {
        el.stream.removeAttribute("src");
    }

    showMessage("Face Recognition Offline");

    setText(
        el.streamStatus,
        "Stream tidak aktif."
    );
}

function startStream() {
    if (!el.stream || streamActive) return;

    streamActive = true;

    showMessage("Menghubungkan ke CCTV...");

    setText(
        el.streamStatus,
        "Menghubungkan ke kamera..."
    );

    el.stream.onload = () => {
        hideMessage();

        setText(
            el.streamStatus,
            "CCTV terhubung · Live Recognition"
        );
    };

    el.stream.onerror = () => {
        console.error(
            "[GateWatch] Face stream gagal:",
            STREAM_URL
        );

        streamActive = false;

        el.stream.removeAttribute("src");

        showMessage(
            "Stream terputus. Mencoba menghubungkan kembali..."
        );

        setText(
            el.streamStatus,
            "Koneksi stream terputus."
        );

        scheduleStreamRetry();
    };

    // Jangan tambahkan query parameter.
    el.stream.src = STREAM_URL;
}

function scheduleStreamRetry() {
    if (streamRetryTimer) return;

    streamRetryTimer = setTimeout(() => {
        streamRetryTimer = null;
        startStream();
    }, 3000);
}

function updateStream(data) {
    if (!data.running) {
        stopStream();
        return;
    }

    if (!data.camera_connected) {
        if (streamActive) {
            stopStream();
        }

        showMessage(
            data.camera_error || "Menunggu koneksi CCTV..."
        );

        setText(
            el.streamStatus,
            "CCTV belum terhubung."
        );

        return;
    }

    if (!streamActive) {
        startStream();
    }
}

// =====================================================
// 4. LATEST RECOGNITION
// =====================================================

function renderLatestRecognition(faces) {
    const recognizedFaces = faces.filter(
        face => Boolean(face.name)
    );

    if (!recognizedFaces.length) {
        setText(
            el.latestName,
            "Belum ada identifikasi"
        );

        setText(
            el.latestDetail,
            "Menunggu hasil Face Recognition."
        );

        setText(
            el.latestConfidence,
            "--"
        );

        setText(
            el.latestAvatar?.querySelector("span"),
            "?"
        );

        return;
    }

    // Prioritaskan wajah dengan similarity tertinggi
    const latest = recognizedFaces.reduce(
        (best, face) =>
            Number(face.similarity) > Number(best.similarity)
                ? face
                : best
    );

    setText(el.latestName, latest.name);

    setText(
        el.latestDetail,
        `Track ID #${latest.id} · ${dataCameraLabel}`
    );

    setText(
        el.latestConfidence,
        formatPercent(latest.similarity)
    );

    setText(
        el.latestAvatar?.querySelector("span"),
        latest.name.charAt(0).toUpperCase()
    );
}

let dataCameraLabel = "CCTV-LANTAI1";

// =====================================================
// 5. RECOGNITION ACTIVITY
// =====================================================

function addActivity(face) {
    if (!face.name) return;

    const key = `${face.id}:${face.name}`;

    // Hindari aktivitas duplikat dari polling yang sama
    if (activityKeys.has(key)) return;

    activityKeys.add(key);

    activityHistory.unshift({
        id: face.id,
        name: face.name,
        similarity: face.similarity,
        time: getTime()
    });

    if (activityHistory.length > MAX_ACTIVITY) {
        const removed = activityHistory.pop();

        activityKeys.delete(
            `${removed.id}:${removed.name}`
        );
    }
}

function renderActivity() {
    if (!el.activity) return;

    el.activity.replaceChildren();

    setText(
        el.activityCount,
        activityHistory.length
    );

    if (!activityHistory.length) {
        const empty = document.createElement("div");
        empty.className = "activity-empty";

        const icon = document.createElement("div");
        icon.className = "empty-icon";
        icon.textContent = "◉";

        const paragraph = document.createElement("p");
        paragraph.textContent =
            "Belum ada aktivitas pengenalan wajah.";

        const small = document.createElement("small");
        small.textContent =
            "Hasil identifikasi akan muncul di sini.";

        empty.append(icon, paragraph, small);
        el.activity.appendChild(empty);

        return;
    }

    activityHistory.forEach(item => {
        const container = document.createElement("div");
        container.className = "recognition-activity-item";

        const avatar = document.createElement("div");
        avatar.className = "activity-avatar";
        avatar.textContent = item.name.charAt(0).toUpperCase();

        const person = document.createElement("div");
        person.className = "activity-person";

        const name = document.createElement("strong");
        name.textContent = item.name;

        const detail = document.createElement("small");
        detail.textContent =
            `Track #${item.id} · ${item.time} · ${formatPercent(item.similarity)}`;

        person.append(name, detail);

        const result = document.createElement("span");
        result.className = "activity-result";
        result.textContent = "KNOWN";

        container.append(avatar, person, result);

        el.activity.appendChild(container);
    });
}

// =====================================================
// 6. MODEL INFORMATION
// =====================================================

function renderModelInformation(data) {
    setText(
        el.detectionModel,
        "InsightFace Detection"
    );

    setText(
        el.recognitionModel,
        "ArcFace"
    );

    setText(
        el.backend,
        "ONNX Runtime"
    );

    setText(
        el.device,
        Number(data.gpu_id) >= 0
            ? `NVIDIA CUDA · GPU ${data.gpu_id}`
            : "CPU"
    );
}

// =====================================================
// 7. MAIN MONITORING
// =====================================================

async function updateMonitoring() {
    if (polling) return;

    polling = true;

    try {
        const data = await fetchStatus();

        dataCameraLabel =
            data.camera_label || "CCTV-LANTAI1";

        const faces = Array.isArray(data.faces)
            ? data.faces
            : [];

        renderEngine(data);
        renderStatistics(data);
        updateStream(data);
        renderLatestRecognition(faces);
        renderModelInformation(data);

        faces.forEach(addActivity);
        renderActivity();

    } catch (error) {
        console.error(
            "[GateWatch] Face Recognition:",
            error
        );

        setText(
            el.engineStatus,
            "Service Unavailable"
        );

        setText(
            el.engineDescription,
            "Tidak dapat mengambil data dari Face Recognition Viewer."
        );

        if (el.engineBadge) {
            el.engineBadge.className =
                "engine-status-badge offline";

            el.engineBadge.lastChild.textContent =
                "UNAVAILABLE";
        }

        stopStream();

    } finally {
        polling = false;
    }
}


// =====================================================
// 8. INITIALIZATION
// =====================================================

async function initialize() {
    try {
        // Load sidebar only.
        await loadComponent(
            "sidebar-container",
            "components/sidebar.html"
        );

        // Initialize active navigation.
        initSidebar();

        console.log(
            "[GateWatch] Sidebar initialized successfully."
        );

    } catch (error) {
        console.error(
            "[GateWatch] Sidebar initialization:",
            error
        );
    }

    // Start Face Recognition monitoring.
    await updateMonitoring();

    // Update monitoring every 2 seconds.
    window.setInterval(
        updateMonitoring,
        POLLING_INTERVAL
    );
}

// Start application.
initialize();

// Cleanup stream when leaving page.
window.addEventListener("beforeunload", stopStream);