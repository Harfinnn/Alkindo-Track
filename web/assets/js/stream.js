
import { getStreamUrl } from "./api.js";

export function initStream() {
  const well = document.getElementById("well");
  const image = document.getElementById("stream");
  const status = document.getElementById("streamStatus");

  if (!well || !image) return;

  let retryTimer = null;
  let destroyed = false;

  function updateStatus(message) {
    if (status) status.textContent = message;
  }

  function startStream() {
    if (destroyed) return;

    updateStatus("Menghubungkan ke kamera...");
    image.src = getStreamUrl();
  }

  image.onload = () => {
    well.classList.remove("down");
    updateStatus("Kamera terhubung");

    if (retryTimer) {
      window.clearTimeout(retryTimer);
      retryTimer = null;
    }
  };

  image.onerror = () => {
    well.classList.add("down");
    updateStatus("Koneksi terputus. Mencoba kembali...");

    if (!retryTimer) {
      retryTimer = window.setTimeout(() => {
        retryTimer = null;
        startStream();
      }, 3000);
    }
  };

  startStream();

  return () => {
    destroyed = true;

    if (retryTimer) {
      window.clearTimeout(retryTimer);
    }

    image.removeAttribute("src");
  };
}
