
const icons = {
  success: "✓",
  error: "✕",
  warning: "!",
  info: "i",
};

const titles = {
  success: "Berhasil",
  error: "Terjadi Kesalahan",
  warning: "Perhatian",
  info: "Informasi",
};

export function showNotification(
  message,
  type = "info",
  duration = 4000
) {
  const container = document.getElementById(
    "notification-container"
  );

  if (!container) {
    console.warn("Notification container tidak ditemukan.");
    return;
  }

  const notification = document.createElement("div");
  notification.className = `notification ${type}`;

  const icon = document.createElement("div");
  icon.className = "notification-icon";
  icon.textContent = icons[type] || icons.info;

  const content = document.createElement("div");
  content.className = "notification-content";

  const title = document.createElement("h4");
  title.className = "notification-title";
  title.textContent = titles[type] || titles.info;

  const description = document.createElement("p");
  description.className = "notification-message";
  description.textContent = message;

  const progress = document.createElement("div");
  progress.className = "notification-progress";

  const progressBar = document.createElement("div");
  progressBar.className = "notification-progress-bar";
  progressBar.style.animationDuration = `${duration}ms`;

  progress.appendChild(progressBar);
  content.append(title, description, progress);

  const closeButton = document.createElement("button");
  closeButton.className = "notification-close";
  closeButton.type = "button";
  closeButton.textContent = "×";
  closeButton.setAttribute("aria-label", "Tutup notifikasi");

  notification.append(icon, content, closeButton);
  container.appendChild(notification);

  let timeout;

  const removeNotification = () => {
    clearTimeout(timeout);

    if (!notification.isConnected) return;

    notification.classList.add("removing");

    notification.addEventListener(
      "animationend",
      () => notification.remove(),
      { once: true }
    );

    // Fallback jika animasi tidak berjalan
    setTimeout(() => notification.remove(), 300);
  };

  closeButton.addEventListener("click", removeNotification);

  if (duration > 0) {
    timeout = setTimeout(removeNotification, duration);
  }

  return notification;
}


export function showConfirmation({
    title = "Konfirmasi",
    message = "Apakah Anda yakin?",
    confirmText = "Ya, Lanjutkan",
    cancelText = "Batal",
    type = "warning"
} = {}) {
    return new Promise((resolve) => {
        const overlay = document.createElement("div");
        overlay.className = "confirmation-overlay";

        const dialog = document.createElement("div");
        dialog.className = `confirmation-dialog ${type}`;

        const icon = document.createElement("div");
        icon.className = "confirmation-icon";
        icon.textContent = type === "error" ? "!" : "?";

        const heading = document.createElement("h3");
        heading.textContent = title;

        const description = document.createElement("p");
        description.textContent = message;

        const actions = document.createElement("div");
        actions.className = "confirmation-actions";

        const cancelButton = document.createElement("button");
        cancelButton.className = "confirmation-cancel";
        cancelButton.textContent = cancelText;

        const confirmButton = document.createElement("button");
        confirmButton.className = "confirmation-confirm";
        confirmButton.textContent = confirmText;

        actions.append(cancelButton, confirmButton);
        dialog.append(icon, heading, description, actions);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        const close = (result) => {
            overlay.remove();
            resolve(result);
        };

        cancelButton.addEventListener("click", () => close(false));
        confirmButton.addEventListener("click", () => close(true));

        overlay.addEventListener("click", (event) => {
            if (event.target === overlay) close(false);
        });

        const handleEscape = (event) => {
            if (event.key === "Escape") {
                document.removeEventListener("keydown", handleEscape);
                close(false);
            }
        };

        document.addEventListener("keydown", handleEscape);
        cancelButton.focus();
    });
}
