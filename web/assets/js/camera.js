function createElement(tag, className = "", text = undefined) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = String(text);
  return element;
}

export function renderCameras(cameras = []) {
  const container = document.getElementById("cams");
  if (!container) return;

  container.replaceChildren();

  cameras.forEach((camera) => {
    const box = createElement("div", "cam");
    box.appendChild(createElement(
      "b",
      "",
      `CCTV ${camera.id} – ${camera.gate}`
    ));

    const stateClass = camera.state === "NO SIGNAL" ? "NOSIGNAL" : camera.state;
    const stateText = camera.state === "IDLE"
      ? "Sepi"
      : camera.state === "AKTIF"
        ? "Aktif"
        : "Tidak ada sinyal";

    box.appendChild(createElement("span", `pill ${stateClass}`, stateText));

    let info = `Kamera ${camera.cam_fps} FPS`;
    if (camera.state === "AKTIF") {
      info += ` · AI ${camera.ai_fps} FPS (${camera.ai_ms} ms)`;
    }
    if (!camera.line) info += " · garis belum diatur";

    box.appendChild(createElement("small", "", info));
    container.appendChild(box);
  });
}
