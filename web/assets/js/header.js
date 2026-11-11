export function initHeader() {
  const dot = document.getElementById("dot");
  const connText = document.getElementById("connText");
  const clock = document.getElementById("clock");

  function setConnection(isOnline) {
    if (!dot || !connText) return;

    dot.className = isOnline ? "dot ok" : "dot bad";
    connText.textContent = isOnline
      ? "Tersambung"
      : "Server tidak terjangkau";
  }

  function tick() {
    if (!clock) return;

    clock.textContent = new Date().toLocaleString("id-ID", {
      dateStyle: "full",
      timeStyle: "medium",
    });
  }

  tick();
  window.setInterval(tick, 1000);

  return { setConnection };
}
