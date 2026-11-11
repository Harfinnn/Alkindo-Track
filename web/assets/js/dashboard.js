function cap(value) {
  const text = String(value ?? "");
  return text.charAt(0) + text.slice(1).toLowerCase();
}

function createCell(text, className = "") {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = String(text ?? "");
  return cell;
}

function createRow(cells) {
  const row = document.createElement("tr");
  cells.forEach(({ text, cls = "" }) => row.appendChild(createCell(text, cls)));
  return row;
}

export function renderDashboard(data = {}) {
  const occupancy = document.getElementById("occ");
  if (occupancy) {
    occupancy.replaceChildren();

    Object.entries(data.occupancy || {}).forEach(([key, value]) => {
      occupancy.appendChild(createRow([
        { text: cap(key) },
        { text: value?.inside, cls: "big in" },
        { text: value?.outside, cls: "big out" },
      ]));
    });
  }

  const combined = document.getElementById("cnt");
  if (combined) {
    combined.replaceChildren();

    Object.entries(data.combined || {}).forEach(([key, value]) => {
      combined.appendChild(createRow([
        { text: cap(key) },
        { text: value?.in, cls: "in" },
        { text: value?.out, cls: "out" },
      ]));
    });
  }
}
