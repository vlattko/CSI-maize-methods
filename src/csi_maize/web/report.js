"use strict";

// All arrays are embedded; no network or filesystem requests are made.
const evidence = JSON.parse(
  document.getElementById("architecture-data").textContent,
);
const palette = [
  "#9d6a00",
  "#008573",
  "#b94268",
  "#3560a8",
  "#82459f",
  "#486e48",
];

function renderModel(mesh) {
  const canvas = document.getElementById("mesh-view");
  if (!canvas || !mesh) return;
  const select = document.getElementById("organ-select");
  mesh.organs.forEach((organ, index) =>
    select.add(new Option(organ.label, String(index))),
  );
  document.getElementById("mesh-note").textContent =
    `${mesh.organs.length} source-assisted local organ hypotheses · ${mesh.vertices.length.toLocaleString()} vertices · ${mesh.faces.length.toLocaleString()} triangles · arbitrary processing-scale gauge`;

  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  mesh.vertices.forEach((vertex) =>
    vertex.forEach((value, axis) => {
      lo[axis] = Math.min(lo[axis], value);
      hi[axis] = Math.max(hi[axis], value);
    }),
  );
  const centre = lo.map((value, axis) => (value + hi[axis]) / 2);
  const span = Math.max(1e-12, ...hi.map((value, axis) => value - lo[axis]));
  const vertices = mesh.vertices.map((vertex) =>
    vertex.map((value, axis) => (value - centre[axis]) / span),
  );
  const organIds = new Int32Array(vertices.length).fill(mesh.organs.length);
  mesh.organs.forEach((organ, index) =>
    organIds.fill(index, organ.start, organ.start + organ.count),
  );
  const faces = mesh.faces.map((indices) => ({
    indices,
    organ: organIds[indices[0]],
  }));
  let yaw = 0.35;
  let pitch = -0.12;
  let zoom = 1;
  let drag = null;
  let queued = false;

  function paint() {
    queued = false;
    const rect = canvas.getBoundingClientRect();
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.max(1, Math.round(rect.width * ratio));
    canvas.height = Math.max(1, Math.round(rect.height * ratio));
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, rect.width, rect.height);
    const factor = Math.min(rect.width * 0.82, rect.height * 0.83) * zoom;
    const rotated = vertices.map(([x, y, z]) => {
      const rx = Math.cos(yaw) * x + Math.sin(yaw) * z;
      const rz = -Math.sin(yaw) * x + Math.cos(yaw) * z;
      const ry = Math.cos(pitch) * y - Math.sin(pitch) * rz;
      const depth = Math.sin(pitch) * y + Math.cos(pitch) * rz;
      return [
        rect.width / 2 + rx * factor,
        rect.height / 2 - ry * factor,
        depth,
      ];
    });
    const selected = select.value === "all" ? null : Number(select.value);
    const visible = faces.filter(
      (face) =>
        selected === null ||
        face.organ === selected ||
        face.organ === mesh.organs.length,
    );
    visible.sort(
      (a, b) =>
        a.indices.reduce((sum, i) => sum + rotated[i][2], 0) -
        b.indices.reduce((sum, i) => sum + rotated[i][2], 0),
    );
    visible.forEach((face) => {
      const points = face.indices.map((index) => rotated[index]);
      context.beginPath();
      context.moveTo(points[0][0], points[0][1]);
      context.lineTo(points[1][0], points[1][1]);
      context.lineTo(points[2][0], points[2][1]);
      context.closePath();
      context.fillStyle = palette[face.organ % palette.length];
      context.fill();
    });
    context.fillStyle = "#53665d";
    context.font = "12px \"Times New Roman\", Times, serif";
    context.fillText(
      "Display orbit only · fitted acquisition camera remains fixed",
      18,
      rect.height - 18,
    );
  }
  function requestPaint() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(paint);
    }
  }
  canvas.addEventListener("pointerdown", (event) => {
    drag = [event.clientX, event.clientY];
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    yaw += (event.clientX - drag[0]) * 0.008;
    pitch = Math.max(
      -1.4,
      Math.min(1.4, pitch + (event.clientY - drag[1]) * 0.008),
    );
    drag = [event.clientX, event.clientY];
    requestPaint();
  });
  ["pointerup", "pointercancel", "lostpointercapture"].forEach((name) =>
    canvas.addEventListener(name, () => {
      drag = null;
    }),
  );
  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      zoom = Math.max(
        0.35,
        Math.min(3, zoom * Math.exp(-event.deltaY * 0.001)),
      );
      requestPaint();
    },
    { passive: false },
  );
  select.addEventListener("change", requestPaint);
  document.getElementById("reset-view").addEventListener("click", () => {
    yaw = 0.35;
    pitch = -0.12;
    zoom = 1;
    requestPaint();
  });
  window.addEventListener("resize", requestPaint);
  requestPaint();
}

function renderResponses(rows) {
  const canvas = document.getElementById("response-chart");
  if (!canvas || !rows.length) return;
  const resolution = Math.max(...rows.map((row) => row.resolution));
  const baseline = rows.filter(
    (row) => row.candidate === 0 && row.resolution === resolution,
  );
  if (!baseline.length) return;
  const modes = [...new Set(baseline.map((row) => row.mode))];
  function paint() {
    const width = canvas.getBoundingClientRect().width;
    const stacked = width < 650;
    const height = stacked ? modes.length * 250 : 320;
    const ratio = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * ratio);
    canvas.height = height * ratio;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    context.scale(ratio, ratio);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    const bound =
      Math.max(0.01, ...baseline.map((row) => Math.abs(row.receiver))) * 1.15;
    modes.forEach((mode, index) => {
      const group = baseline
        .filter((row) => row.mode === mode)
        .sort((a, b) => a.angle - b.angle);
      const top = stacked ? index * 250 : 0;
      const left = stacked ? 55 : 55 + (index * width) / modes.length;
      const panelWidth = (stacked ? width : width / modes.length) - 85;
      const angles = group.map((row) => row.angle);
      const minAngle = Math.min(...angles);
      const maxAngle = Math.max(...angles);
      const x = (angle) =>
        left +
        ((angle - minAngle) / Math.max(1, maxAngle - minAngle)) * panelWidth;
      const y = (value) =>
        top + (stacked ? 135 : 160) - (value / bound) * (stacked ? 80 : 105);
      context.strokeStyle = "#cbd5c8";
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(left, y(0));
      context.lineTo(left + panelWidth, y(0));
      context.stroke();
      context.strokeStyle = index ? "#b28228" : "#08756b";
      context.lineWidth = 2.5;
      context.beginPath();
      group.forEach((row, i) =>
        i
          ? context.lineTo(x(row.angle), y(row.receiver))
          : context.moveTo(x(row.angle), y(row.receiver)),
      );
      context.stroke();
      context.fillStyle = context.strokeStyle;
      group.forEach((row) => {
        context.beginPath();
        context.arc(x(row.angle), y(row.receiver), 3, 0, Math.PI * 2);
        context.fill();
      });
      context.fillStyle = "#163e35";
      context.font = "13px \"Times New Roman\", Times, serif";
      context.fillText(
        mode.replace("_degrees", "") + " · signed receiver fraction",
        left,
        top + 25,
      );
      context.fillText(`${minAngle}°`, left, top + (stacked ? 235 : 293));
      context.fillText(
        `${maxAngle}°`,
        left + panelWidth - 25,
        top + (stacked ? 235 : 293),
      );
      context.fillText("0", left - 18, y(0) + 4);
      context.fillText(bound.toFixed(2), left - 40, y(bound) + 4);
      context.fillText((-bound).toFixed(2), left - 40, y(-bound) + 4);
    });
  }
  window.addEventListener("resize", paint);
  paint();
}

renderModel(evidence.mesh);
renderResponses(evidence.reposition);

// Sandbox frames have an opaque origin. Export through this document instead.
window.addEventListener("message", (event) => {
  const frame = document.querySelector(
    'iframe[title="Supporting RGB and NoIR spatial atlas"]',
  );
  const request = event.data;
  if (
    !frame || event.source !== frame.contentWindow ||
    !request || request.type !== "csi-maize-investigation-export" ||
    typeof request.text !== "string" || request.text.length > 1000000
  ) return;
  const url = URL.createObjectURL(
    new Blob([request.text], {type: "text/plain;charset=utf-8"}),
  );
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "CSI_maize_spatial_fusion_methods.txt";
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
