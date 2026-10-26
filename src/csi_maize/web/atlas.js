"use strict";
const rawAtlasData = document.getElementById("atlas-data").textContent;
if (rawAtlasData.trim().startsWith("_")) {
  document.body.innerHTML =
    "<header><h1>Spatial atlas awaits populated data</h1><p>Generate the observation outputs to inspect this supporting atlas.</p></header>";
} else {
  const DATA = JSON.parse(rawAtlasData);
  const $ = (id) => document.getElementById(id);
  const defs = DATA.definitions;
  const sessions = [...new Set(DATA.examples.map((e) => e.meta.session))];
  let current = DATA.examples[DATA.default_example],
    loaded = {},
    renderToken = 0;
  const fmt = (x) =>
    x === null || x === undefined ? "Unavailable" : Number(x).toFixed(3);
  const esc = (x) =>
    String(x).replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
  sessions.forEach((s) =>
    $("session").add(
      new Option(s.split("/").pop().replace("capture_session_", ""), s),
    ),
  );
  defs.forEach((d) => $("index").add(new Option(d.id + " · " + d.label, d.id)));
  $("session").value = current.meta.session;
  $("index").value = "PRODUCT4";
  $("metrics").innerHTML = [
    ["41", "acquisitions"],
    ["1,640", "audited image pairs"],
    ["27", "spatial definitions"],
    [DATA.summary.displayed_frames, "displayed indexed views"],
  ]
    .map(([n, l]) => "<div><b>" + n + "</b><span>" + l + "</span></div>")
    .join("");
  $("coverageText").textContent =
    DATA.summary.frames_with_cross_support +
    " / 1640 views contain accepted cross-camera candidate support. " +
    DATA.summary.frames_with_40_joint_six_pixels +
    " views from " +
    DATA.summary.acquisitions_with_40_joint_six_pixels +
    " acquisitions retain at least40 joint-six-channel valid pixels under the declared guards.";
  $("coverageRows").innerHTML = DATA.summary.index_summary
    .map(
      (r) =>
        "<tr><td>" +
        esc(r.id) +
        "</td><td>" +
        r.qualified_frames +
        "</td><td>" +
        r.qualified_acquisitions +
        "</td><td>" +
        fmt(r.balanced_median) +
        "</td></tr>",
    )
    .join("");
  $("definitionsRows").innerHTML = defs
    .map(
      (d) =>
        "<tr><td>" +
        esc(d.id) +
        "<br>" +
        esc(d.label) +
        '</td><td class="formula">' +
        esc(d.formula) +
        "</td><td>" +
        esc(d.support) +
        "</td><td>[" +
        d.vmin +
        ", " +
        d.vmax +
        "]</td></tr>",
    )
    .join("");
  $("references").innerHTML =
    "<h3>Formula provenance and camera context</h3>" +
    DATA.references
      .map(
        (r) =>
          '<p class="resource"><a href="' +
          r.url +
          '">' +
          esc(r.title) +
          "</a><small>" +
          esc(r.role) +
          "</small></p>",
      )
      .join("");
  function populateViews(selected) {
    $("view").replaceChildren();
    DATA.examples
      .filter((e) => e.meta.session === $("session").value)
      .sort((a, b) => a.meta.view - b.meta.view)
      .forEach((e) =>
        $("view").add(
          new Option(
            "Index " +
              e.meta.view +
              " · " +
              e.meta.nominal_degrees +
              "° nominal" +
              (e.meta.selection.startsWith("highest") ? " · best signal" : ""),
            e.id,
          ),
        ),
      );
    if (selected) $("view").value = selected;
  }
  function profile() {
    let c = $("profile"),
      ctx = c.getContext("2d");
    c.width = 1100;
    c.height = 250;
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, c.width, c.height);
    const ff = DATA.frames
      .filter((f) => f.session === current.meta.session)
      .sort((a, b) => a.view - b.view);
    const max = Math.max(1, ...ff.map((f) => f.cross_support_pixels));
    const x = (v) => 65 + (v / 39) * 1000,
      y = (v) => 210 - (v / max) * 175;
    ctx.strokeStyle = "#ccd3c8";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(65, 30);
    ctx.lineTo(65, 210);
    ctx.lineTo(1065, 210);
    ctx.stroke();
    ctx.fillStyle = "#52665d";
    ctx.font = "13px \"Times New Roman\", Times, serif";
    ctx.fillText(max + " px", 5, 35);
    ctx.fillText("0", 35, 213);
    [0, 10, 20, 30, 39].forEach((v) => ctx.fillText(v, x(v) - 5, 235));
    for (const [key, color] of [
      ["cross_support_pixels", "#ba8627"],
      ["joint_six_valid_pixels", "#12786f"],
    ]) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ff.forEach((f, i) =>
        i ? ctx.lineTo(x(f.view), y(f[key])) : ctx.moveTo(x(f.view), y(f[key])),
      );
      ctx.stroke();
    }
    ctx.strokeStyle = "#14352e";
    ctx.setLineDash([4, 5]);
    ctx.beginPath();
    ctx.moveTo(x(current.meta.view), 22);
    ctx.lineTo(x(current.meta.view), 210);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  function paint() {
    if (!loaded.rgb) return;
    const c = $("overlay"),
      ctx = c.getContext("2d"),
      a = Number($("alpha").value) / 100,
      mode = $("mode").value;
    c.width = loaded.rgb.width;
    c.height = loaded.rgb.height;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.globalAlpha = 1;
    ctx.globalCompositeOperation = "source-over";
    if (mode === "blend") {
      ctx.globalAlpha = 1 - a;
      ctx.drawImage(loaded.rgb, 0, 0);
      ctx.globalCompositeOperation = "lighter";
      ctx.globalAlpha = a;
      ctx.drawImage(loaded.noir_registered, 0, 0);
      ctx.globalAlpha = 1;
      ctx.globalCompositeOperation = "source-over";
    } else if (mode === "wipe") {
      ctx.drawImage(loaded.rgb, 0, 0);
      ctx.save();
      ctx.beginPath();
      ctx.rect(c.width * a, 0, c.width * (1 - a), c.height);
      ctx.clip();
      ctx.drawImage(loaded.noir_registered, 0, 0);
      ctx.restore();
      ctx.strokeStyle = "#e9bc49";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(c.width * a, 0);
      ctx.lineTo(c.width * a, c.height);
      ctx.stroke();
    } else ctx.drawImage(loaded[mode], 0, 0);
    $("overlaycaption").textContent =
      mode === "blend"
        ? "Registered blend · " +
          Math.round(a * 100) +
          "% NoIR · segmented projections on black"
        : mode === "wipe"
          ? "Registered wipe · RGB left, NoIR right"
          : mode === "overlap"
            ? "Silhouette classes: RGB only, NoIR only and overlap"
            : "Gold: accepted candidate interiors used for cross-camera calculation; grey: other segmented tissue";
  }
  function map() {
    const d = defs.find((d) => d.id === $("index").value),
      s = current.stats[d.id];
    $("mapView").src = current.maps[d.id];
    $("mapcaption").textContent = d.id + " · " + d.label;
    $("formula").textContent = d.formula;
    $("lo").textContent = d.vmin;
    $("hi").textContent = d.vmax;
    $("mid").textContent = (d.vmin + d.vmax) / 2;
    $("stats").innerHTML = [
      ["Qualified map median", fmt(s.median)],
      ["Valid / eligible pixels", s.valid_pixels + " / " + s.eligible_pixels],
      ["Spatial 10th–90th percentile", fmt(s.q10) + " to " + fmt(s.q90)],
      ["RGB-grid silhouette IoU", fmt(current.meta.rgb_grid_iou)],
      ["Pixels beyond display range", s.display_below + s.display_above],
    ]
      .map(([label, value]) => "<div>" + label + "<b>" + value + "</b></div>")
      .join("");
    $("qualification").textContent =
      "Support family: " +
      d.support +
      ". " +
      (s.qualified
        ? "At least40 valid pixels; distribution is descriptive, not a confidence interval."
        : "Fewer than40 valid pixels: summary withheld; sparse valid map pixels may remain visible.") +
      " Six-channel completeness is not required for formulas using fewer channels.";
  }
  async function render() {
    const token = ++renderToken;
    current = DATA.examples.find((e) => e.id === $("view").value);
    const m = current.meta;
    $("identity").textContent =
      m.session.split("/").pop() +
      " · filename index " +
      m.view +
      " · " +
      m.selection;
    $("phase").textContent =
      "Nominal phase diagnostic best shift: " +
      m.phase_best_shift +
      " indices. " +
      (m.phase_best_shift !== 0
        ? "This acquisition did not favour same-index alignment in that audit. "
        : "") +
      "The displayed pair retains its original filename index; no measured servo angle or synchronized tissue correspondence is implied.";
    for (const k of ["rgb", "noir_native", "noir_registered"])
      $(k).src = current.images[k];
    $("provenance").textContent = JSON.stringify(m, null, 2);
    map();
    profile();
    $("tiles").innerHTML = defs
      .map(
        (d) =>
          '<figure><button data-index="' +
          d.id +
          '"><img loading="lazy" src="' +
          current.maps[d.id] +
          '" alt="' +
          esc(d.label) +
          ' map"><figcaption><b>' +
          d.id +
          "</b> · " +
          d.support +
          " support<br>Range [" +
          d.vmin +
          ", " +
          d.vmax +
          "] · valid " +
          current.stats[d.id].valid_pixels +
          " px</figcaption></button></figure>",
      )
      .join("");
    $("tiles")
      .querySelectorAll("button")
      .forEach(
        (b) =>
          (b.onclick = () => {
            $("index").value = b.dataset.index;
            map();
            $("mapView").scrollIntoView({
              behavior: "smooth",
              block: "center",
            });
          }),
      );
    const entries = await Promise.all(
      ["rgb", "noir_registered", "overlap", "support"].map(
        (k) =>
          new Promise((resolve, reject) => {
            const im = new Image();
            im.onload = () => resolve([k, im]);
            im.onerror = reject;
            im.src = current.images[k];
          }),
      ),
    );
    if (token !== renderToken) return;
    loaded = Object.fromEntries(entries);
    paint();
  }
  $("session").onchange = () => {
    populateViews();
    render();
  };
  $("view").onchange = render;
  $("index").onchange = map;
  $("mode").onchange = paint;
  $("alpha").oninput = paint;
  function step(n) {
    const s = $("view");
    s.selectedIndex =
      (s.selectedIndex + n + s.options.length) % s.options.length;
    render();
  }
  $("previous").onclick = () => step(-1);
  $("next").onclick = () => step(1);
  $("download").onclick = () => {
    const text = document.querySelector("#methods pre").textContent;
    if (window.parent !== window) {
      window.parent.postMessage({type: "csi-maize-investigation-export", text}, "*");
      return;
    }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(
      new Blob([text], {
        type: "text/plain;charset=utf-8",
      }),
    );
    a.download = "CSI_maize_spatial_fusion_methods.txt";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  populateViews(current.id);
  render();
}
