(() => {
  "use strict";

  // Adapted from the Graphcore Research blog; see README.md.
  const DATA_URL = "assets/data/centroids.json";
  const SIGN_PATTERNS = [
    [1, 1, 1],
    [1, 1, -1],
    [1, -1, 1],
    [1, -1, -1],
    [-1, 1, 1],
    [-1, 1, -1],
    [-1, -1, 1],
    [-1, -1, -1]
  ];
  const SIGN_LABELS = SIGN_PATTERNS.map(signLabel);
  const CENTROID_BLUE = "#4c78a8";
  const ORTHANT_COLORS = {
    "+++": CENTROID_BLUE,
    "++-": "#f58518",
    "+-+": "#54a24b",
    "+--": "#e45756",
    "-++": "#72b7b2",
    "-+-": "#b279a2",
    "--+": "#ff9da6",
    "---": "#9d755d"
  };
  const DEFAULT_YAW = -2.35;
  const DEFAULT_PITCH = 0.42;

  const CSS = `
.centroid-cloud {
  margin: 1.2rem 0 1.8rem;
  padding: 0.7rem;
  border: 1px solid #dedede;
  border-radius: 0.25rem;
  background: #fafafa;
}

.centroid-cloud .s3d8-cloud-controls {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.55rem 1rem;
  margin-bottom: 0.55rem;
}

.centroid-cloud .s3d8-cloud-control {
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.centroid-cloud .s3d8-cloud-label {
  color: #707070;
  font-size: 0.68rem;
  font-weight: 700;
  line-height: 1;
  text-transform: uppercase;
}

.centroid-cloud .s3d8-cloud-segmented {
  display: inline-flex;
  overflow: hidden;
  border: 1px solid #d7d7d7;
  border-radius: 0.25rem;
  background: #fff;
}

.centroid-cloud .s3d8-cloud-button {
  appearance: none;
  margin: 0;
  padding: 0.26rem 0.52rem;
  border: 0;
  border-right: 1px solid #d7d7d7;
  background: transparent;
  color: #555;
  cursor: pointer;
  font: inherit;
  font-size: 0.8rem;
  line-height: 1.2;
}

.centroid-cloud .s3d8-cloud-button:last-child {
  border-right: 0;
}

.centroid-cloud .s3d8-cloud-button[aria-pressed="true"] {
  background: #ffedf1;
  box-shadow: inset 0 -0.11rem 0 #ff6d81;
  color: #c93f54;
  font-weight: 700;
}

.centroid-cloud .s3d8-cloud-button:focus-visible {
  outline: 2px solid #ff6d81;
  outline-offset: -2px;
}

.centroid-cloud .s3d8-cloud-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 1rem;
  margin: 0 0 0.25rem;
  color: #666;
  font-size: 0.82rem;
}

.centroid-cloud .s3d8-cloud-meta strong {
  color: #444;
}

.centroid-cloud .s3d8-cloud-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 0.25rem 0.7rem;
  margin: 0.15rem 0 0.45rem;
  color: #666;
  font-size: 0.76rem;
}

.centroid-cloud .s3d8-cloud-legend-item {
  display: inline-flex;
  align-items: center;
  gap: 0.25rem;
}

.centroid-cloud .s3d8-cloud-swatch {
  width: 0.55rem;
  height: 0.55rem;
  border: 1px solid rgba(0, 0, 0, 0.12);
  border-radius: 50%;
}

.centroid-cloud .s3d8-cloud-stage {
  position: relative;
  width: 100%;
  height: 520px;
  overflow: hidden;
  border: 1px solid #e4e4e4;
  border-radius: 0.2rem;
  background: #fff;
}

.centroid-cloud .s3d8-cloud-canvas {
  display: block;
  width: 100%;
  height: 100%;
  touch-action: none;
  cursor: grab;
}

.centroid-cloud .s3d8-cloud-canvas:active {
  cursor: grabbing;
}

.centroid-cloud .s3d8-cloud-tooltip {
  position: absolute;
  max-width: 14rem;
  padding: 0.28rem 0.38rem;
  border: 1px solid rgba(0, 0, 0, 0.16);
  border-radius: 0.18rem;
  background: rgba(255, 255, 255, 0.96);
  box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.12);
  color: #4b4b4b;
  font-size: 0.68rem;
  line-height: 1.35;
  pointer-events: none;
  white-space: nowrap;
}

.centroid-cloud .s3d8-cloud-help {
  margin: 0.35rem 0 0;
  color: #777;
  font-size: 0.72rem;
  font-style: italic;
}

.centroid-cloud .s3d8-cloud-error {
  margin: 0;
  color: #666;
  font-size: 0.85rem;
  font-style: italic;
}

@media (max-width: 640px) {
  .centroid-cloud {
    padding: 0.55rem;
  }

  .centroid-cloud .s3d8-cloud-control {
    width: 100%;
    justify-content: space-between;
  }

  .centroid-cloud .s3d8-cloud-segmented {
    flex: 1;
  }

  .centroid-cloud .s3d8-cloud-button {
    flex: 1;
    padding-inline: 0.28rem;
  }

  .centroid-cloud .s3d8-cloud-stage {
    height: 380px;
  }
}
`;

  function signLabel(signs) {
    return signs.map((sign) => sign < 0 ? "-" : "+").join("");
  }

  function pointSignLabel(point) {
    return signLabel(point.map((value) => value < 0 ? -1 : 1));
  }

  function uniqueValues(records, key) {
    const seen = new Set();
    const values = [];
    for (const record of records) {
      if (!seen.has(record[key])) {
        seen.add(record[key]);
        values.push(record[key]);
      }
    }
    return values;
  }

  function findRecord(records, distribution, method) {
    return records.find((record) =>
      record.distribution === distribution && record.method === method);
  }

  function round3(value) {
    return Number(value).toFixed(3);
  }

  function expandRecord(record) {
    if (record.method === "S3D8") {
      const points = [];
      record.centroids.forEach((centroid, sourceIndex) => {
        for (const signs of SIGN_PATTERNS) {
          const point = centroid.map((value, axis) => value * signs[axis]);
          points.push({
            point,
            sign: signLabel(signs),
            source: sourceIndex
          });
        }
      });
      return points;
    }

    if (record.method === "1D(8/3)") {
      const scalars = record.centroids.map((centroid) => centroid[0]);
      const points = [];
      scalars.forEach((x, i) => {
        scalars.forEach((y, j) => {
          scalars.forEach((z, k) => {
            const point = [x, y, z];
            points.push({
              point,
              sign: pointSignLabel(point),
              source: `${i},${j},${k}`
            });
          });
        });
      });
      return points;
    }

    return record.centroids.map((point, sourceIndex) => ({
      point,
      sign: pointSignLabel(point),
      source: sourceIndex
    }));
  }

  function distributionRanges(records) {
    const ranges = {};
    for (const distribution of uniqueValues(records, "distribution")) {
      let maxAbs = 0;
      for (const record of records) {
        if (record.distribution !== distribution) {
          continue;
        }
        for (const item of expandRecord(record)) {
          for (const value of item.point) {
            maxAbs = Math.max(maxAbs, Math.abs(value));
          }
        }
      }
      ranges[distribution] = Math.max(1, maxAbs * 1.06);
    }
    return ranges;
  }

  function colorFor(record, item) {
    if (!record.signed) {
      return CENTROID_BLUE;
    }
    return ORTHANT_COLORS[item.sign];
  }

  function hexToRgb(hex) {
    const value = parseInt(hex.slice(1), 16);
    return {
      r: (value >> 16) & 255,
      g: (value >> 8) & 255,
      b: value & 255
    };
  }

  function mixColor(a, b, t) {
    const aa = hexToRgb(a);
    const bb = hexToRgb(b);
    const r = Math.round(aa.r * (1 - t) + bb.r * t);
    const g = Math.round(aa.g * (1 - t) + bb.g * t);
    const blue = Math.round(aa.b * (1 - t) + bb.b * t);
    return `rgb(${r}, ${g}, ${blue})`;
  }

  function depthColor(record, item, minDepth, maxDepth) {
    const span = Math.max(1e-6, maxDepth - minDepth);
    const far = (item.depth - minDepth) / span;
    const fog = 0.9 * far;
    return mixColor(colorFor(record, item), "#d8d8d8", fog);
  }

  function depthRadius(item, minDepth, maxDepth) {
    const span = Math.max(1e-6, maxDepth - minDepth);
    const far = (item.depth - minDepth) / span;
    return 7.2 - 2.8 * far;
  }

  function rotatePoint(point, camera) {
    const [x, y, z] = point;
    const cy = Math.cos(camera.yaw);
    const sy = Math.sin(camera.yaw);
    const cp = Math.cos(camera.pitch);
    const sp = Math.sin(camera.pitch);
    const x1 = cy * x - sy * y;
    const y1 = sy * x + cy * y;
    const z1 = z;
    const y2 = cp * y1 - sp * z1;
    const z2 = sp * y1 + cp * z1;
    return [x1, y2, z2];
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function defaultZoomForDistribution(distribution) {
    return distribution === "Normal" ? 0.6 : 1;
  }

  function renderLegend(container, record) {
    const items = record.signed
      ? SIGN_LABELS.map((label) => [label, ORTHANT_COLORS[label]])
      : [["centroids", CENTROID_BLUE]];
    container.textContent = "";
    for (const [label, color] of items) {
      const item = document.createElement("span");
      item.className = "s3d8-cloud-legend-item";
      const swatch = document.createElement("span");
      swatch.className = "s3d8-cloud-swatch";
      swatch.style.backgroundColor = color;
      const text = document.createElement("span");
      text.textContent = label;
      item.appendChild(swatch);
      item.appendChild(text);
      container.appendChild(item);
    }
  }

  function renderControls(container, values, selected, onSelect) {
    container.textContent = "";
    for (const value of values) {
      const button = document.createElement("button");
      button.className = "s3d8-cloud-button";
      button.type = "button";
      button.textContent = value;
      button.setAttribute("aria-pressed", value === selected ? "true" : "false");
      button.addEventListener("click", () => onSelect(value));
      container.appendChild(button);
    }
  }

  function injectStyle() {
    if (document.getElementById("s3d8-centroid-cloud-style")) {
      return;
    }
    const style = document.createElement("style");
    style.id = "s3d8-centroid-cloud-style";
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  function showError(root, message) {
    root.innerHTML = `<p class="s3d8-cloud-error">${message}</p>`;
  }

  async function initialize(root) {
    if (root === null) {
      return;
    }
    injectStyle();

    let records;
    try {
      const response = await fetch(DATA_URL);
      records = await response.json();
    } catch (error) {
      showError(root, `Interactive centroid plot data failed to load: ${error}`);
      return;
    }

    const distributions = uniqueValues(records, "distribution");
    const methods = uniqueValues(records, "method");
    const ranges = distributionRanges(records);
    const defaultDistribution = distributions.includes("Normal") ? "Normal" : distributions[0];
    const state = {
      distribution: defaultDistribution,
      method: root.dataset.method,
      yaw: DEFAULT_YAW,
      pitch: DEFAULT_PITCH,
      zoom: defaultZoomForDistribution(defaultDistribution),
      points: [],
      projected: [],
      record: null
    };
    const pointers = new Map();
    let lastPinchDistance = null;

    root.innerHTML = `
<div class="s3d8-cloud-controls">
  <div class="s3d8-cloud-control">
    <span class="s3d8-cloud-label">Distribution</span>
    <div class="s3d8-cloud-segmented" data-control="distribution"></div>
  </div>
  <div class="s3d8-cloud-control">
    <span class="s3d8-cloud-label">Method</span>
    <div class="s3d8-cloud-segmented" data-control="method"></div>
  </div>
</div>
<div class="s3d8-cloud-meta">
  <span><strong>RMS error:</strong> <span data-field="rmse"></span></span>
  <span><strong>Points:</strong> <span data-field="points"></span></span>
</div>
<div class="s3d8-cloud-legend" data-field="legend"></div>
<div class="s3d8-cloud-stage">
  <canvas class="s3d8-cloud-canvas" aria-label="Interactive 3D centroid point cloud"></canvas>
  <div class="s3d8-cloud-tooltip" hidden></div>
</div>
<p class="s3d8-cloud-help">Drag to rotate. Scroll or pinch to zoom.</p>`;

    const distributionControl = root.querySelector('[data-control="distribution"]');
    const methodControl = root.querySelector('[data-control="method"]');
    const rmse = root.querySelector('[data-field="rmse"]');
    const pointCount = root.querySelector('[data-field="points"]');
    const legend = root.querySelector('[data-field="legend"]');
    const canvas = root.querySelector(".s3d8-cloud-canvas");
    const tooltip = root.querySelector(".s3d8-cloud-tooltip");
    const ctx = canvas.getContext("2d");
    if (ctx === null) {
      showError(root, "Interactive centroid plot could not create a canvas context.");
      return;
    }

    function screenProject(point, width, height, range) {
      const rotated = rotatePoint(point, state);
      const scale = Math.min(width, height) * 0.39 * state.zoom / range;
      return {
        x: width / 2 + rotated[0] * scale,
        y: height / 2 - rotated[2] * scale,
        depth: rotated[1]
      };
    }

    function drawAxis(width, height, range, start, end, label) {
      const a = screenProject(start, width, height, range);
      const b = screenProject(end, width, height, range);
      ctx.beginPath();
      ctx.moveTo(a.x, a.y);
      ctx.lineTo(b.x, b.y);
      ctx.stroke();
      ctx.fillText(label, b.x + 5, b.y - 5);
    }

    function draw() {
      tooltip.hidden = true;
      const rect = canvas.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) {
        return;
      }
      const dpr = window.devicePixelRatio || 1;
      const width = rect.width;
      const height = rect.height;
      if (canvas.width !== Math.round(width * dpr) ||
        canvas.height !== Math.round(height * dpr)) {
        canvas.width = Math.round(width * dpr);
        canvas.height = Math.round(height * dpr);
      }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, width, height);

      const range = ranges[state.record.distribution];
      ctx.lineWidth = 1;
      ctx.strokeStyle = "#d4d4d4";
      ctx.fillStyle = "#777";
      ctx.font = "11px Graphik, sans-serif";
      drawAxis(width, height, range, [-range, 0, 0], [range, 0, 0], "x");
      drawAxis(width, height, range, [0, -range, 0], [0, range, 0], "y");
      drawAxis(width, height, range, [0, 0, -range], [0, 0, range], "z");

      state.projected = state.points.map((item) => {
        const projected = screenProject(item.point, width, height, range);
        return { ...item, ...projected };
      }).sort((a, b) => b.depth - a.depth);
      const depths = state.projected.map((item) => item.depth);
      const minDepth = Math.min(...depths);
      const maxDepth = Math.max(...depths);

      for (const item of state.projected) {
        ctx.beginPath();
        ctx.arc(item.x, item.y, depthRadius(item, minDepth, maxDepth), 0, Math.PI * 2);
        ctx.fillStyle = depthColor(state.record, item, minDepth, maxDepth);
        ctx.fill();
        ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
        ctx.lineWidth = 0.85;
        ctx.stroke();
      }
    }

    function updateTooltip(clientX, clientY) {
      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;
      let best = null;
      let bestDistance = Infinity;
      for (const item of state.projected) {
        const dx = item.x - x;
        const dy = item.y - y;
        const distance = dx * dx + dy * dy;
        if (distance < bestDistance) {
          best = item;
          bestDistance = distance;
        }
      }
      if (best === null || bestDistance > 100) {
        tooltip.hidden = true;
        return;
      }
      tooltip.hidden = false;
      tooltip.innerHTML =
        `${state.record.distribution} / ${state.record.method}` +
        (state.record.signed ? `<br>orthant ${best.sign}` : "") +
        `<br>source ${best.source}` +
        `<br>x=${round3(best.point[0])}, y=${round3(best.point[1])}, z=${round3(best.point[2])}`;
      // Hit testing uses viewport pixels; CSS positioning uses unscaled slide pixels.
      const localX = x * canvas.clientWidth / rect.width;
      const localY = y * canvas.clientHeight / rect.height;
      const left = localX + 10 + tooltip.offsetWidth <= canvas.clientWidth
        ? localX + 10 : localX - 10 - tooltip.offsetWidth;
      const top = localY + 10 + tooltip.offsetHeight <= canvas.clientHeight
        ? localY + 10 : localY - 10 - tooltip.offsetHeight;
      tooltip.style.left = `${clamp(left, 0, Math.max(0, canvas.clientWidth - tooltip.offsetWidth))}px`;
      tooltip.style.top = `${clamp(top, 0, Math.max(0, canvas.clientHeight - tooltip.offsetHeight))}px`;
    }

    function setRecord() {
      const record = findRecord(records, state.distribution, state.method);
      if (record === undefined) {
        showError(root, `Missing centroid data for ${state.distribution} / ${state.method}.`);
        return;
      }
      state.record = record;
      state.points = expandRecord(record);
      rmse.textContent = Number(record.rmse).toFixed(4);
      pointCount.textContent = state.points.length.toString();
      renderLegend(legend, record);
      renderControls(distributionControl, distributions, state.distribution, (value) => {
        state.distribution = value;
        state.zoom = defaultZoomForDistribution(value);
        setRecord();
      });
      renderControls(methodControl, methods, state.method, (value) => {
        state.method = value;
        setRecord();
      });
      tooltip.hidden = true;
      draw();
    }

    canvas.addEventListener("pointerdown", (event) => {
      canvas.setPointerCapture(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      lastPinchDistance = null;
    });

    canvas.addEventListener("pointermove", (event) => {
      if (!pointers.has(event.pointerId)) {
        updateTooltip(event.clientX, event.clientY);
        return;
      }

      const previous = pointers.get(event.pointerId);
      pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

      if (pointers.size === 1) {
        state.yaw += (event.clientX - previous.x) * 0.01;
        state.pitch = clamp(state.pitch + (event.clientY - previous.y) * 0.01, -1.35, 1.35);
        tooltip.hidden = true;
        draw();
        return;
      }

      if (pointers.size === 2) {
        const values = Array.from(pointers.values());
        const dx = values[0].x - values[1].x;
        const dy = values[0].y - values[1].y;
        const distance = Math.hypot(dx, dy);
        if (lastPinchDistance !== null) {
          state.zoom = clamp(state.zoom * (distance / lastPinchDistance), 0.45, 3.2);
          draw();
        }
        lastPinchDistance = distance;
      }
    });

    canvas.addEventListener("pointerup", (event) => {
      pointers.delete(event.pointerId);
      lastPinchDistance = null;
    });

    canvas.addEventListener("pointercancel", (event) => {
      pointers.delete(event.pointerId);
      lastPinchDistance = null;
    });

    canvas.addEventListener("pointerleave", () => {
      tooltip.hidden = true;
    });

    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const factor = Math.exp(-event.deltaY * 0.001);
      state.zoom = clamp(state.zoom * factor, 0.45, 3.2);
      tooltip.hidden = true;
      draw();
    }, { passive: false });

    window.addEventListener("resize", draw);
    setRecord();
    // Reveal hides inactive slides; redraw when the plot becomes visible or is laid out for print.
    new ResizeObserver(draw).observe(canvas);
    root.selectDistribution = (distribution) => {
      state.distribution = distribution;
      state.yaw = DEFAULT_YAW;
      state.pitch = DEFAULT_PITCH;
      state.zoom = defaultZoomForDistribution(distribution);
      setRecord();
    };
    root.resetView = () => {
      state.yaw = DEFAULT_YAW;
      state.pitch = DEFAULT_PITCH;
      state.zoom = defaultZoomForDistribution(state.distribution);
      draw();
    };
  }

  window.initializeCentroidClouds = () => Promise.all(
    [...document.querySelectorAll(".centroid-cloud")].map(initialize)
  );
})();
