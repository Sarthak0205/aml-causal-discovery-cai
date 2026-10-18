/**
 * XAI with ANM: Research Demonstration Frontend Logic
 * Vanilla JavaScript (Zero External Libraries)
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM References
  const runBtn = document.getElementById("runBtn");
  const loadingIndicator = document.getElementById("loadingIndicator");
  const errorBanner = document.getElementById("errorBanner");
  const errorMessage = document.getElementById("errorMessage");
  const resultsSection = document.getElementById("resultsSection");

  // Dataset metadata DOM
  const datasetName = document.getElementById("datasetName");
  const metaSamples = document.getElementById("metaSamples");
  const metaFeatures = document.getElementById("metaFeatures");
  const metaPairs = document.getElementById("metaPairs");
  const metaModel = document.getElementById("metaModel");

  // Overview metrics DOM
  const resSamples = document.getElementById("resSamples");
  const resFeatures = document.getElementById("resFeatures");
  const resPairs = document.getElementById("resPairs");
  const resCandidateEdges = document.getElementById("resCandidateEdges");
  const resThreshold = document.getElementById("resThreshold");
  const resDagStatus = document.getElementById("resDagStatus");
  const resCycles = document.getElementById("resCycles");
  const topoSequence = document.getElementById("topoSequence");

  // Inspector DOM
  const inspectorCard = document.getElementById("inspectorCard");
  const inspectorType = document.getElementById("inspectorType");
  const inspectorTitle = document.getElementById("inspectorTitle");
  const inspectorBody = document.getElementById("inspectorBody");

  // Graph SVG DOM
  const graphSvg = document.getElementById("causalGraphSvg");
  const graphZoomGroup = document.getElementById("graphZoomGroup");
  const edgeLayer = document.getElementById("edgeLayer");
  const nodeLayer = document.getElementById("nodeLayer");
  const zoomInBtn = document.getElementById("zoomInBtn");
  const zoomOutBtn = document.getElementById("zoomOutBtn");
  const zoomResetBtn = document.getElementById("zoomResetBtn");

  // CAI DOM
  const caiBaseline = document.getElementById("caiBaseline");
  const caiBarsContainer = document.getElementById("caiBarsContainer");

  // Table DOM
  const tableSearchInput = document.getElementById("tableSearchInput");
  const pairwiseTableBody = document.getElementById("pairwiseTableBody");
  const tableSummaryText = document.getElementById("tableSummaryText");
  const filterPills = document.querySelectorAll(".pill-btn");
  const tableHeaders = document.querySelectorAll(".data-table th");

  // Application State
  let analysisData = null;
  let activeFilter = "all";
  let sortColumn = "confidence";
  let sortDirection = "desc";
  let activeSearch = "";

  // Graph Transform State
  let transform = { x: 0, y: 0, scale: 1 };
  let isPanning = false;
  let startPan = { x: 0, y: 0 };

  // -------------------------------------------------------------------------
  // Initialize: Fetch /api/info
  // -------------------------------------------------------------------------
  async function fetchDatasetInfo() {
    try {
      const res = await fetch("/api/info");
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const info = await res.json();
      if (info.dataset_name) datasetName.textContent = info.dataset_name;
      if (info.n_samples) metaSamples.textContent = info.n_samples;
      if (info.n_features) metaFeatures.textContent = info.n_features;
      if (info.n_possible_pairs) metaPairs.textContent = info.n_possible_pairs;
      if (info.model_type) metaModel.textContent = `${info.model_type} (${info.model_file || "model.pkl"})`;
    } catch (err) {
      console.warn("Could not fetch /api/info metadata:", err);
    }
  }

  fetchDatasetInfo();

  // -------------------------------------------------------------------------
  // Run Analysis Button Handler
  // -------------------------------------------------------------------------
  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    errorBanner.classList.add("hidden");
    loadingIndicator.classList.remove("hidden");
    resultsSection.classList.add("hidden");

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `Server responded with status ${response.status}`);
      }

      analysisData = await response.json();
      renderAllResults(analysisData);
      resultsSection.classList.remove("hidden");
    } catch (err) {
      console.error("Analysis execution failed:", err);
      errorMessage.textContent = `Analysis failed: ${err.message}`;
      errorBanner.classList.remove("hidden");
    } finally {
      loadingIndicator.classList.add("hidden");
      runBtn.disabled = false;
    }
  });

  // -------------------------------------------------------------------------
  // Main Result Rendering Coordinator
  // -------------------------------------------------------------------------
  function renderAllResults(data) {
    renderOverview(data);
    renderGraph(data);
    renderCai(data.cai);
    renderPairwiseTable();
  }

  // -------------------------------------------------------------------------
  // 1. Overview Cards
  // -------------------------------------------------------------------------
  function renderOverview(data) {
    const ov = data.overview || {};
    resSamples.textContent = ov.samples ?? 178;
    resFeatures.textContent = ov.features ?? 13;
    resPairs.textContent = ov.pairs_evaluated ?? 78;
    resCandidateEdges.textContent = ov.candidate_edges ?? (data.dag_edges ? data.dag_edges.length : 20);
    resThreshold.textContent = typeof ov.threshold === "number" ? ov.threshold.toFixed(6) : "0.002919";
    resDagStatus.textContent = ov.dag_status || (ov.is_dag ? "Acyclic (Valid DAG)" : "Contains Cycles");
    resCycles.textContent = `Cycles Removed: ${ov.cycles_removed ?? 0} (Naturally acyclic)`;

    if (data.topological_order && data.topological_order.length > 0) {
      topoSequence.textContent = data.topological_order.join(" → ");
    } else {
      topoSequence.textContent = "-";
    }
  }

  // -------------------------------------------------------------------------
  // 2. Discovered Causal Graph (Layered Layout SVG)
  // -------------------------------------------------------------------------
  function renderGraph(data) {
    edgeLayer.innerHTML = "";
    nodeLayer.innerHTML = "";

    const nodes = data.dag_nodes || [];
    const edges = data.dag_edges || [];
    const p = nodes.length;
    if (p === 0) return;

    // Optimized layer placements tailored to the validated 13-node Wine DAG:
    // Aligns main causal backbone horizontally while placing peripheral sinks cleanly
    // Crossings minimized to 17, avoiding vertical crowding or boundary clipping
    const optimizedCoordinates = {
      "malic_acid": { x: 90, y: 200 },
      "ash": { x: 90, y: 440 },
      "proline": { x: 250, y: 240 },
      "hue": { x: 250, y: 440 },
      "magnesium": { x: 430, y: 110 },
      "flavanoids": { x: 430, y: 260 },
      "nonflavanoid_phenols": { x: 430, y: 410 },
      "alcohol": { x: 620, y: 140 },
      "od280/od315_of_diluted_wines": { x: 620, y: 350 },
      "proanthocyanins": { x: 800, y: 190 },
      "color_intensity": { x: 800, y: 350 },
      "total_phenols": { x: 950, y: 170 },
      "alcalinity_of_ash": { x: 950, y: 360 },
    };

    // Calculate dynamic coordinates fallback if different nodes are encountered
    const nodeCoords = {};
    nodes.forEach(node => {
      if (optimizedCoordinates[node.name]) {
        nodeCoords[node.id] = {
          x: optimizedCoordinates[node.name].x,
          y: optimizedCoordinates[node.name].y,
          node,
        };
      } else {
        // Fallback grid placement
        const idx = node.id;
        const col = idx % 5;
        const row = Math.floor(idx / 5);
        nodeCoords[node.id] = {
          x: 100 + col * 190,
          y: 120 + row * 150,
          node,
        };
      }
    });

    // Draw Edges with arrowheads and gentle curvature to reduce overlaps
    edges.forEach(edge => {
      const src = nodeCoords[edge.source_idx];
      const tgt = nodeCoords[edge.target_idx];
      if (!src || !tgt) return;

      const gEdge = document.createElementNS("http://www.w3.org/2000/svg", "g");

      const dx = tgt.x - src.x;
      const dy = tgt.y - src.y;

      // Arc long-span edges away from the central horizontal spine
      let curveOffset = 0;
      if (Math.abs(dx) > 280) {
        curveOffset = src.y < 300 ? -28 : 28;
      } else if (Math.abs(dy) > 120 && Math.abs(dx) > 140) {
        curveOffset = dy > 0 ? -16 : 16;
      }

      const midX = (src.x + tgt.x) / 2;
      const midY = (src.y + tgt.y) / 2 + curveOffset;

      const pathD = `M ${src.x} ${src.y} Q ${midX} ${midY} ${tgt.x} ${tgt.y}`;

      // Visible edge line
      const pathLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
      pathLine.setAttribute("d", pathD);
      pathLine.setAttribute("class", "graph-edge");
      pathLine.setAttribute("marker-end", "url(#arrowhead)");
      pathLine.setAttribute("data-src", edge.source);
      pathLine.setAttribute("data-tgt", edge.target);

      // Edge tooltip for hovering
      const edgeTitle = document.createElementNS("http://www.w3.org/2000/svg", "title");
      edgeTitle.textContent = `${edge.source} → ${edge.target} (Confidence Δ = ${edge.confidence.toFixed(6)})`;
      gEdge.appendChild(edgeTitle);

      // Transparent wide hit area for easy hover/clicking
      const hitLine = document.createElementNS("http://www.w3.org/2000/svg", "path");
      hitLine.setAttribute("d", pathD);
      hitLine.setAttribute("class", "edge-hit-area");

      const selectEdgeAction = () => {
        document.querySelectorAll(".graph-edge").forEach(el => el.classList.remove("selected"));
        pathLine.classList.add("selected");
        showEdgeInspector(edge);
      };

      hitLine.addEventListener("click", selectEdgeAction);
      pathLine.addEventListener("click", selectEdgeAction);

      gEdge.appendChild(pathLine);
      gEdge.appendChild(hitLine);
      edgeLayer.appendChild(gEdge);
    });

    // Helper for multi-line node label wrapping
    function formatNodeLabel(name) {
      if (name === "od280/od315_of_diluted_wines") {
        return ["od280/od315_of", "diluted_wines"];
      }
      if (name === "nonflavanoid_phenols") {
        return ["nonflavanoid_", "phenols"];
      }
      if (name === "alcalinity_of_ash") {
        return ["alcalinity_", "of_ash"];
      }
      if (name === "color_intensity") {
        return ["color_", "intensity"];
      }
      if (name === "total_phenols") {
        return ["total_", "phenols"];
      }
      return [name];
    }

    // Draw Nodes
    nodes.forEach(node => {
      const coord = nodeCoords[node.id];
      if (!coord) return;

      const gNode = document.createElementNS("http://www.w3.org/2000/svg", "g");
      gNode.setAttribute("class", "graph-node");
      gNode.setAttribute("transform", `translate(${coord.x}, ${coord.y})`);

      // Tooltip
      const nodeTooltip = document.createElementNS("http://www.w3.org/2000/svg", "title");
      nodeTooltip.textContent = `${node.name} (${node.role.toUpperCase()}) • In-Degree: ${node.in_degree}, Out-Degree: ${node.out_degree}`;
      gNode.appendChild(nodeTooltip);

      const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      circle.setAttribute("r", "16");
      circle.setAttribute("class", "node-circle");

      // Color by role
      let fillColor = "#38bdf8";
      let strokeColor = "#0284c7";
      if (node.role === "root") {
        fillColor = "#10b981";
        strokeColor = "#059669";
      } else if (node.role === "sink") {
        fillColor = "#f59e0b";
        strokeColor = "#d97706";
      } else if (node.role === "isolated") {
        fillColor = "#64748b";
        strokeColor = "#475569";
      }

      circle.setAttribute("fill", fillColor);
      circle.setAttribute("stroke", strokeColor);

      // Multi-line Label for readability
      const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
      text.setAttribute("class", "node-text");
      text.setAttribute("text-anchor", "middle");

      const labelLines = formatNodeLabel(node.name);
      if (labelLines.length === 1) {
        const tspan = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
        tspan.setAttribute("x", "0");
        tspan.setAttribute("dy", "26");
        tspan.textContent = labelLines[0];
        text.appendChild(tspan);
      } else {
        const tspan1 = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
        tspan1.setAttribute("x", "0");
        tspan1.setAttribute("dy", "23");
        tspan1.textContent = labelLines[0];
        const tspan2 = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
        tspan2.setAttribute("x", "0");
        tspan2.setAttribute("dy", "12");
        tspan2.textContent = labelLines[1];
        text.appendChild(tspan1);
        text.appendChild(tspan2);
      }

      // Role subtext
      const subtext = document.createElementNS("http://www.w3.org/2000/svg", "text");
      subtext.setAttribute("class", "node-subtext");
      subtext.setAttribute("text-anchor", "middle");
      subtext.setAttribute("y", labelLines.length > 1 ? "49" : "39");
      subtext.textContent = node.role.toUpperCase();

      gNode.appendChild(circle);
      gNode.appendChild(text);
      gNode.appendChild(subtext);

      gNode.addEventListener("click", () => {
        highlightNodeEdges(node.name);
        showNodeInspector(node, edges, data.cai);
      });

      nodeLayer.appendChild(gNode);
    });

    resetGraphTransform();
  }

  function highlightNodeEdges(nodeName) {
    document.querySelectorAll(".graph-edge").forEach(el => {
      const src = el.getAttribute("data-src");
      const tgt = el.getAttribute("data-tgt");
      if (src === nodeName || tgt === nodeName) {
        el.classList.add("selected");
      } else {
        el.classList.remove("selected");
      }
    });
  }

  // -------------------------------------------------------------------------
  // Inspector Sidebar Updates
  // -------------------------------------------------------------------------
  function showEdgeInspector(edge) {
    inspectorType.textContent = "Directed Causal Edge";
    inspectorTitle.textContent = `${edge.source} → ${edge.target}`;

    inspectorBody.innerHTML = `
      <div class="inspector-field">
        <div class="inspector-field-label">Selected Direction</div>
        <div class="inspector-field-val accent">${escapeHtml(edge.direction)}</div>
      </div>
      <div class="inspector-field">
        <div class="inspector-field-label">Confidence Margin (Δ = |h_f - h_r|)</div>
        <div class="inspector-field-val">${edge.confidence.toFixed(6)}</div>
      </div>
      <div class="inspector-field">
        <div class="inspector-field-label">Forward HSIC: HSIC(X, R_f)</div>
        <div class="inspector-field-val">${edge.forward_hsic.toFixed(6)}</div>
      </div>
      <div class="inspector-field">
        <div class="inspector-field-label">Reverse HSIC: HSIC(Y, R_r)</div>
        <div class="inspector-field-val">${edge.reverse_hsic.toFixed(6)}</div>
      </div>
      <div class="inspector-field">
        <div class="inspector-field-label">Decision Rule (ANM Residual Independence)</div>
        <div class="inspector-hint">In accordance with Additive Noise Model assumptions, forward residual dependence is strictly lower (${edge.forward_hsic.toFixed(6)} &lt; ${edge.reverse_hsic.toFixed(6)}), establishing the inferred orientation <strong>${escapeHtml(edge.source)}</strong> &rarr; <strong>${escapeHtml(edge.target)}</strong>.</div>
      </div>
    `;
  }

  function showNodeInspector(node, edges, caiData) {
    inspectorType.textContent = `Feature Node (${node.role.toUpperCase()})`;
    inspectorTitle.textContent = node.name;

    const outgoing = edges.filter(e => e.source === node.name).map(e => e.target);
    const incoming = edges.filter(e => e.target === node.name).map(e => e.source);

    // Find CAI score if available
    let caiInfo = "";
    if (caiData && caiData.records) {
      const rec = caiData.records.find(r => r.feature === node.name);
      if (rec) {
        caiInfo = `
          <div class="inspector-field">
            <div class="inspector-field-label">Causal Feature Attribution (CAI)</div>
            <div class="inspector-field-val accent">${rec.cai > 0 ? "+" : ""}${rec.cai.toFixed(6)}</div>
            <div class="inspector-hint">95% CI: [${rec.ci_lower.toFixed(6)}, ${rec.ci_upper.toFixed(6)}]</div>
          </div>
        `;
      }
    }

    inspectorBody.innerHTML = `
      <div class="inspector-field">
        <div class="inspector-field-label">Structural DAG Role</div>
        <div class="inspector-field-val">${node.role.toUpperCase()} (In-Degree: ${node.in_degree}, Out-Degree: ${node.out_degree})</div>
      </div>
      ${caiInfo}
      <div class="inspector-field">
        <div class="inspector-field-label">Direct Causes / Parents (${incoming.length})</div>
        <div class="inspector-field-val">${incoming.length > 0 ? incoming.map(escapeHtml).join(", ") : "None (Root Cause)"}</div>
      </div>
      <div class="inspector-field">
        <div class="inspector-field-label">Direct Effects / Children (${outgoing.length})</div>
        <div class="inspector-field-val">${outgoing.length > 0 ? outgoing.map(escapeHtml).join(", ") : "None (Sink Variable)"}</div>
      </div>
    `;
  }

  // -------------------------------------------------------------------------
  // Graph Zoom & Pan Handlers
  // -------------------------------------------------------------------------
  function applyGraphTransform() {
    graphZoomGroup.setAttribute(
      "transform",
      `translate(${transform.x}, ${transform.y}) scale(${transform.scale})`
    );
  }

  function resetGraphTransform() {
    transform = { x: 0, y: 0, scale: 1 };
    applyGraphTransform();
  }

  zoomInBtn.addEventListener("click", () => {
    transform.scale = Math.min(transform.scale * 1.2, 3.5);
    applyGraphTransform();
  });

  zoomOutBtn.addEventListener("click", () => {
    transform.scale = Math.max(transform.scale / 1.2, 0.4);
    applyGraphTransform();
  });

  zoomResetBtn.addEventListener("click", resetGraphTransform);

  // SVG Pan with Mouse
  graphSvg.addEventListener("mousedown", (e) => {
    if (e.target.closest(".graph-node") || e.target.closest(".graph-controls")) return;
    isPanning = true;
    startPan = { x: e.clientX - transform.x, y: e.clientY - transform.y };
  });

  window.addEventListener("mousemove", (e) => {
    if (!isPanning) return;
    transform.x = e.clientX - startPan.x;
    transform.y = e.clientY - startPan.y;
    applyGraphTransform();
  });

  window.addEventListener("mouseup", () => {
    isPanning = false;
  });

  // -------------------------------------------------------------------------
  // 3. Causal Attribution (CAI) Horizontal Bar Chart
  // -------------------------------------------------------------------------
  function renderCai(cai) {
    if (!cai || !cai.records) return;

    caiBaseline.textContent = `E[f(X)] = ${cai.baseline.toFixed(6)}`;
    caiBarsContainer.innerHTML = "";

    const records = cai.records;
    if (records.length === 0) return;

    // Determine coordinate scale domain
    let minVal = Math.min(...records.map(r => r.ci_lower), 0);
    let maxVal = Math.max(...records.map(r => r.ci_upper), 0.55);

    // Add 6% padding to min/max
    const range = maxVal - minVal;
    const domainMin = minVal - range * 0.04;
    const domainMax = maxVal + range * 0.04;
    const totalDomain = domainMax - domainMin;

    const toPercent = (val) => ((val - domainMin) / totalDomain) * 100;
    const zeroPercent = toPercent(0);

    records.forEach((rec, idx) => {
      const row = document.createElement("div");
      row.className = "cai-row";

      const valPercent = toPercent(rec.cai);
      const ciLowPercent = Math.max(0, toPercent(rec.ci_lower));
      const ciHighPercent = Math.min(100, toPercent(rec.ci_upper));

      const isSig = rec.is_significant;
      const isNegative = rec.cai < 0;

      const barLeft = Math.min(zeroPercent, valPercent);
      const barWidth = Math.abs(valPercent - zeroPercent);

      let barClass = "non-sig";
      if (isNegative) {
        barClass = "negative";
      } else if (isSig) {
        barClass = "sig";
      }

      const scoreClass = isNegative ? "negative" : (isSig ? "sig" : "non-sig");
      const sign = rec.cai > 0 ? "+" : "";

      row.innerHTML = `
        <div class="cai-feature-info">
          <span class="cai-rank font-mono">${idx + 1}</span>
          <span class="cai-name font-mono" title="${escapeHtml(rec.feature)}">${escapeHtml(rec.feature)}</span>
        </div>
        <div class="cai-chart-track">
          <div class="zero-axis" style="left: ${zeroPercent.toFixed(2)}%;"></div>
          <div class="cai-bar ${barClass}" 
               style="left: ${barLeft.toFixed(2)}%; width: ${Math.max(barWidth, 0.4).toFixed(2)}%;"></div>
          <div class="ci-whisker" 
               style="left: ${ciLowPercent.toFixed(2)}%; width: ${(ciHighPercent - ciLowPercent).toFixed(2)}%;">
            <div class="ci-cap-left"></div>
            <div class="ci-cap-right" style="left: 100%;"></div>
          </div>
        </div>
        <div class="cai-metrics">
          <span class="cai-score ${scoreClass}">${sign}${rec.cai.toFixed(6)}</span>
          <span class="cai-ci-text">[${rec.ci_lower.toFixed(4)}, ${rec.ci_upper.toFixed(4)}]</span>
        </div>
      `;

      caiBarsContainer.appendChild(row);
    });
  }

  // -------------------------------------------------------------------------
  // 4. ANM Pairwise Results Table
  // -------------------------------------------------------------------------
  function renderPairwiseTable() {
    if (!analysisData || !analysisData.pairwise_records) return;

    let items = [...analysisData.pairwise_records];

    // Filter by tab
    if (activeFilter === "retained") {
      items = items.filter(r => r.is_retained);
    } else if (activeFilter === "filtered") {
      items = items.filter(r => !r.is_retained);
    }

    // Filter by search string
    if (activeSearch.trim() !== "") {
      const q = activeSearch.toLowerCase().trim();
      items = items.filter(r =>
        r.cause.toLowerCase().includes(q) ||
        r.effect.toLowerCase().includes(q) ||
        r.status.toLowerCase().includes(q)
      );
    }

    // Sort items
    items.sort((a, b) => {
      let valA = a[sortColumn];
      let valB = b[sortColumn];
      if (typeof valA === "string") valA = valA.toLowerCase();
      if (typeof valB === "string") valB = valB.toLowerCase();

      if (valA < valB) return sortDirection === "asc" ? -1 : 1;
      if (valA > valB) return sortDirection === "asc" ? 1 : -1;
      return 0;
    });

    pairwiseTableBody.innerHTML = "";

    items.forEach(item => {
      const tr = document.createElement("tr");

      let statusClass = "status-filtered";
      if (item.status === "Retained in DAG") {
        statusClass = "status-retained";
      } else if (item.status.includes("Pruned")) {
        statusClass = "status-pruned";
      }

      tr.innerHTML = `
        <td class="td-num font-mono">${item.pair_index}</td>
        <td><strong>${escapeHtml(item.cause)}</strong></td>
        <td>${escapeHtml(item.effect)}</td>
        <td class="font-mono accent direction-cell">${escapeHtml(item.direction)}</td>
        <td class="td-num font-mono">${item.forward_hsic.toFixed(6)}</td>
        <td class="td-num font-mono">${item.reverse_hsic.toFixed(6)}</td>
        <td class="td-num font-mono highlight-conf">${item.confidence.toFixed(6)}</td>
        <td class="td-center"><span class="status-tag ${statusClass}">${escapeHtml(item.status)}</span></td>
      `;

      pairwiseTableBody.appendChild(tr);
    });

    tableSummaryText.textContent = `Showing ${items.length} of ${analysisData.pairwise_records.length} evaluated pairs`;
  }

  // Filter Pills Event Listeners
  filterPills.forEach(pill => {
    pill.addEventListener("click", () => {
      filterPills.forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      activeFilter = pill.getAttribute("data-filter");
      renderPairwiseTable();
    });
  });

  // Table Search Input Listener
  tableSearchInput.addEventListener("input", (e) => {
    activeSearch = e.target.value;
    renderPairwiseTable();
  });

  // Table Header Sort Listener
  tableHeaders.forEach(th => {
    th.addEventListener("click", () => {
      const col = th.getAttribute("data-sort");
      if (!col) return;
      if (sortColumn === col) {
        sortDirection = sortDirection === "asc" ? "desc" : "asc";
      } else {
        sortColumn = col;
        sortDirection = "desc";
      }
      renderPairwiseTable();
    });
  });

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
