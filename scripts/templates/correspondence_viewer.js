/* Embedded by correspondence_to_html_viewer.py. No network or runtime dependencies. */
"use strict";
(() => {
  const data = JSON.parse(document.getElementById("viewer-data").textContent);
  const $ = id => document.getElementById(id);
  const stage = $("stage"), canvas = $("scene"), overlay = $("overlay");
  const ctx = overlay.getContext("2d");
  const sides = ["source", "target"];
  const state = {yaw: -0.15, pitch: -0.65, zoom: 0.8, panX: 0, panY: 0,
    gap: 0.95, selectedMatch: -1, selection: null};
  let width = 1, height = 1, frame = 0, markerHits = [], lineHits = [], pointer = null;
  const display = {}, projected = {}, gpu = {};
  const palette = {source: "#2584db", target: "#e4742e", gt: "#25955b", match: "#168996", error: "#d94859"};
  const count = n => Number(n).toLocaleString("en-US");
  const fmt = (n, digits = 4) => Number.isFinite(n) ? Number(n).toFixed(digits) : "—";
  const xyz = p => p.map(x => fmt(x, 5)).join(", ");
  const coord = p => p.map(Number);
  const label = m => m.label || "Match " + (data.matches.indexOf(m) + 1);
  function notice(message) { $("notice").textContent = message; $("notice").style.display = "block"; }
  for (const side of sides) {
    const item = data[side];
    $(side + "-name").textContent = item.name;
    $(side + "-count").textContent = count(item.vertices.length) + " vertices · " + count(item.faces.length) + " faces";
    display[side] = new Float32Array(item.vertices.length * 3);
    projected[side] = new Float64Array(item.vertices.length * 3);
    item.vertices.forEach((v, i) => {
      for (let k = 0; k < 3; k++) display[side][3 * i + k] = (v[k] - item.center[k]) / data.scale;
    });
  }
  const hasGT = data.matches.some(m => m.gt !== null);
  $("dataset-info").textContent = count(data.matches.length) + (hasGT ? " evaluated landmarks" : " supplied matches") + " · interactive 3D";
  $("provenance").textContent = "Matches: " + data.meta.csv + " | Source: " + data.source.path +
    " | Target: " + data.target.path + ". Coordinates use each original geometry file's units. Display centering, scale and separation never change the reported values.";
  if (!hasGT) {
    for (const id of ["show-gt", "show-errors"]) { $(id).checked = false; $(id).disabled = true; }
  }
  if (!data.source.faces.length && !data.target.faces.length) {
    $("render-mode").value = "points";
    $("render-mode").querySelector('[value="mesh"]').disabled = true;
  }
  data.matches.forEach((m, i) => {
    const option = document.createElement("option"); option.value = i;
    option.textContent = label(m) + (Number.isFinite(m.metrics.target_error_bbox_pct) ? " · " + fmt(m.metrics.target_error_bbox_pct, 2) + "% error" : "");
    $("match-select").append(option);
  });

  // A shared orthographic transform is used by WebGL, overlays and CPU picking.
  function rotate(x, y, z) {
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const rx = cy * x + sy * z, rz = -sy * x + cy * z;
    return [rx, cp * y - sp * rz, sp * y + cp * rz];
  }
  function projectPoint(side, original) {
    const item = data[side], offset = (side === "source" ? -0.5 : 0.5) * state.gap;
    const p = rotate((original[0] - item.center[0]) / data.scale + offset,
      (original[1] - item.center[1]) / data.scale, (original[2] - item.center[2]) / data.scale);
    const base = Math.min(width, height) * state.zoom;
    return [width / 2 + state.panX + p[0] * base, height / 2 + state.panY - p[1] * base, p[2]];
  }
  function projectVertices() {
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw), cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    const base = Math.min(width, height) * state.zoom;
    for (const side of sides) {
      const arr = display[side], out = projected[side], offset = (side === "source" ? -0.5 : 0.5) * state.gap;
      for (let j = 0; j < arr.length; j += 3) {
        const x = arr[j] + offset, y = arr[j + 1], z = arr[j + 2];
        const rx = cy * x + sy * z, rz = -sy * x + cy * z;
        out[j] = width / 2 + state.panX + rx * base;
        out[j + 1] = height / 2 + state.panY - (cp * y - sp * rz) * base;
        out[j + 2] = sp * y + cp * rz;
      }
    }
  }

  const gl = canvas.getContext("webgl", {alpha: false, antialias: true, preserveDrawingBuffer: true});
  let program = null, attrib = {}, uniform = {};
  if (!gl) {
    notice("WebGL is unavailable. Showing a point preview; vertex picking and correspondence inspection still work. Enable browser hardware acceleration for mesh surfaces.");
    $("render-mode").value = "points"; $("render-mode").disabled = true;
  } else {
    const vertexShader = `
      attribute vec3 aPosition; attribute vec3 aNormal; attribute vec3 aColor;
      uniform vec2 uViewport; uniform vec2 uPan; uniform vec2 uAngles;
      uniform float uBase; uniform float uOffset; uniform float uDpr;
      varying vec3 vColor; varying float vLight;
      vec3 rotatePoint(vec3 p) {
        float cy=cos(uAngles.x), sy=sin(uAngles.x), cp=cos(uAngles.y), sp=sin(uAngles.y);
        float x=cy*p.x+sy*p.z, z=-sy*p.x+cy*p.z;
        return vec3(x,cp*p.y-sp*z,sp*p.y+cp*z);
      }
      void main() {
        vec3 p=rotatePoint(aPosition+vec3(uOffset,0.0,0.0));
        gl_Position=vec4((p.x*uBase+uPan.x)*2.0/uViewport.x,
          (p.y*uBase-uPan.y)*2.0/uViewport.y,-p.z/10.0,1.0);
        gl_PointSize=3.2*uDpr;
        vColor=aColor;
        vec3 n=rotatePoint(aNormal);
        vLight=0.62+0.38*abs(dot(n,normalize(vec3(-0.3,0.5,1.0))));
      }`;
    const fragmentShader = `
      precision mediump float;
      varying vec3 vColor; varying float vLight;
      uniform float uOpacity; uniform bool uPoints;
      void main() {
        if(uPoints && distance(gl_PointCoord,vec2(0.5))>0.5) discard;
        gl_FragColor=vec4(vColor*(uPoints?1.0:vLight),uOpacity);
      }`;
    function shader(type, source) {
      const s = gl.createShader(type); gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    }
    try {
      program = gl.createProgram();
      gl.attachShader(program, shader(gl.VERTEX_SHADER, vertexShader));
      gl.attachShader(program, shader(gl.FRAGMENT_SHADER, fragmentShader)); gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);
      for (const name of ["aPosition", "aNormal", "aColor"]) attrib[name] = gl.getAttribLocation(program, name);
      for (const name of ["uViewport", "uPan", "uAngles", "uBase", "uOffset", "uDpr", "uOpacity", "uPoints"]) uniform[name] = gl.getUniformLocation(program, name);
      for (const side of sides) gpu[side] = buildBuffers(side);
    } catch (error) {
      program = null; notice("Mesh rendering failed: " + error.message + ". A point preview is available.");
      $("render-mode").value = "points"; $("render-mode").disabled = true;
    }
  }
  function buildBuffers(side) {
    const item = data[side], points = display[side], normals = new Float32Array(points.length), colors = new Float32Array(points.length);
    const fallback = side === "source" ? [163, 189, 205] : [199, 185, 164];
    item.vertices.forEach((v, i) => {
      const color = item.colors ? item.colors[i] : fallback;
      for (let k = 0; k < 3; k++) colors[i * 3 + k] = color[k] / 255;
    });
    for (const f of item.faces) {
      const a = f[0] * 3, b = f[1] * 3, c = f[2] * 3;
      const ux = points[b]-points[a], uy = points[b+1]-points[a+1], uz = points[b+2]-points[a+2];
      const vx = points[c]-points[a], vy = points[c+1]-points[a+1], vz = points[c+2]-points[a+2];
      const nx = uy*vz-uz*vy, ny = uz*vx-ux*vz, nz = ux*vy-uy*vx;
      for (const j of [a, b, c]) { normals[j] += nx; normals[j+1] += ny; normals[j+2] += nz; }
    }
    for (let i = 0; i < normals.length; i += 3) {
      const size = Math.hypot(normals[i], normals[i+1], normals[i+2]) || 1;
      for (let k = 0; k < 3; k++) normals[i+k] /= size;
    }
    const make = values => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, values, gl.STATIC_DRAW); return b; };
    // Expanded triangles avoid a dependency on 32-bit element-index extensions.
    const expand = values => {
      const arr = new Float32Array(item.faces.length * 9); let j = 0;
      for (const face of item.faces) for (const index of face) for (let k = 0; k < 3; k++) arr[j++] = values[index * 3 + k];
      return arr;
    };
    return {points: {position: make(points), normal: make(normals), color: make(colors), count: item.vertices.length},
      mesh: {position: make(expand(points)), normal: make(expand(normals)), color: make(expand(colors)), count: item.faces.length * 3}};
  }
  function drawGeometry() {
    if (!program) return;
    gl.viewport(0, 0, canvas.width, canvas.height);
    gl.clearColor(0.953, 0.961, 0.965, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST); gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.useProgram(program);
    gl.uniform2f(uniform.uViewport, width, height); gl.uniform2f(uniform.uPan, state.panX, state.panY);
    gl.uniform2f(uniform.uAngles, state.yaw, state.pitch); gl.uniform1f(uniform.uBase, Math.min(width, height) * state.zoom);
    gl.uniform1f(uniform.uDpr, Math.min(window.devicePixelRatio || 1, 2));
    gl.uniform1f(uniform.uOpacity, Number($("surface-opacity").value));
    for (const side of sides) {
      const asPoints = $("render-mode").value === "points" || !data[side].faces.length;
      const b = gpu[side][asPoints ? "points" : "mesh"];
      gl.uniform1f(uniform.uOffset, (side === "source" ? -0.5 : 0.5) * state.gap);
      gl.uniform1i(uniform.uPoints, asPoints ? 1 : 0);
      for (const [attribute, buffer] of [["aPosition", b.position], ["aNormal", b.normal], ["aColor", b.color]]) {
        gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.enableVertexAttribArray(attrib[attribute]);
        gl.vertexAttribPointer(attrib[attribute], 3, gl.FLOAT, false, 0, 0);
      }
      gl.drawArrays(asPoints ? gl.POINTS : gl.TRIANGLES, 0, b.count);
    }
  }
  function fallbackPoints() {
    ctx.fillStyle = "#f3f5f6"; ctx.fillRect(0, 0, width, height);
    const order = [];
    for (const side of sides) for (let i = 0; i < data[side].vertices.length; i++) order.push([side, i]);
    order.sort((a, b) => projected[a[0]][a[1]*3+2] - projected[b[0]][b[1]*3+2]);
    for (const [side, i] of order) {
      const p = projected[side], color = data[side].colors && data[side].colors[i];
      ctx.fillStyle = color ? "rgb(" + color.join(",") + ")" : side === "source" ? "#a3bdcd" : "#c7b9a4";
      ctx.fillRect(p[i*3]-1.5, p[i*3+1]-1.5, 3, 3);
    }
  }
  function line(a, b, color, weight, dashed = false) {
    ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]);
    ctx.strokeStyle = color; ctx.lineWidth = weight; ctx.setLineDash(dashed ? [5, 4] : []); ctx.stroke(); ctx.setLineDash([]);
  }
  function marker(p, color, radius, text) {
    ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, Math.PI*2); ctx.fillStyle = color; ctx.fill();
    ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
    if (text) {
      ctx.font = "11px system-ui"; ctx.lineWidth = 3.5; ctx.strokeStyle = "rgba(255,255,255,.95)";
      ctx.strokeText(text, p[0]+9, p[1]-8); ctx.fillStyle = "#22303c"; ctx.fillText(text, p[0]+9, p[1]-8);
    }
  }
  function visibleMatches() {
    const indices = data.matches.map((m, i) => i);
    if ($("isolate").checked && state.selectedMatch >= 0) return [state.selectedMatch];
    return indices.sort((a, b) => Number(a === state.selectedMatch) - Number(b === state.selectedMatch));
  }
  function drawCorrespondences() {
    markerHits = []; lineHits = [];
    for (const i of visibleMatches()) {
      const m = data.matches[i], selected = i === state.selectedMatch;
      const a = projectPoint("source", m.source), b = projectPoint("target", m.target);
      const c = m.gt ? projectPoint("target", m.gt) : null;
      ctx.globalAlpha = state.selectedMatch < 0 || selected ? 1 : 0.24;
      if ($("show-matches").checked) {
        line(a, b, palette.match, selected ? 2.7 : 1.2);
        lineHits.push({a, b, index: i});
      }
      if (c && $("show-errors").checked) {
        line(b, c, palette.error, selected ? 3 : 1.8, true);
        lineHits.push({a: b, b: c, index: i});
      }
      const r = selected ? 6.5 : 4.3;
      marker(a, palette.source, r); marker(b, palette.target, r);
      markerHits.push({p: a, side: "source", vertex: m.sourceIndex, match: i, kind: "source vertex"},
        {p: b, side: "target", vertex: m.targetIndex, match: i, kind: "predicted target"});
      if (c && $("show-gt").checked) {
        marker(c, palette.gt, r, selected || $("show-labels").checked ? label(m) : "");
        markerHits.push({p: c, side: "target", vertex: m.gtIndex, match: i, kind: "manual target", original: m.gt});
      } else if (selected || $("show-labels").checked) marker(b, palette.target, r, label(m));
    }
    ctx.globalAlpha = 1;
  }
  function drawSelection() {
    const s = state.selection; if (!s) return;
    const p = projectPoint(s.side, s.surface_xyz || s.landmark_xyz || s.vertex_xyz);
    ctx.beginPath(); ctx.arc(p[0], p[1], 10, 0, Math.PI*2); ctx.lineWidth = 2; ctx.strokeStyle = "#172e3a"; ctx.stroke();
    line([p[0]-14,p[1]], [p[0]-6,p[1]], "#172e3a", 1.5);
    line([p[0]+6,p[1]], [p[0]+14,p[1]], "#172e3a", 1.5);
    if (s.surface_xyz || s.landmark_xyz) {
      const v = projectPoint(s.side, s.vertex_xyz);
      line(p, v, "#172e3a", 1, true); marker(v, "#172e3a", 2.5);
    }
  }
  function drawAxes() {
    const origin = [35, height-70];
    const axes = [[1,0,0,"#cf5960","X"], [0,1,0,"#37905b","Y"], [0,0,1,"#397ac0","Z"]];
    ctx.font = "11px system-ui";
    for (const [x,y,z,color,text] of axes) {
      const v = rotate(x,y,z), p = [origin[0]+v[0]*24, origin[1]-v[1]*24];
      line(origin, p, color, 2); ctx.fillStyle = color; ctx.fillText(text, p[0]+3, p[1]-3);
    }
  }
  function render() {
    frame = 0; projectVertices(); drawGeometry(); ctx.clearRect(0,0,width,height);
    if (!program) fallbackPoints();
    drawCorrespondences(); drawSelection(); drawAxes();
  }
  function schedule() { if (!frame) frame = requestAnimationFrame(render); }
  function resize() {
    const r = stage.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
    width = r.width; height = r.height;
    for (const c of [canvas, overlay]) { c.width = Math.max(1, Math.round(width*dpr)); c.height = Math.max(1, Math.round(height*dpr)); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); schedule();
  }
  function fitView() {
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    for (const side of sides) {
      const arr = display[side], offset = (side === "source" ? -0.5 : 0.5) * state.gap;
      for (let j = 0; j < arr.length; j += 3) {
        const p = rotate(arr[j]+offset, arr[j+1], arr[j+2]);
        xmin = Math.min(xmin,p[0]); xmax = Math.max(xmax,p[0]); ymin = Math.min(ymin,p[1]); ymax = Math.max(ymax,p[1]);
      }
    }
    const base = Math.min(width*.84/Math.max(xmax-xmin,1e-6), height*.76/Math.max(ymax-ymin,1e-6));
    state.zoom = Math.min(20, Math.max(.08, base / Math.min(width,height)));
    state.panX = -(xmin+xmax)*base/2; state.panY = (ymin+ymax)*base/2 + 10;
    schedule();
  }

  function table(rows) {
    const dl = document.createElement("dl");
    for (const [key, value, mono] of rows) {
      const dt = document.createElement("dt"), dd = document.createElement("dd");
      dt.textContent = key; dd.textContent = value; if (mono) dd.className = "coords"; dl.append(dt, dd);
    }
    return dl;
  }
  function updateMatchDetails() {
    $("match-select").value = state.selectedMatch;
    const panel = $("match-details"); panel.replaceChildren();
    if (state.selectedMatch < 0) {
      const p = document.createElement("p"); p.className = "muted";
      p.textContent = "Select a landmark or click a line to inspect its prediction."; panel.append(p); return;
    }
    const m = data.matches[state.selectedMatch], v = m.metrics;
    const rows = [["Source vertex", count(m.sourceIndex)], ["Predicted vertex", count(m.targetIndex)],
      ["Cosine similarity", fmt(v.cosine_score,6)]];
    if (m.gt) {
      rows.push(["Manual target vertex", count(m.gtIndex)], ["Error (mesh units)",fmt(v.target_error,5)],
        ["Error / target bbox",fmt(v.target_error_bbox_pct ?? v.target_error_bbox*100,3)+"%"],
        ["GT feature rank",Number.isFinite(v.gt_feature_rank)?count(v.gt_feature_rank):"—"],
        ["NN margin",fmt(v.nn_margin,6)]);
    }
    if (m.mutual !== null) rows.push(["Mutual confirmed", m.mutual ? "Yes" : "Not confirmed"]);
    panel.append(table(rows));
    const p = document.createElement("p"); p.className = "muted";
    p.textContent = m.gt ? "Error = distance from the prediction to your manual target position. Cosine similarity alone does not measure anatomical accuracy." : "Cosine similarity describes the stored feature match.";
    if (m.mutual === false) p.textContent += " The CSV does not distinguish a non-mutual match from an unchecked match.";
    panel.append(p);
  }
  function relatedMatches(side, index) {
    return data.matches.flatMap((m,i) => side === "source" ? (m.sourceIndex === index ? [i] : []) :
      (m.targetIndex === index || m.gtIndex === index ? [i] : []));
  }
  function nearestVertex(side, point) {
    let index = 0, best = Infinity;
    data[side].vertices.forEach((v,i) => {
      const distance = (v[0]-point[0])**2+(v[1]-point[1])**2+(v[2]-point[2])**2;
      if (distance < best) { best = distance; index = i; }
    });
    return {index, distance: Math.sqrt(best)};
  }
  function updatePointDetails() {
    const s = state.selection; $("point-details").replaceChildren();
    $("pick-readout").hidden = !s;
    $("copy-selection").disabled = !s; $("selection-json").value = s ? JSON.stringify(s, null, 2) : "";
    $("selection-title").textContent = s ? data[s.side].name + " · " + s.kind : "Click either mesh";
    $("selection-hint").hidden = Boolean(s);
    if (!s) return;
    $("pick-readout-title").textContent = data[s.side].name + " · vertex #" + s.vertex_index;
    const readout = ["Vertex XYZ: " + xyz(s.vertex_xyz)];
    if (s.surface_xyz) readout.unshift("Surface XYZ: " + xyz(s.surface_xyz));
    if (s.landmark_xyz) readout.unshift("Manual XYZ: " + xyz(s.landmark_xyz));
    $("pick-readout-coords").textContent = readout.join("\n");
    const rows = [["Vertex index", count(s.vertex_index)], ["Vertex XYZ", xyz(s.vertex_xyz), true]];
    if (s.surface_xyz) rows.push(["Surface XYZ", xyz(s.surface_xyz), true], ["Face index", count(s.face_index)], ["Snap distance", fmt(s.snap_distance,6)]);
    if (s.landmark_xyz) rows.push(["Manual XYZ", xyz(s.landmark_xyz), true], ["Snap distance", fmt(s.snap_distance,6)]);
    if (s.rgb) rows.push(["Vertex RGB", s.rgb.join(", ")]);
    rows.push(["Stored matches", s.related_matches.length ? s.related_matches.map(i => label(data.matches[i])).join(", ") : "None for this vertex"]);
    $("point-details").append(table(rows));
    $("vertex-index").value = s.vertex_index; $("pick-side").value = s.side;
  }
  function selectVertex(side, index, extra = {}, syncMatch = true) {
    if (!sides.includes(side) || !Number.isInteger(index) || index < 0 || index >= data[side].vertices.length) {
      $("status").textContent = "Enter an index from 0 to " + (data[sides.includes(side) ? side : "source"].vertices.length - 1) + "."; return false;
    }
    const related = relatedMatches(side,index);
    state.selection = {kind: "vertex", shape: data[side].name, side, geometry: data[side].path,
      coordinate_system: "original geometry coordinates", vertex_index: index, vertex_xyz: coord(data[side].vertices[index]),
      rgb: data[side].colors ? data[side].colors[index] : null, related_matches: related, ...extra};
    if (syncMatch) { state.selectedMatch = related.length ? related[0] : -1; updateMatchDetails(); }
    $("status").textContent = ""; updatePointDetails(); schedule(); return true;
  }
  function selectMatch(index) {
    if (!Number.isInteger(index) || index < -1 || index >= data.matches.length) return;
    state.selectedMatch = index; updateMatchDetails();
    if (index >= 0) {
      const m = data.matches[index];
      selectVertex("source", m.sourceIndex, {kind: "source vertex", match: label(m), correspondence: m}, false);
    } else { state.selection = null; updatePointDetails(); schedule(); }
  }
  function selectMarker(hit) {
    selectMatch(hit.match);
    const extra = {kind: hit.kind, match: label(data.matches[hit.match]), correspondence: data.matches[hit.match]};
    let index = hit.vertex;
    if (hit.original) {
      if (index === null) index = nearestVertex(hit.side, hit.original).index;
      extra.landmark_xyz = coord(hit.original);
      extra.snap_distance = Math.hypot(...hit.original.map((x,k) => x-data[hit.side].vertices[index][k]));
    }
    selectVertex(hit.side, index, extra, false);
  }
  function distanceToSegment(x,y,a,b) {
    const dx=b[0]-a[0], dy=b[1]-a[1], length=dx*dx+dy*dy;
    const t=length ? Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/length)) : 0;
    return Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy);
  }
  function pickSurface(x,y,side) {
    const item = data[side], p = projected[side]; let best = null;
    for (let fi=0; fi<item.faces.length; fi++) {
      const f=item.faces[fi], a=f[0]*3, b=f[1]*3, c=f[2]*3;
      const ax=p[a], ay=p[a+1], bx=p[b], by=p[b+1], cx=p[c], cy=p[c+1];
      if (x<Math.min(ax,bx,cx) || x>Math.max(ax,bx,cx) || y<Math.min(ay,by,cy) || y>Math.max(ay,by,cy)) continue;
      const den=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy); if (Math.abs(den)<1e-10) continue;
      const u=((by-cy)*(x-cx)+(cx-bx)*(y-cy))/den, v=((cy-ay)*(x-cx)+(ax-cx)*(y-cy))/den, w=1-u-v;
      if (u< -1e-8 || v< -1e-8 || w< -1e-8) continue;
      const z=u*p[a+2]+v*p[b+2]+w*p[c+2];
      if (!best || z>best.z) best={side,z,face:fi,weights:[u,v,w],vertices:f};
    }
    return best;
  }
  function pickPoints(x,y,side) {
    const p=projected[side]; let best=null;
    for (let i=0; i<p.length/3; i++) {
      const distance=Math.hypot(x-p[i*3],y-p[i*3+1]);
      if (distance>6) continue;
      // A drawn point under the cursor takes precedence over the wider click tolerance.
      const direct=distance<=2;
      if (!best || (direct && !best.direct) || (direct===best.direct && (direct ? p[i*3+2]>best.z : distance<best.distance)))
        best={side,index:i,z:p[i*3+2],distance,direct};
    }
    return best;
  }
  function pick(x,y) {
    render();
    const markers=markerHits.map((h,order)=>({...h,order,d:Math.hypot(x-h.p[0],y-h.p[1])})).filter(h=>h.d<=9).sort((a,b)=>a.d-b.d || b.order-a.order);
    if (markers.length) { selectMarker(markers[0]); return state.selection; }
    const lines=lineHits.map((h,order)=>({...h,order,d:distanceToSegment(x,y,h.a,h.b)})).filter(h=>h.d<4).sort((a,b)=>a.d-b.d || b.order-a.order);
    if (lines.length) { selectMatch(lines[0].index); return state.selection; }
    const surfaces=[], points=[];
    for (const side of sides) {
      if ($("render-mode").value === "mesh" && data[side].faces.length && program) {
        const hit=pickSurface(x,y,side); if (hit) surfaces.push(hit);
      } else { const hit=pickPoints(x,y,side); if (hit) points.push(hit); }
    }
    const hits=[...surfaces,...points].sort((a,b)=>b.z-a.z);
    if (!hits.length) { $("status").textContent="No surface at this position. Click a mesh, marker or line."; return null; }
    const hit=hits[0];
    if (hit.face === undefined) selectVertex(hit.side,hit.index);
    else {
      const vertices=data[hit.side].vertices;
      const point=[0,1,2].map(k=>hit.weights.reduce((sum,w,j)=>sum+w*vertices[hit.vertices[j]][k],0));
      const nearest=nearestVertex(hit.side,point);
      selectVertex(hit.side,nearest.index,{kind:"surface point",surface_xyz:point,face_index:hit.face,
        barycentric:hit.weights,snap_distance:nearest.distance});
    }
    return state.selection;
  }

  overlay.addEventListener("pointerdown", e => {
    if (pointer || ![0,1,2].includes(e.button)) return;
    pointer={id:e.pointerId,startX:e.clientX,startY:e.clientY,x:e.clientX,y:e.clientY,moved:false,pan:e.shiftKey||e.button!==0};
    overlay.setPointerCapture(e.pointerId); overlay.classList.add("dragging");
  });
  overlay.addEventListener("pointermove", e => {
    if (!pointer || pointer.id!==e.pointerId) return;
    if (Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)>4) pointer.moved=true;
    const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y; pointer.x=e.clientX;pointer.y=e.clientY;
    if (!pointer.moved) return;
    if (pointer.pan || e.shiftKey) { state.panX+=dx;state.panY+=dy; }
    else { state.yaw+=dx*.008;state.pitch+=dy*.008; }
    schedule();
  });
  overlay.addEventListener("pointerup", e => {
    if (!pointer || pointer.id!==e.pointerId) return;
    const click=!pointer.moved && !pointer.pan && Math.hypot(e.clientX-pointer.startX,e.clientY-pointer.startY)<=4;
    pointer=null; overlay.classList.remove("dragging");
    if (overlay.hasPointerCapture(e.pointerId)) overlay.releasePointerCapture(e.pointerId);
    if (click) { const r=overlay.getBoundingClientRect();pick(e.clientX-r.left,e.clientY-r.top); }
  });
  for (const name of ["pointercancel","lostpointercapture"]) overlay.addEventListener(name,e=>{if(pointer && pointer.id===e.pointerId){pointer=null;overlay.classList.remove("dragging");}});
  overlay.addEventListener("contextmenu",e=>e.preventDefault());
  overlay.addEventListener("wheel",e=>{
    e.preventDefault(); const delta=e.deltaY*(e.deltaMode===1?16:e.deltaMode===2?height:1);
    const next=Math.max(.08,Math.min(20,state.zoom*Math.exp(-delta*.001))), ratio=next/state.zoom;
    const r=overlay.getBoundingClientRect(),x=e.clientX-r.left-width/2,y=e.clientY-r.top-height/2;
    state.panX=x-(x-state.panX)*ratio;state.panY=y-(y-state.panY)*ratio;state.zoom=next;schedule();
  },{passive:false});
  overlay.addEventListener("keydown",e=>{if(e.key==="Escape") $("clear-selection").click();if(e.key.toLowerCase()==="f") fitView();});
  $("match-select").addEventListener("change",()=>selectMatch(Number($("match-select").value)));
  $("previous-match").addEventListener("click",()=>{if(data.matches.length) selectMatch(state.selectedMatch<0?data.matches.length-1:(state.selectedMatch-1+data.matches.length)%data.matches.length);});
  $("next-match").addEventListener("click",()=>{if(data.matches.length) selectMatch((state.selectedMatch+1)%data.matches.length);});
  for (const id of ["render-mode","surface-opacity","show-matches","show-gt","show-errors","show-labels","isolate"]) $(id).addEventListener("input",schedule);
  $("gap").addEventListener("input",()=>{state.gap=Number($("gap").value);schedule();});
  $("reset-view").addEventListener("click",fitView);
  $("front-view").addEventListener("click",()=>{state.yaw=0;state.pitch=0;fitView();});
  $("side-view").addEventListener("click",()=>{state.yaw=Math.PI/2;state.pitch=0;fitView();});
  $("clear-selection").addEventListener("click",()=>{selectMatch(-1);$("status").textContent="";});
  $("jump-vertex").addEventListener("click",()=>{if($("vertex-index").value.trim()!=="")selectVertex($("pick-side").value,Number($("vertex-index").value));});
  $("vertex-index").addEventListener("keydown",e=>{if(e.key==="Enter")$("jump-vertex").click();});
  $("random-vertex").addEventListener("click",()=>{const side=$("pick-side").value;selectVertex(side,Math.floor(Math.random()*data[side].vertices.length));});
  $("copy-selection").addEventListener("click",async()=>{
    if(!state.selection)return;
    try { await navigator.clipboard.writeText($("selection-json").value);$("status").textContent="Selection copied as JSON."; }
    catch { $("selection-json").closest("details").open=true;$("selection-json").focus();$("selection-json").select();
      $("status").textContent="Press Ctrl+C (or Cmd+C) to copy the selected JSON."; }
  });
  canvas.addEventListener("webglcontextlost",e=>{e.preventDefault();notice("The graphics context was lost. Reload this file to restore the scene.");});
  window.addEventListener("resize",resize);
  window.correspondenceViewer={data,state,projectPoint,pick,selectVertex,selectMatch,render,getSelection:()=>state.selection};
  resize();fitView();render();
})();
