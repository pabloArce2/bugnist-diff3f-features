<script setup>
import {
  computed,
  nextTick,
  onActivated,
  onBeforeUnmount,
  onDeactivated,
  onMounted,
  ref,
  watch,
} from "vue";
import {
  Activity,
  AlertTriangle,
  Check,
  Copy,
  Crosshair,
  Expand,
  Eye,
  Loader2,
  Minimize2,
  MousePointer2,
  RefreshCw,
  RotateCcw,
  Shuffle,
  Target,
} from "@lucide/vue";
import * as THREE from "three";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { HtmlStyleControls } from "../lib/HtmlStyleControls.js";

const PRESETS_URL = "/api/correspondence-presets";
const ENDPOINT_TOLERANCE = 1e-3;
const SOURCE_COLOR = 0x2f9b92;
const TARGET_COLOR = 0x5277b8;
const MATCH_COLOR = new THREE.Color(0x6dd4ca);
const SELECTED_MATCH_COLOR = new THREE.Color(0xffd166);
const ERROR_COLOR = new THREE.Color(0xe97451);
const SELECTED_ERROR_COLOR = new THREE.Color(0xffb347);
const SOURCE_MARKER_COLOR = 0x30c5b5;
const PREDICTED_MARKER_COLOR = 0xffd166;
const GROUND_TRUTH_MARKER_COLOR = 0xff6f61;

const viewerFrame = ref(null);
const stage = ref(null);
const presets = ref([]);
const selectedPresetKey = ref("");
const selectedMatchIndex = ref(0);
const loadingPresets = ref(false);
const loadingGeometry = ref(false);
const loadError = ref("");
const showMatchLines = ref(true);
const showErrorLines = ref(true);
const showMarkers = ref(true);
const isolateSelected = ref(false);
const meshOpacity = ref(0.88);
const separation = ref(1);
const inspectSide = ref("source");
const vertexIndexInput = ref("0");
const inspection = ref(null);
const copyStatus = ref("");
const isFullscreen = ref(false);

const presetCollections = computed(() => {
  const groups = new Map();
  for (const preset of presets.value) {
    const id = String(preset.datasetId || "correspondence");
    if (!groups.has(id)) {
      groups.set(id, {
        id,
        title: preset.datasetTitle || "Correspondence datasets",
        presets: [],
      });
    }
    groups.get(id).presets.push(preset);
  }
  return Array.from(groups.values());
});

const activePreset = computed(() =>
  presets.value.find((preset) => presetKey(preset) === selectedPresetKey.value) || null,
);

const matches = computed(() => activePreset.value?.matches || []);
const selectedMatch = computed(() => matches.value[selectedMatchIndex.value] || null);
const selectedPckEntries = computed(() => Object.entries(selectedMatch.value?.metrics?.pckBBox || {}));
const summaryPckEntries = computed(() => Object.entries(activePreset.value?.summary?.pckAtBBox || {}));
const inspectVertexCount = computed(() => {
  const side = activePreset.value?.[inspectSide.value];
  return Number(side?.vertexCount || 0);
});
const copyableMatch = computed(() => {
  if (!activePreset.value || !selectedMatch.value) return null;
  return {
    schemaVersion: 1,
    presetId: activePreset.value.id,
    direction: {
      source: activePreset.value.source.name,
      target: activePreset.value.target.name,
    },
    match: selectedMatch.value,
  };
});

let scene;
let camera3d;
let renderer;
let controls;
let resizeObserver;
let animationFrame;
let manifestRequest;
let geometryRequest;
let sourceModel;
let targetModel;
let correspondenceOverlays;
let inspectionMarker;
let combinedBounds;
let loadGeneration = 0;
let modelRadius = 1;
let componentActive = true;
let copyStatusTimer;
let debugApi;
const raycaster = new THREE.Raycaster();
const pointerNdc = new THREE.Vector2();

onMounted(async () => {
  await nextTick();
  initializeViewer();
  installViewerDebugApi();
  document.addEventListener("fullscreenchange", syncFullscreenState);
  await refreshPresets();
});

onActivated(() => {
  componentActive = true;
  nextTick(() => {
    resizeViewport();
    requestRender();
  });
});

onDeactivated(() => {
  componentActive = false;
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = undefined;
});

onBeforeUnmount(() => {
  componentActive = false;
  manifestRequest?.abort();
  geometryRequest?.abort();
  if (animationFrame) cancelAnimationFrame(animationFrame);
  animationFrame = undefined;
  if (copyStatusTimer) clearTimeout(copyStatusTimer);
  resizeObserver?.disconnect();
  document.removeEventListener("fullscreenchange", syncFullscreenState);
  uninstallViewerDebugApi();
  clearInspectionMarker();
  disposeCorrespondenceOverlays();
  disposeModels();
  controls?.dispose();
  renderer?.dispose();
  renderer?.forceContextLoss();
});

watch(selectedPresetKey, () => {
  if (selectedPresetKey.value) loadSelectedPreset();
});

watch(selectedMatchIndex, () => {
  rebuildCorrespondenceOverlays();
});

watch([showMatchLines, showErrorLines, showMarkers, isolateSelected], () => {
  rebuildCorrespondenceOverlays();
});

watch(meshOpacity, () => {
  applyMeshOpacity();
  requestRender();
});

watch(separation, () => {
  updateModelLayout();
  rebuildCorrespondenceOverlays();
  requestRender();
});

watch(inspectSide, () => {
  const maximum = Math.max(0, inspectVertexCount.value - 1);
  const current = Number.parseInt(vertexIndexInput.value, 10);
  if (!Number.isInteger(current) || current < 0 || current > maximum) vertexIndexInput.value = "0";
});

function initializeViewer() {
  if (!stage.value) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111719);
  camera3d = new THREE.PerspectiveCamera(38, 1, 0.01, 10000);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute("aria-label", "Interactive point correspondence viewport");
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.setAttribute(
    "title",
    "Click to inspect · Left-drag to rotate · Shift-left, middle, or right-drag to pan · Scroll to zoom · F to fit",
  );
  renderer.domElement.tabIndex = 0;
  stage.value.prepend(renderer.domElement);

  controls = new HtmlStyleControls(camera3d, renderer.domElement, {
    onChange: requestRender,
    onClick: (event) => {
      if (!loadingGeometry.value) inspectSurface(event);
    },
    onEscape: clearInspection,
  });

  resizeObserver = new ResizeObserver(resizeViewport);
  resizeObserver.observe(stage.value);
  resizeViewport();
  requestRender();
}

function resizeViewport() {
  if (!renderer || !camera3d || !stage.value) return;
  const width = Math.max(1, Math.floor(stage.value.clientWidth));
  const height = Math.max(1, Math.floor(stage.value.clientHeight));
  renderer.setSize(width, height, false);
  controls?.setViewport(width, height);
  requestRender();
}

async function toggleFullscreen() {
  if (!viewerFrame.value) return;
  if (document.fullscreenElement === viewerFrame.value) await document.exitFullscreen();
  else await viewerFrame.value.requestFullscreen();
}

function syncFullscreenState() {
  isFullscreen.value = document.fullscreenElement === viewerFrame.value;
  requestAnimationFrame(resizeViewport);
}

function requestRender() {
  if (!componentActive || animationFrame || !renderer || !scene || !camera3d) return;
  animationFrame = requestAnimationFrame(renderFrame);
}

function renderFrame() {
  animationFrame = undefined;
  if (!componentActive || !renderer || !scene || !camera3d) return;
  renderer.render(scene, camera3d);
}

function installViewerDebugApi() {
  if (!import.meta.env.DEV) return;
  debugApi = Object.freeze({
    getControlsState: () => controls?.getState(),
  });
  window.__BUGNIST_CORRESPONDENCE_VIEWER_DEBUG__ = debugApi;
}

function uninstallViewerDebugApi() {
  if (!debugApi || window.__BUGNIST_CORRESPONDENCE_VIEWER_DEBUG__ !== debugApi) return;
  delete window.__BUGNIST_CORRESPONDENCE_VIEWER_DEBUG__;
  debugApi = undefined;
}

async function refreshPresets() {
  manifestRequest?.abort();
  const request = new AbortController();
  manifestRequest = request;
  loadingPresets.value = true;
  loadError.value = "";

  try {
    const response = await fetch(PRESETS_URL, { signal: request.signal });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.error || "Could not load correspondence datasets.");
    validateManifest(payload);
    presets.value = payload.presets;
    if (!presets.value.length) throw new Error("No correspondence datasets were found.");

    const currentExists = presets.value.some((preset) => presetKey(preset) === selectedPresetKey.value);
    if (currentExists) await loadSelectedPreset();
    else selectedPresetKey.value = presetKey(presets.value[0]);
  } catch (error) {
    if (error.name !== "AbortError") loadError.value = error.message || String(error);
  } finally {
    if (manifestRequest === request) {
      manifestRequest = undefined;
      loadingPresets.value = false;
    }
  }
}

function validateManifest(payload) {
  if (payload?.schemaVersion !== 1) throw new Error("Unsupported correspondence preset schema.");
  const semantics = payload.semantics || {};
  if (
    semantics.coordinateSpace !== "per-geometry-local"
    || semantics.indexBase !== 0
    || semantics.sourceEndpoint !== "sourceVertex"
    || semantics.errorEndpoint !== "manualTarget"
    || semantics.targetGroundTruthVertexRole !== "snapped-diagnostic"
    || semantics.directionsIndependent !== true
  ) {
    throw new Error("The correspondence preset coordinate semantics are not supported.");
  }
  if (!Array.isArray(payload.presets)) throw new Error("Correspondence presets must be an array.");
}

async function loadSelectedPreset() {
  const preset = activePreset.value;
  if (!preset || !scene) return;

  geometryRequest?.abort();
  const request = new AbortController();
  const generation = ++loadGeneration;
  geometryRequest = request;
  loadingGeometry.value = true;
  loadError.value = "";
  selectedMatchIndex.value = 0;
  inspection.value = null;
  clearInspectionMarker();
  disposeCorrespondenceOverlays();
  disposeModels();
  requestRender();

  try {
    validatePresetShape(preset);
    const results = await Promise.allSettled([
      loadIndexedPly(preset.source, "source", request.signal),
      loadIndexedPly(preset.target, "target", request.signal),
    ]);
    if (generation !== loadGeneration || request.signal.aborted) {
      disposeSettledModels(results);
      return;
    }
    const failed = results.find((result) => result.status === "rejected");
    if (failed) {
      disposeSettledModels(results);
      throw failed.reason;
    }

    sourceModel = results[0].value;
    targetModel = results[1].value;
    validateMatchEndpoints(preset, sourceModel, targetModel);
    scene.add(sourceModel.root, targetModel.root);
    updateModelLayout();
    applyMeshOpacity();
    rebuildCorrespondenceOverlays();
    resetView();
  } catch (error) {
    disposeModels();
    if (error.name !== "AbortError") loadError.value = error.message || String(error);
  } finally {
    if (geometryRequest === request) {
      geometryRequest = undefined;
      loadingGeometry.value = false;
    }
    requestRender();
  }
}

function validatePresetShape(preset) {
  for (const side of ["source", "target"]) {
    const descriptor = preset?.[side];
    if (!descriptor?.displayGeometry?.assetUrl) {
      throw new Error(`${side} display geometry is missing an asset URL.`);
    }
    if (!Number.isInteger(descriptor.vertexCount) || descriptor.vertexCount < 1) {
      throw new Error(`${side} has an invalid vertex count.`);
    }
  }
  if (!Array.isArray(preset.matches) || !preset.matches.length) {
    throw new Error("This dataset does not contain any landmark matches.");
  }
}

async function loadIndexedPly(descriptor, side, signal) {
  const response = await fetch(descriptor.displayGeometry.assetUrl, { signal });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.error || `Could not load ${descriptor.displayGeometry.path || `${side} geometry`}.`);
  }
  const buffer = await response.arrayBuffer();
  let geometry;
  try {
    geometry = new PLYLoader().parse(buffer);
    validateIndexedGeometry(geometry, descriptor, side);
    const hasColors = geometry.getAttribute("color")?.count === geometry.getAttribute("position").count;
    const material = new THREE.MeshBasicMaterial({
      color: hasColors ? 0xffffff : side === "source" ? SOURCE_COLOR : TARGET_COLOR,
      side: THREE.DoubleSide,
      vertexColors: hasColors,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `${side}:${descriptor.name}`;
    mesh.userData.correspondenceSide = side;
    const root = new THREE.Group();
    root.name = `${side}-geometry-root`;
    root.userData.correspondenceSide = side;
    root.add(mesh);

    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const bounds = geometry.boundingBox.clone();
    return {
      descriptor,
      side,
      root,
      mesh,
      geometry,
      positions: geometry.getAttribute("position"),
      bounds,
      center: bounds.getCenter(new THREE.Vector3()),
      size: bounds.getSize(new THREE.Vector3()),
    };
  } catch (error) {
    geometry?.dispose();
    throw error;
  }
}

function validateIndexedGeometry(geometry, descriptor, side) {
  const positions = geometry?.getAttribute("position");
  if (!positions || positions.itemSize !== 3 || !positions.count) {
    throw new Error(`${side} PLY does not contain XYZ positions.`);
  }
  if (positions.count !== descriptor.vertexCount) {
    throw new Error(
      `${side} PLY has ${positions.count.toLocaleString()} vertices; the preset requires ${descriptor.vertexCount.toLocaleString()}.`,
    );
  }
  if (!geometry.index || geometry.index.count < 3 || geometry.index.count % 3 !== 0) {
    throw new Error(`${side} display geometry must be an indexed triangle PLY.`);
  }
  for (let index = 0; index < positions.count; index += 1) {
    if (!Number.isFinite(positions.getX(index)) || !Number.isFinite(positions.getY(index)) || !Number.isFinite(positions.getZ(index))) {
      throw new Error(`${side} PLY contains a non-finite coordinate at vertex ${index}.`);
    }
  }
  const indices = geometry.index.array;
  for (let offset = 0; offset < indices.length; offset += 1) {
    if (indices[offset] < 0 || indices[offset] >= positions.count) {
      throw new Error(`${side} PLY contains an invalid face index at offset ${offset}.`);
    }
  }
}

function validateMatchEndpoints(preset, source, target) {
  preset.matches.forEach((match, matchIndex) => {
    const label = match.label || `match ${matchIndex + 1}`;
    validateCoordinate(match.sourceManual, `${label} source manual annotation`);
    validateCoordinate(match.manualTarget, `${label} manual target annotation`);
    assertIndexedEndpoint(source, match.sourceIndex, match.sourceVertex, `${label} source`);
    assertIndexedEndpoint(
      target,
      match.predictedTargetIndex,
      match.predictedTarget,
      `${label} predicted target`,
    );
    assertIndexedEndpoint(
      target,
      match.targetGroundTruthIndex,
      match.targetGroundTruthVertex,
      `${label} snapped target GT`,
    );
  });
}

function validateCoordinate(coordinate, label) {
  if (!Array.isArray(coordinate) || coordinate.length !== 3 || coordinate.some((value) => !Number.isFinite(value))) {
    throw new Error(`${label} is not a finite XYZ coordinate.`);
  }
}

function assertIndexedEndpoint(model, index, expected, label) {
  if (!Number.isInteger(index) || index < 0 || index >= model.positions.count) {
    throw new Error(`${label} index ${index} is outside the ${model.side} geometry.`);
  }
  validateCoordinate(expected, `${label} endpoint`);
  const actual = positionArray(model.positions, index);
  const difference = Math.max(...actual.map((value, axis) => Math.abs(value - expected[axis])));
  if (difference > ENDPOINT_TOLERANCE) {
    throw new Error(
      `${label} index ${index} differs from the display PLY by ${formatNumber(difference, 6)}; check that this is the benchmark's exact geometry.`,
    );
  }
}

function disposeSettledModels(results) {
  for (const result of results) {
    if (result.status === "fulfilled") disposeModel(result.value);
  }
}

function disposeModels() {
  if (sourceModel) disposeModel(sourceModel);
  if (targetModel) disposeModel(targetModel);
  sourceModel = undefined;
  targetModel = undefined;
  combinedBounds = undefined;
}

function disposeModel(model) {
  scene?.remove(model.root);
  model.geometry?.dispose();
  disposeMaterial(model.mesh?.material);
}

function updateModelLayout() {
  if (!sourceModel || !targetModel) return;
  const sourceWidth = Math.max(sourceModel.size.x, 0.001);
  const targetWidth = Math.max(targetModel.size.x, 0.001);
  const sourceDiagonal = sourceModel.size.length();
  const targetDiagonal = targetModel.size.length();
  const sharedDiagonal = Math.max(sourceDiagonal, targetDiagonal, 0.001);
  const baseDistance = (sourceWidth + targetWidth) * 0.5 + sharedDiagonal * 0.16;
  const centerDistance = baseDistance * Number(separation.value);

  sourceModel.root.position.set(-centerDistance * 0.5, 0, 0).sub(sourceModel.center);
  targetModel.root.position.set(centerDistance * 0.5, 0, 0).sub(targetModel.center);
  sourceModel.root.updateMatrixWorld(true);
  targetModel.root.updateMatrixWorld(true);

  combinedBounds = new THREE.Box3()
    .setFromObject(sourceModel.root)
    .union(new THREE.Box3().setFromObject(targetModel.root));
  const sphere = combinedBounds.getBoundingSphere(new THREE.Sphere());
  modelRadius = Math.max(sphere.radius || 1, 0.001);
  controls?.setBounds(combinedBounds, 0.18, 14);
}

function applyMeshOpacity() {
  const opacity = Number(meshOpacity.value);
  for (const model of [sourceModel, targetModel]) {
    const material = model?.mesh?.material;
    if (!material) continue;
    material.opacity = opacity;
    material.transparent = opacity < 0.999;
    material.depthWrite = opacity >= 0.999;
    material.needsUpdate = true;
  }
}

function resetView() {
  if (!controls || !combinedBounds) return;
  controls.fit(combinedBounds);
  requestRender();
}

function frontView() {
  if (!controls || !combinedBounds) return;
  controls.setView(0, 0);
}

function sideView() {
  if (!controls || !combinedBounds) return;
  controls.setView(Math.PI / 2, 0);
}

function rebuildCorrespondenceOverlays() {
  disposeCorrespondenceOverlays();
  const preset = activePreset.value;
  if (!scene || !sourceModel || !targetModel || !preset?.matches?.length) {
    requestRender();
    return;
  }

  sourceModel.root.updateMatrixWorld(true);
  targetModel.root.updateMatrixWorld(true);
  correspondenceOverlays = new THREE.Group();
  correspondenceOverlays.name = "correspondence-overlays";

  const indexes = isolateSelected.value
    ? [Math.min(selectedMatchIndex.value, preset.matches.length - 1)]
    : preset.matches.map((_match, index) => index);

  if (showMatchLines.value) {
    const vertices = [];
    const colors = [];
    for (const index of indexes) {
      const match = preset.matches[index];
      const sourcePoint = localToWorld(sourceModel, match.sourceVertex);
      const targetPoint = localToWorld(targetModel, match.predictedTarget);
      vertices.push(...sourcePoint.toArray(), ...targetPoint.toArray());
      const color = index === selectedMatchIndex.value ? SELECTED_MATCH_COLOR : MATCH_COLOR;
      colors.push(...color.toArray(), ...color.toArray());
    }
    correspondenceOverlays.add(makeLineSegments(vertices, colors, 0.8, "predicted-correspondence-lines"));
  }

  if (showErrorLines.value) {
    const vertices = [];
    const colors = [];
    for (const index of indexes) {
      const match = preset.matches[index];
      const predicted = localToWorld(targetModel, match.predictedTarget);
      const manualTarget = localToWorld(targetModel, match.manualTarget);
      vertices.push(...predicted.toArray(), ...manualTarget.toArray());
      const color = index === selectedMatchIndex.value ? SELECTED_ERROR_COLOR : ERROR_COLOR;
      colors.push(...color.toArray(), ...color.toArray());
    }
    correspondenceOverlays.add(makeLineSegments(vertices, colors, 0.92, "manual-target-error-lines"));
  }

  if (showMarkers.value) {
    const markerRadius = Math.max(modelRadius * 0.012, 0.001);
    const sourcePoints = [];
    const predictedPoints = [];
    const groundTruthPoints = [];
    const selectedFlags = [];
    for (const index of indexes) {
      const match = preset.matches[index];
      sourcePoints.push(localToWorld(sourceModel, match.sourceVertex));
      predictedPoints.push(localToWorld(targetModel, match.predictedTarget));
      groundTruthPoints.push(localToWorld(targetModel, match.manualTarget));
      selectedFlags.push(index === selectedMatchIndex.value);
    }
    correspondenceOverlays.add(
      makeMarkers(sourcePoints, selectedFlags, markerRadius, SOURCE_MARKER_COLOR, "source-markers"),
      makeMarkers(predictedPoints, selectedFlags, markerRadius, PREDICTED_MARKER_COLOR, "predicted-markers"),
      makeMarkers(groundTruthPoints, selectedFlags, markerRadius, GROUND_TRUTH_MARKER_COLOR, "manual-gt-markers"),
    );
  }

  scene.add(correspondenceOverlays);
  requestRender();
}

function makeLineSegments(vertices, colors, opacity, name) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  const material = new THREE.LineBasicMaterial({
    opacity,
    transparent: true,
    vertexColors: true,
  });
  const lines = new THREE.LineSegments(geometry, material);
  lines.name = name;
  lines.renderOrder = 3;
  return lines;
}

function makeMarkers(points, selectedFlags, radius, color, name) {
  const geometry = new THREE.SphereGeometry(radius, 14, 10);
  const material = new THREE.MeshBasicMaterial({ color });
  const markers = new THREE.InstancedMesh(geometry, material, points.length);
  const matrix = new THREE.Matrix4();
  for (let index = 0; index < points.length; index += 1) {
    const scale = selectedFlags[index] ? 1.5 : 1;
    matrix.compose(points[index], new THREE.Quaternion(), new THREE.Vector3(scale, scale, scale));
    markers.setMatrixAt(index, matrix);
  }
  markers.instanceMatrix.needsUpdate = true;
  markers.name = name;
  markers.renderOrder = 4;
  return markers;
}

function disposeCorrespondenceOverlays() {
  if (!correspondenceOverlays) return;
  scene?.remove(correspondenceOverlays);
  disposeObjectResources(correspondenceOverlays);
  correspondenceOverlays = undefined;
}

function localToWorld(model, coordinate) {
  return model.root.localToWorld(new THREE.Vector3(coordinate[0], coordinate[1], coordinate[2]));
}

function inspectSurface(event) {
  if (!renderer || !camera3d || !sourceModel || !targetModel) return;
  const bounds = renderer.domElement.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  pointerNdc.set(
    ((event.clientX - bounds.left) / bounds.width) * 2 - 1,
    -((event.clientY - bounds.top) / bounds.height) * 2 + 1,
  );
  raycaster.setFromCamera(pointerNdc, camera3d);
  const hit = raycaster.intersectObjects([sourceModel.mesh, targetModel.mesh], false)[0];
  if (!hit) return;

  const side = hit.object.userData.correspondenceSide;
  const model = side === "source" ? sourceModel : targetModel;
  const localPoint = hit.object.worldToLocal(hit.point.clone());
  const nearest = nearestVertex(model.positions, localPoint);
  const triangle = hit.face ? triangleDetails(model.positions, hit.face, localPoint) : null;

  inspectSide.value = side;
  vertexIndexInput.value = String(nearest.index);
  inspection.value = {
    schemaVersion: 1,
    presetId: activePreset.value?.id,
    side,
    geometry: model.descriptor.indexGeometry.path,
    mode: "surface-click",
    surface: localPoint.toArray(),
    displayWorld: hit.point.toArray(),
    nearestVertex: {
      index: nearest.index,
      xyz: nearest.xyz,
      distance: Math.sqrt(nearest.distanceSquared),
    },
    triangle,
  };
  setInspectionMarker(model, localPoint);
}

function nearestVertex(positions, point) {
  let nearestIndex = -1;
  let nearestDistanceSquared = Number.POSITIVE_INFINITY;
  for (let index = 0; index < positions.count; index += 1) {
    const dx = positions.getX(index) - point.x;
    const dy = positions.getY(index) - point.y;
    const dz = positions.getZ(index) - point.z;
    const distanceSquared = dx * dx + dy * dy + dz * dz;
    if (distanceSquared < nearestDistanceSquared) {
      nearestDistanceSquared = distanceSquared;
      nearestIndex = index;
    }
  }
  return {
    index: nearestIndex,
    xyz: positionArray(positions, nearestIndex),
    distanceSquared: nearestDistanceSquared,
  };
}

function triangleDetails(positions, face, point) {
  const vertexIndices = [face.a, face.b, face.c];
  const a = positionVector(positions, face.a);
  const b = positionVector(positions, face.b);
  const c = positionVector(positions, face.c);
  const barycentric = THREE.Triangle.getBarycoord(point, a, b, c, new THREE.Vector3());
  return {
    vertexIndices,
    barycentric: barycentric ? barycentric.toArray() : null,
  };
}

function inspectKnownVertex() {
  const model = inspectSide.value === "source" ? sourceModel : targetModel;
  if (!model) return;
  const index = Number.parseInt(vertexIndexInput.value, 10);
  if (!Number.isInteger(index) || index < 0 || index >= model.positions.count) {
    loadError.value = `Vertex index must be between 0 and ${(model.positions.count - 1).toLocaleString()}.`;
    return;
  }
  loadError.value = "";
  selectKnownVertex(model, index, "known-vertex");
}

function inspectRandomVertex() {
  const model = inspectSide.value === "source" ? sourceModel : targetModel;
  if (!model?.positions.count) return;
  const index = Math.floor(Math.random() * model.positions.count);
  vertexIndexInput.value = String(index);
  selectKnownVertex(model, index, "random-vertex");
}

function selectKnownVertex(model, index, mode) {
  const localPoint = positionVector(model.positions, index);
  const displayWorld = model.root.localToWorld(localPoint.clone());
  inspection.value = {
    schemaVersion: 1,
    presetId: activePreset.value?.id,
    side: model.side,
    geometry: model.descriptor.indexGeometry.path,
    mode,
    surface: localPoint.toArray(),
    displayWorld: displayWorld.toArray(),
    nearestVertex: {
      index,
      xyz: localPoint.toArray(),
      distance: 0,
    },
    triangle: null,
  };
  setInspectionMarker(model, localPoint);
}

function setInspectionMarker(model, localPoint) {
  clearInspectionMarker();
  const geometry = new THREE.SphereGeometry(Math.max(modelRadius * 0.016, 0.001), 16, 12);
  const material = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false });
  inspectionMarker = new THREE.Mesh(geometry, material);
  inspectionMarker.name = "inspected-surface-point";
  inspectionMarker.position.copy(localPoint);
  inspectionMarker.renderOrder = 20;
  model.root.add(inspectionMarker);
  requestRender();
}

function clearInspectionMarker() {
  if (!inspectionMarker) return;
  inspectionMarker.parent?.remove(inspectionMarker);
  inspectionMarker.geometry?.dispose();
  disposeMaterial(inspectionMarker.material);
  inspectionMarker = undefined;
  requestRender();
}

function clearInspection() {
  inspection.value = null;
  clearInspectionMarker();
}

function focusInspection() {
  if (!inspection.value || !controls) return;
  const model = inspection.value.side === "source" ? sourceModel : targetModel;
  if (!model) return;
  const local = inspection.value.surface;
  const target = model.root.localToWorld(new THREE.Vector3(local[0], local[1], local[2]));
  controls.focus(target);
}

async function copyJson(value, successMessage) {
  if (!value) return;
  const text = JSON.stringify(value, null, 2);
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    textarea.remove();
  }
  copyStatus.value = successMessage;
  if (copyStatusTimer) clearTimeout(copyStatusTimer);
  copyStatusTimer = setTimeout(() => {
    copyStatus.value = "";
  }, 1800);
}

function selectMatch(index) {
  selectedMatchIndex.value = index;
}

function presetKey(preset) {
  return `${preset.datasetId || "correspondence"}::${preset.id}`;
}

function positionArray(attribute, index) {
  return [attribute.getX(index), attribute.getY(index), attribute.getZ(index)];
}

function positionVector(attribute, index) {
  return new THREE.Vector3(attribute.getX(index), attribute.getY(index), attribute.getZ(index));
}

function disposeObjectResources(object) {
  const geometries = new Set();
  const materials = new Set();
  object.traverse((child) => {
    if (child.geometry) geometries.add(child.geometry);
    for (const material of arrayValue(child.material)) materials.add(material);
  });
  for (const geometry of geometries) geometry.dispose();
  for (const material of materials) material?.dispose();
}

function disposeMaterial(material) {
  for (const item of arrayValue(material)) item?.dispose();
}

function arrayValue(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function formatNumber(value, digits = 4) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return "—";
  if (numeric !== 0 && Math.abs(numeric) < 10 ** -digits) return numeric.toExponential(2);
  return numeric.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatCoordinate(coordinate) {
  if (!Array.isArray(coordinate)) return "—";
  return coordinate.map((value) => formatNumber(value, 6)).join(", ");
}

function formatPercent(value, digits = 2) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? `${numeric.toFixed(digits)}%` : "—";
}
</script>

<template>
  <article ref="viewerFrame" class="correspondence-viewer" :class="{ fullscreen: isFullscreen }">
    <header class="viewer-header">
      <div class="viewer-title">
        <span class="viewer-kicker"><Activity :size="15" />Landmark correspondence viewer</span>
        <h2>{{ activePreset?.label || "Point-to-point correspondence" }}</h2>
        <p>Prediction lines connect the two original coordinate systems; error lines end at the manual annotation.</p>
      </div>
      <div class="header-actions">
        <button class="icon-button" type="button" title="Refresh datasets" :disabled="loadingPresets" @click="refreshPresets">
          <RefreshCw :class="{ spin: loadingPresets }" :size="18" />
        </button>
        <button class="icon-button" type="button" :title="isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'" @click="toggleFullscreen">
          <Minimize2 v-if="isFullscreen" :size="18" />
          <Expand v-else :size="18" />
        </button>
      </div>
    </header>

    <div class="dataset-bar">
      <label>
        <span>Dataset direction</span>
        <select v-model="selectedPresetKey" :disabled="loadingPresets || !presets.length">
          <optgroup v-for="collection in presetCollections" :key="collection.id" :label="collection.title">
            <option v-for="preset in collection.presets" :key="presetKey(preset)" :value="presetKey(preset)">
              {{ preset.label }}
            </option>
          </optgroup>
        </select>
      </label>
      <div v-if="activePreset" class="dataset-meta">
        <span>{{ activePreset.matches.length }} landmarks</span>
        <span>0-based indices</span>
        <span>original XYZ</span>
      </div>
    </div>

    <div v-if="loadError" class="viewer-error" role="alert">
      <AlertTriangle :size="18" />
      <span>{{ loadError }}</span>
    </div>

    <div class="viewer-workspace">
      <div
        ref="stage"
        class="viewer-stage"
        title="Click to inspect · Left-drag to rotate · Shift-left, middle, or right-drag to pan · Scroll to zoom · F to fit"
      >
        <div v-if="activePreset" class="canvas-chips" aria-hidden="true">
          <span class="source-chip"><i />{{ activePreset.source.name }}</span>
          <span class="target-chip"><i />{{ activePreset.target.name }}</span>
        </div>
        <div v-if="loadingPresets || loadingGeometry" class="stage-state">
          <Loader2 class="spin" :size="25" />
          {{ loadingPresets ? "Loading datasets" : "Validating indexed geometry" }}
        </div>
        <div v-else-if="!activePreset && !loadError" class="stage-state">No correspondence dataset selected</div>
        <div class="canvas-help"><MousePointer2 :size="14" />Click inspects · drag rotates · Shift/right-drag pans · wheel zooms</div>
      </div>

      <aside class="inspector" aria-label="Correspondence and coordinate inspector">
        <section class="inspector-section display-controls">
          <div class="section-heading">
            <div><Eye :size="16" /><strong>Display</strong></div>
            <div class="view-buttons" aria-label="Camera views">
              <button type="button" title="Fit current view (F)" @click="resetView"><RotateCcw :size="14" />Fit</button>
              <button type="button" @click="frontView">Front</button>
              <button type="button" @click="sideView">Side</button>
            </div>
          </div>
          <div class="toggle-grid">
            <label><input v-model="showMatchLines" type="checkbox" />Matches</label>
            <label><input v-model="showErrorLines" type="checkbox" />Manual error</label>
            <label><input v-model="showMarkers" type="checkbox" />Markers</label>
            <label><input v-model="isolateSelected" type="checkbox" />Isolate selected</label>
          </div>
          <label class="range-control">
            <span>Mesh opacity <output>{{ Number(meshOpacity).toFixed(2) }}</output></span>
            <input v-model.number="meshOpacity" type="range" min="0.15" max="1" step="0.05" />
          </label>
          <label class="range-control">
            <span>Separation <output>{{ Number(separation).toFixed(2) }}×</output></span>
            <input v-model.number="separation" type="range" min="0.75" max="2.2" step="0.05" />
          </label>
          <div class="legend" aria-label="Correspondence legend">
            <span><i class="marker source" />Source</span>
            <span><i class="marker predicted" />Prediction</span>
            <span><i class="marker ground-truth" />Manual GT</span>
          </div>
        </section>

        <section class="inspector-section match-section">
          <div class="section-heading">
            <div><Target :size="16" /><strong>Landmarks</strong></div>
            <span>{{ selectedMatchIndex + 1 }} / {{ matches.length || 0 }}</span>
          </div>
          <select v-model.number="selectedMatchIndex" class="match-select" :disabled="!matches.length">
            <option v-for="(match, index) in matches" :key="`${match.label}:${index}`" :value="index">
              {{ index + 1 }}. {{ match.label }}
            </option>
          </select>
          <div class="match-list">
            <button
              v-for="(match, index) in matches"
              :key="`${match.label}:row:${index}`"
              type="button"
              :class="{ selected: index === selectedMatchIndex }"
              @click="selectMatch(index)"
            >
              <span><strong>{{ match.label }}</strong><small>#{{ match.sourceIndex }} → #{{ match.predictedTargetIndex }}</small></span>
              <em>{{ formatPercent(match.metrics?.targetErrorBBoxPct) }}</em>
            </button>
          </div>
        </section>

        <section v-if="selectedMatch" class="inspector-section selected-details">
          <div class="section-heading">
            <div><Crosshair :size="16" /><strong>{{ selectedMatch.label }}</strong></div>
            <button type="button" title="Copy selected match as JSON" @click="copyJson(copyableMatch, 'Match JSON copied')">
              <Copy :size="15" />Copy JSON
            </button>
          </div>
          <dl class="metric-grid">
            <div><dt>Cosine</dt><dd>{{ formatNumber(selectedMatch.metrics?.cosineScore, 5) }}</dd></div>
            <div><dt>NN margin</dt><dd>{{ formatNumber(selectedMatch.metrics?.nearestNeighborMargin, 5) }}</dd></div>
            <div><dt>GT feature rank</dt><dd>{{ formatNumber(selectedMatch.metrics?.groundTruthFeatureRank, 0) }}</dd></div>
            <div><dt>Manual error</dt><dd>{{ formatNumber(selectedMatch.metrics?.targetError, 4) }}</dd></div>
            <div><dt>Error / bbox</dt><dd>{{ formatPercent(selectedMatch.metrics?.targetErrorBBoxPct) }}</dd></div>
            <div><dt>Vertex error</dt><dd>{{ formatNumber(selectedMatch.metrics?.targetVertexError, 4) }}</dd></div>
          </dl>
          <div class="index-grid">
            <span>Source index<strong>{{ selectedMatch.sourceIndex }}</strong></span>
            <span>Predicted index<strong>{{ selectedMatch.predictedTargetIndex }}</strong></span>
            <span>Snapped GT index<strong>{{ selectedMatch.targetGroundTruthIndex }}</strong></span>
          </div>
          <div v-if="selectedPckEntries.length" class="pck-row">
            <span
              v-for="([threshold, passed]) in selectedPckEntries"
              :key="threshold"
              :class="{ passed }"
            >
              <Check v-if="passed" :size="12" />PCK {{ threshold }}
            </span>
          </div>
          <details>
            <summary>Endpoint coordinates</summary>
            <dl class="coordinate-list">
              <div><dt>Source vertex</dt><dd>{{ formatCoordinate(selectedMatch.sourceVertex) }}</dd></div>
              <div><dt>Predicted target</dt><dd>{{ formatCoordinate(selectedMatch.predictedTarget) }}</dd></div>
              <div><dt>Manual target</dt><dd>{{ formatCoordinate(selectedMatch.manualTarget) }}</dd></div>
              <div><dt>Snapped GT vertex</dt><dd>{{ formatCoordinate(selectedMatch.targetGroundTruthVertex) }}</dd></div>
            </dl>
          </details>
        </section>

        <section class="inspector-section point-inspector">
          <div class="section-heading">
            <div><MousePointer2 :size="16" /><strong>Point inspector</strong></div>
            <span>global nearest vertex</span>
          </div>
          <p>Click either surface, or inspect an exact 0-based geometry row.</p>
          <div class="vertex-tools">
            <select v-model="inspectSide" aria-label="Geometry to inspect">
              <option value="source">Source</option>
              <option value="target">Target</option>
            </select>
            <input
              v-model="vertexIndexInput"
              type="number"
              min="0"
              :max="Math.max(0, inspectVertexCount - 1)"
              step="1"
              aria-label="Original vertex index"
              @keydown.enter="inspectKnownVertex"
            />
            <button type="button" title="Inspect vertex index" :disabled="!inspectVertexCount" @click="inspectKnownVertex">
              <Target :size="15" />Go
            </button>
            <button type="button" title="Inspect a random vertex" :disabled="!inspectVertexCount" @click="inspectRandomVertex">
              <Shuffle :size="15" />Random
            </button>
          </div>

          <div v-if="inspection" class="inspection-card">
            <div class="inspection-heading">
              <span :class="inspection.side">{{ inspection.side }}</span>
              <strong>#{{ inspection.nearestVertex.index }}</strong>
              <small>{{ inspection.mode.replaceAll("-", " ") }}</small>
            </div>
            <dl class="coordinate-list">
              <div><dt>Surface XYZ</dt><dd>{{ formatCoordinate(inspection.surface) }}</dd></div>
              <div><dt>Nearest vertex XYZ</dt><dd>{{ formatCoordinate(inspection.nearestVertex.xyz) }}</dd></div>
              <div><dt>Vertex distance</dt><dd>{{ formatNumber(inspection.nearestVertex.distance, 6) }}</dd></div>
              <div v-if="inspection.triangle"><dt>Triangle vertices</dt><dd>{{ inspection.triangle.vertexIndices.join(", ") }}</dd></div>
              <div v-if="inspection.triangle"><dt>Barycentric</dt><dd>{{ formatCoordinate(inspection.triangle.barycentric) }}</dd></div>
            </dl>
            <div class="inspection-actions">
              <button type="button" @click="focusInspection"><Crosshair :size="15" />Focus</button>
              <button type="button" @click="copyJson(inspection, 'Point JSON copied')"><Copy :size="15" />Copy JSON</button>
            </div>
          </div>
          <div v-else class="inspection-empty"><Crosshair :size="18" />No point selected yet</div>
        </section>

        <section v-if="activePreset?.summary" class="inspector-section summary-section">
          <div class="section-heading"><div><Activity :size="16" /><strong>Direction summary</strong></div></div>
          <dl class="metric-grid">
            <div><dt>Exact top-1</dt><dd>{{ formatPercent(activePreset.summary.top1ExactIndexRatio * 100) }}</dd></div>
            <div><dt>Mean bbox error</dt><dd>{{ formatNumber(activePreset.summary.statistics?.targetErrorBBox?.mean, 5) }}</dd></div>
            <div><dt>Median bbox error</dt><dd>{{ formatNumber(activePreset.summary.statistics?.targetErrorBBox?.median, 5) }}</dd></div>
            <div><dt>Mean cosine</dt><dd>{{ formatNumber(activePreset.summary.statistics?.cosineScore?.mean, 5) }}</dd></div>
          </dl>
          <div v-if="summaryPckEntries.length" class="summary-pck">
            <span v-for="([threshold, value]) in summaryPckEntries" :key="threshold">
              PCK {{ threshold }}<strong>{{ formatPercent(value * 100) }}</strong>
            </span>
          </div>
        </section>

        <div v-if="copyStatus" class="copy-status" role="status"><Check :size="15" />{{ copyStatus }}</div>
      </aside>
    </div>
  </article>
</template>

<style scoped>
.correspondence-viewer {
  background: var(--panel, #ffffff);
  border: 1px solid var(--line, #d9e0df);
  box-shadow: var(--shadow, 0 16px 40px rgba(24, 38, 42, 0.08));
  min-width: 0;
}

.viewer-header {
  align-items: flex-start;
  border-bottom: 1px solid var(--line, #d9e0df);
  display: flex;
  gap: 18px;
  justify-content: space-between;
  padding: 16px 18px 14px;
}

.viewer-title {
  min-width: 0;
}

.viewer-kicker {
  align-items: center;
  color: var(--teal, #1f7676);
  display: flex;
  font-size: 11px;
  font-weight: 750;
  gap: 6px;
  letter-spacing: 0.07em;
  margin-bottom: 4px;
  text-transform: uppercase;
}

.viewer-title h2 {
  font-size: 19px;
  line-height: 1.25;
  margin: 0;
  overflow-wrap: anywhere;
}

.viewer-title p {
  color: var(--muted, #67717b);
  font-size: 12px;
  line-height: 1.45;
  margin: 5px 0 0;
}

.header-actions {
  display: flex;
  flex: 0 0 auto;
  gap: 7px;
}

button,
input,
select {
  font: inherit;
}

button {
  cursor: pointer;
}

button:disabled,
select:disabled,
input:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.icon-button {
  align-items: center;
  background: #f7f9f8;
  border: 1px solid var(--line, #d9e0df);
  color: var(--ink, #20262d);
  display: inline-flex;
  flex: 0 0 auto;
  height: 38px;
  justify-content: center;
  padding: 0;
  width: 38px;
}

.icon-button:hover:not(:disabled) {
  background: var(--panel-soft, #eef4f2);
  color: var(--teal-dark, #175c5d);
}

.dataset-bar {
  align-items: end;
  background: #f8faf9;
  border-bottom: 1px solid var(--line, #d9e0df);
  display: grid;
  gap: 14px;
  grid-template-columns: minmax(260px, 1fr) auto;
  padding: 10px 14px;
}

.dataset-bar label {
  color: var(--muted, #67717b);
  display: grid;
  font-size: 11px;
  font-weight: 700;
  gap: 4px;
  letter-spacing: 0.03em;
  text-transform: uppercase;
}

.dataset-bar select,
.match-select,
.vertex-tools select,
.vertex-tools input {
  background: #ffffff;
  border: 1px solid var(--line, #d9e0df);
  color: var(--ink, #20262d);
  min-height: 36px;
  min-width: 0;
  padding: 6px 8px;
  text-transform: none;
  width: 100%;
}

.dataset-meta {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  justify-content: flex-end;
}

.dataset-meta span,
.canvas-chips span {
  background: #eef4f2;
  border: 1px solid #d2dfdc;
  color: #47605f;
  font-size: 10px;
  font-weight: 700;
  padding: 4px 7px;
  text-transform: uppercase;
}

.viewer-error {
  align-items: flex-start;
  background: #fff3f0;
  border-bottom: 1px solid #ecc7c0;
  color: #8a3933;
  display: flex;
  font-size: 13px;
  gap: 8px;
  line-height: 1.45;
  padding: 10px 14px;
}

.viewer-error svg {
  flex: 0 0 auto;
  margin-top: 1px;
}

.viewer-workspace {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 350px;
  min-width: 0;
}

.viewer-stage {
  background: #111719;
  height: clamp(560px, 72vh, 760px);
  min-width: 0;
  overflow: hidden;
  position: relative;
  touch-action: none;
}

.viewer-stage :deep(canvas:focus-visible) {
  outline: 3px solid rgba(31, 118, 118, 0.85);
  outline-offset: -3px;
}

.viewer-stage :deep(canvas) {
  display: block;
  height: 100%;
  width: 100%;
}

.canvas-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  left: 10px;
  pointer-events: none;
  position: absolute;
  top: 10px;
  z-index: 2;
}

.canvas-chips span {
  align-items: center;
  background: rgba(247, 251, 250, 0.9);
  border-color: rgba(255, 255, 255, 0.25);
  display: inline-flex;
  gap: 5px;
  max-width: min(240px, 44vw);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.canvas-chips i {
  background: #2f9b92;
  border-radius: 50%;
  height: 7px;
  width: 7px;
}

.canvas-chips .target-chip i {
  background: #5277b8;
}

.stage-state {
  align-items: center;
  background: rgba(17, 23, 25, 0.78);
  color: #e4ecea;
  display: flex;
  font-size: 13px;
  gap: 9px;
  inset: 0;
  justify-content: center;
  position: absolute;
  z-index: 3;
}

.canvas-help {
  align-items: center;
  background: rgba(17, 23, 25, 0.76);
  bottom: 10px;
  color: #c7d2d0;
  display: flex;
  font-size: 10px;
  gap: 6px;
  left: 10px;
  padding: 5px 7px;
  pointer-events: none;
  position: absolute;
  z-index: 2;
}

.inspector {
  background: #f7f9f8;
  border-left: 1px solid #354246;
  height: clamp(560px, 72vh, 760px);
  min-width: 0;
  overflow-y: auto;
  position: relative;
}

.correspondence-viewer.fullscreen {
  background: #111719;
  border: 0;
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
}

.fullscreen .viewer-header,
.fullscreen .dataset-bar,
.fullscreen .viewer-error {
  flex: 0 0 auto;
}

.fullscreen .viewer-workspace {
  flex: 1 1 auto;
  grid-template-columns: minmax(0, 1fr) clamp(330px, 29vw, 430px);
  min-height: 0;
}

.fullscreen .viewer-stage,
.fullscreen .inspector {
  height: 100%;
  min-height: 0;
}

.inspector-section {
  border-bottom: 1px solid var(--line, #d9e0df);
  padding: 13px;
}

.section-heading {
  align-items: center;
  display: flex;
  gap: 8px;
  justify-content: space-between;
  margin-bottom: 10px;
}

.section-heading > div {
  align-items: center;
  display: flex;
  gap: 7px;
  min-width: 0;
}

.section-heading .view-buttons {
  flex: 0 0 auto;
  gap: 4px;
}

.view-buttons button {
  min-width: 0;
  padding-inline: 6px;
}

.section-heading strong {
  font-size: 13px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.section-heading > span {
  color: var(--muted, #67717b);
  font-size: 10px;
  text-transform: uppercase;
}

.section-heading button,
.vertex-tools button,
.inspection-actions button {
  align-items: center;
  background: #ffffff;
  border: 1px solid var(--line, #d9e0df);
  color: var(--ink, #20262d);
  display: inline-flex;
  font-size: 11px;
  gap: 5px;
  justify-content: center;
  min-height: 31px;
  padding: 5px 7px;
}

.section-heading button:hover,
.vertex-tools button:hover:not(:disabled),
.inspection-actions button:hover {
  background: var(--panel-soft, #eef4f2);
  color: var(--teal-dark, #175c5d);
}

.toggle-grid {
  display: grid;
  gap: 6px;
  grid-template-columns: 1fr 1fr;
}

.toggle-grid label {
  align-items: center;
  background: #ffffff;
  border: 1px solid var(--line, #d9e0df);
  display: flex;
  font-size: 11px;
  gap: 6px;
  min-height: 31px;
  padding: 5px 7px;
}

.toggle-grid input {
  accent-color: var(--teal, #1f7676);
}

.range-control {
  display: grid;
  gap: 4px;
  margin-top: 10px;
}

.range-control span {
  align-items: center;
  color: var(--muted, #67717b);
  display: flex;
  font-size: 10px;
  justify-content: space-between;
  text-transform: uppercase;
}

.range-control output {
  color: var(--ink, #20262d);
  font-weight: 700;
}

.range-control input {
  accent-color: var(--teal, #1f7676);
  width: 100%;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 9px;
}

.legend span {
  align-items: center;
  color: var(--muted, #67717b);
  display: flex;
  font-size: 10px;
  gap: 5px;
}

.marker {
  background: #30c5b5;
  border-radius: 50%;
  display: inline-block;
  height: 8px;
  width: 8px;
}

.marker.predicted {
  background: #ffd166;
}

.marker.ground-truth {
  background: #ff6f61;
}

.match-select {
  display: none;
  margin-bottom: 8px;
}

.match-list {
  border: 1px solid var(--line, #d9e0df);
  max-height: 190px;
  overflow-y: auto;
}

.match-list button {
  align-items: center;
  background: #ffffff;
  border: 0;
  border-bottom: 1px solid var(--line, #d9e0df);
  color: var(--ink, #20262d);
  display: flex;
  gap: 8px;
  justify-content: space-between;
  padding: 7px 8px;
  text-align: left;
  width: 100%;
}

.match-list button:last-child {
  border-bottom: 0;
}

.match-list button:hover,
.match-list button.selected {
  background: #e8f3f0;
}

.match-list button.selected {
  box-shadow: inset 3px 0 var(--teal, #1f7676);
}

.match-list button span {
  display: grid;
  min-width: 0;
}

.match-list strong {
  font-size: 11px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.match-list small {
  color: var(--muted, #67717b);
  font-size: 9px;
}

.match-list em {
  color: var(--amber, #b96f26);
  flex: 0 0 auto;
  font-size: 10px;
  font-style: normal;
  font-weight: 700;
}

.metric-grid {
  display: grid;
  gap: 1px;
  grid-template-columns: 1fr 1fr;
  margin: 0;
}

.metric-grid > div {
  background: #ffffff;
  border: 1px solid var(--line, #d9e0df);
  display: grid;
  gap: 2px;
  margin: 0 -1px -1px 0;
  padding: 7px;
}

.metric-grid dt,
.coordinate-list dt {
  color: var(--muted, #67717b);
  font-size: 9px;
  text-transform: uppercase;
}

.metric-grid dd,
.coordinate-list dd {
  color: var(--ink, #20262d);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  margin: 0;
  overflow-wrap: anywhere;
}

.metric-grid dd {
  font-weight: 720;
}

.index-grid {
  display: grid;
  gap: 6px;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  margin-top: 9px;
}

.index-grid span {
  color: var(--muted, #67717b);
  display: grid;
  font-size: 9px;
  gap: 2px;
}

.index-grid strong {
  color: var(--ink, #20262d);
  font-size: 12px;
}

.pck-row {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 9px;
}

.pck-row span {
  align-items: center;
  background: #f0f1f1;
  border: 1px solid var(--line, #d9e0df);
  color: var(--muted, #67717b);
  display: inline-flex;
  font-size: 9px;
  gap: 3px;
  padding: 3px 5px;
}

.pck-row span.passed {
  background: #eaf5ec;
  border-color: #bfd8c4;
  color: var(--green, #3d804a);
}

details {
  border-top: 1px solid var(--line, #d9e0df);
  margin-top: 10px;
  padding-top: 8px;
}

summary {
  color: var(--teal-dark, #175c5d);
  cursor: pointer;
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
}

.coordinate-list {
  display: grid;
  gap: 7px;
  margin: 9px 0 0;
}

.coordinate-list > div {
  display: grid;
  gap: 2px;
}

.point-inspector > p {
  color: var(--muted, #67717b);
  font-size: 11px;
  line-height: 1.45;
  margin: -2px 0 9px;
}

.vertex-tools {
  display: grid;
  gap: 5px;
  grid-template-columns: 78px minmax(70px, 1fr) auto auto;
}

.inspection-card {
  background: #ffffff;
  border: 1px solid var(--line, #d9e0df);
  margin-top: 9px;
  padding: 9px;
}

.inspection-heading {
  align-items: center;
  display: flex;
  gap: 6px;
  margin-bottom: 9px;
}

.inspection-heading > span {
  background: #e2f2ef;
  color: #176b64;
  font-size: 9px;
  font-weight: 750;
  padding: 3px 5px;
  text-transform: uppercase;
}

.inspection-heading > span.target {
  background: #e8edf7;
  color: #385b9b;
}

.inspection-heading strong {
  font-size: 12px;
}

.inspection-heading small {
  color: var(--muted, #67717b);
  margin-left: auto;
  text-transform: capitalize;
}

.inspection-actions {
  display: flex;
  gap: 6px;
  justify-content: flex-end;
  margin-top: 9px;
}

.inspection-empty {
  align-items: center;
  border: 1px dashed #cdd6d4;
  color: var(--muted, #67717b);
  display: flex;
  font-size: 11px;
  gap: 7px;
  justify-content: center;
  margin-top: 9px;
  min-height: 54px;
}

.summary-pck {
  display: grid;
  gap: 4px;
  margin-top: 9px;
}

.summary-pck span {
  align-items: center;
  color: var(--muted, #67717b);
  display: flex;
  font-size: 10px;
  justify-content: space-between;
}

.summary-pck strong {
  color: var(--ink, #20262d);
}

.copy-status {
  align-items: center;
  background: #eaf5ec;
  bottom: 10px;
  box-shadow: 0 8px 24px rgba(24, 38, 42, 0.18);
  color: var(--green, #3d804a);
  display: flex;
  font-size: 11px;
  gap: 6px;
  padding: 7px 9px;
  position: sticky;
}

.spin {
  animation: spin 0.85s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 1080px) {
  .viewer-workspace {
    grid-template-columns: minmax(0, 1fr);
  }

  .viewer-stage {
    height: clamp(500px, 65vh, 700px);
  }

  .inspector {
    border-left: 0;
    border-top: 1px solid #354246;
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    height: auto;
    overflow: visible;
  }

  .inspector-section {
    border-right: 1px solid var(--line, #d9e0df);
  }

  .copy-status {
    grid-column: 1 / -1;
  }

  .correspondence-viewer.fullscreen .viewer-workspace {
    grid-template-columns: minmax(0, 1fr) 340px;
  }

  .correspondence-viewer.fullscreen .viewer-stage,
  .correspondence-viewer.fullscreen .inspector {
    height: 100%;
  }

  .correspondence-viewer.fullscreen .inspector {
    border-left: 1px solid #354246;
    border-top: 0;
    display: block;
    overflow-y: auto;
  }
}

@media (max-width: 700px) {
  .viewer-header,
  .dataset-bar {
    align-items: stretch;
    grid-template-columns: minmax(0, 1fr);
  }

  .viewer-header {
    display: grid;
  }

  .icon-button {
    justify-self: end;
  }

  .dataset-meta {
    justify-content: flex-start;
  }

  .viewer-stage {
    height: clamp(400px, 58vh, 560px);
  }

  .inspector {
    display: block;
  }

  .inspector-section {
    border-right: 0;
  }

  .match-list {
    display: none;
  }

  .match-select {
    display: block;
  }

  .vertex-tools {
    grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  }

  .canvas-help {
    max-width: calc(100% - 20px);
  }

  .correspondence-viewer.fullscreen {
    overflow-y: auto;
  }

  .correspondence-viewer.fullscreen .viewer-workspace {
    display: grid;
    flex: 0 0 auto;
    grid-template-columns: minmax(0, 1fr);
  }

  .correspondence-viewer.fullscreen .viewer-stage {
    height: min(62vh, 560px);
    min-height: 390px;
  }

  .correspondence-viewer.fullscreen .inspector {
    border-left: 0;
    border-top: 1px solid #354246;
    height: auto;
    overflow: visible;
  }
}
</style>
