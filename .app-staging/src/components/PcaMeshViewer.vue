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
  Camera,
  Expand,
  FolderOpen,
  Loader2,
  Minimize2,
  Palette,
  PanelLeftClose,
  PanelLeftOpen,
  Pause,
  Play,
  RotateCcw,
  Sun,
} from "@lucide/vue";
import * as THREE from "three";
import { OBJLoader } from "three/addons/loaders/OBJLoader.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";
import { HtmlStyleControls } from "../lib/HtmlStyleControls.js";
import MeshExplorer from "./MeshExplorer.vue";

const RECENT_STORAGE_KEY = "bugnist-mesh-viewer-recents-v1";
const NARROW_VIEW_QUERY = "(max-width: 760px)";
const SUPPORTED_EXTENSIONS = new Set([".obj", ".ply"]);
const AUTO_ROTATE_RADIANS_PER_SECOND = Math.PI / 30;

const presets = [
  {
    id: "brown",
    label: "Brown cricket - shared PCA",
    path: "visualizations\\bugnist_crickets_features_16v_512\\shared_pca_brown_black\\brownCricket_shared_pca_features.ply",
  },
  {
    id: "black",
    label: "Black cricket - shared PCA",
    path: "visualizations\\bugnist_crickets_features_16v_512\\shared_pca_brown_black\\blackCricket_shared_pca_features.ply",
  },
];

const viewerFrame = ref(null);
const stage = ref(null);
const fileInput = ref(null);
const selectedPreset = ref("brown");
const projectPath = ref(presets[0].path);
const displayName = ref("Brown cricket - shared PCA");
const loading = ref(false);
const loadError = ref("");
const draggingFile = ref(false);
const isSpinning = ref(false);
const spinSpeed = ref(0.65);
const renderMode = ref("exact");
const isFullscreen = ref(false);
const stats = ref({ vertices: 0, faces: 0, hasColors: false });
const currentFormat = ref("PLY");
const projectFiles = ref([]);
const explorerLoading = ref(false);
const explorerError = ref("");
const explorerOpen = ref(true);
const activePath = ref(normalizePath(presets[0].path));
const projectRecents = ref([]);
const sessionRecents = ref([]);

const recentMeshes = computed(() => {
  const seen = new Set();
  return [...sessionRecents.value, ...projectRecents.value]
    .sort((left, right) => right.openedAt - left.openedAt)
    .filter((item) => {
      if (seen.has(item.id)) return false;
      seen.add(item.id);
      return true;
    })
    .slice(0, 10);
});

let scene;
let camera3d;
let renderer;
let controls;
let displayedObject;
let animationFrame;
let renderedFrames = 0;
let resizeObserver;
let activeRequest;
let dragDepth = 0;
let modelRadius = 1;
let modelBounds;
let lastFrameTime;
let componentActive = true;
let debugApi;
const localFileCache = new Map();

onMounted(async () => {
  explorerOpen.value = !isNarrowViewport();
  restoreRecentMeshes();
  await nextTick();
  initializeViewer();
  installViewerDebugApi();
  document.addEventListener("fullscreenchange", syncFullscreenState);
  document.addEventListener("visibilitychange", onVisibilityChange);
  await Promise.all([refreshProjectFiles(), loadProjectMesh()]);
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
  cancelAnimationFrame(animationFrame);
  animationFrame = undefined;
  lastFrameTime = undefined;
});

onBeforeUnmount(() => {
  activeRequest?.abort();
  cancelAnimationFrame(animationFrame);
  animationFrame = undefined;
  resizeObserver?.disconnect();
  document.removeEventListener("fullscreenchange", syncFullscreenState);
  document.removeEventListener("visibilitychange", onVisibilityChange);
  uninstallViewerDebugApi();
  disposeDisplayedObject();
  controls?.dispose();
  renderer?.dispose();
  renderer?.forceContextLoss();
});

watch(isSpinning, () => {
  lastFrameTime = undefined;
  requestRender();
});

watch(spinSpeed, () => {
  if (isSpinning.value) requestRender();
});

watch(renderMode, () => updateMaterial());

function initializeViewer() {
  if (!stage.value) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x111719);

  camera3d = new THREE.PerspectiveCamera(38, 1, 0.01, 10000);

  renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: "high-performance",
    preserveDrawingBuffer: false,
  });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.domElement.setAttribute("aria-label", "Interactive PCA RGB mesh viewport");
  renderer.domElement.setAttribute("role", "img");
  renderer.domElement.setAttribute(
    "title",
    "Left-drag to rotate · Shift-left, middle, or right-drag to pan · Scroll to zoom · F to fit",
  );
  renderer.domElement.tabIndex = 0;
  stage.value.prepend(renderer.domElement);

  controls = new HtmlStyleControls(camera3d, renderer.domElement, { onChange: requestRender });

  const hemisphere = new THREE.HemisphereLight(0xf3f8f7, 0x273036, 2.2);
  const keyLight = new THREE.DirectionalLight(0xffffff, 2.8);
  const fillLight = new THREE.DirectionalLight(0xaecbe0, 1.4);
  keyLight.position.set(3, 4, 5);
  fillLight.position.set(-4, -1, 2);
  scene.add(hemisphere, keyLight, fillLight);

  resizeObserver = new ResizeObserver(resizeViewport);
  resizeObserver.observe(stage.value);
  resizeViewport();
  requestRender();
}

function resizeViewport() {
  if (!renderer || !camera3d || !stage.value) return;
  const width = Math.max(stage.value.clientWidth, 1);
  const height = Math.max(stage.value.clientHeight, 1);
  renderer.setSize(width, height, false);
  controls?.setViewport(width, height);
  requestRender();
}

function requestRender() {
  if (!componentActive || document.hidden || animationFrame || !renderer) return;
  animationFrame = requestAnimationFrame(renderFrame);
}

function renderFrame(timestamp) {
  animationFrame = undefined;
  if (!componentActive || document.hidden) return;
  if (isSpinning.value && controls) {
    const deltaSeconds = lastFrameTime === undefined
      ? 0
      : Math.min(Math.max((timestamp - lastFrameTime) / 1000, 0), 0.05);
    lastFrameTime = timestamp;
    controls.rotateByRadians(deltaSeconds * AUTO_ROTATE_RADIANS_PER_SECOND * Number(spinSpeed.value));
  } else {
    lastFrameTime = undefined;
  }
  renderScene();
  if (isSpinning.value) requestRender();
}

function renderScene() {
  if (!renderer || !scene || !camera3d || !stage.value?.clientWidth) return false;
  renderer.render(scene, camera3d);
  renderedFrames += 1;
  return true;
}

function onVisibilityChange() {
  if (document.hidden) {
    cancelAnimationFrame(animationFrame);
    animationFrame = undefined;
    return;
  }
  requestRender();
}

function installViewerDebugApi() {
  if (!import.meta.env.DEV) return;
  debugApi = Object.freeze({
    getRenderCount: () => renderedFrames,
    getControlsState: () => controls?.getState(),
    renderNow: () => renderScene(),
  });
  window.__BUGNIST_PCA_VIEWER_DEBUG__ = debugApi;
}

function uninstallViewerDebugApi() {
  if (!debugApi || window.__BUGNIST_PCA_VIEWER_DEBUG__ !== debugApi) return;
  delete window.__BUGNIST_PCA_VIEWER_DEBUG__;
  debugApi = undefined;
}

async function refreshProjectFiles() {
  explorerLoading.value = true;
  explorerError.value = "";
  try {
    const response = await fetch("/api/files?group=mesh");
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || "Could not index project meshes.");
    projectFiles.value = (data.files || [])
      .filter((file) => SUPPORTED_EXTENSIONS.has(fileExtension(file.relative)))
      .sort((left, right) => normalizePath(left.relative).localeCompare(normalizePath(right.relative)));
  } catch (error) {
    explorerError.value = error.message || "Could not index project meshes.";
  } finally {
    explorerLoading.value = false;
  }
}

async function loadProjectMesh(pathValue = projectPath.value, explicitLabel = "") {
  const requestedPath = typeof pathValue === "string" ? pathValue.trim() : projectPath.value.trim();
  if (!requestedPath) {
    loadError.value = "Enter an OBJ or PLY path.";
    return;
  }
  if (!SUPPORTED_EXTENSIONS.has(fileExtension(requestedPath))) {
    loadError.value = "The 3D viewer accepts OBJ and PLY files.";
    return;
  }

  activeRequest?.abort();
  const request = new AbortController();
  activeRequest = request;
  projectPath.value = requestedPath;
  loading.value = true;
  loadError.value = "";

  try {
    const response = await fetch(`/api/download?path=${encodeURIComponent(requestedPath)}`, {
      signal: request.signal,
    });
    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.error || `Could not load ${requestedPath}.`);
    }
    const buffer = await response.arrayBuffer();
    const preset = presets.find((item) => normalizePath(item.path).toLowerCase() === normalizePath(requestedPath).toLowerCase());
    showMeshBuffer(buffer, requestedPath, explicitLabel || preset?.label || fileLabel(requestedPath));
    activePath.value = normalizePath(requestedPath);
    selectedPreset.value = preset?.id || "custom";
    rememberProjectMesh(requestedPath);
  } catch (error) {
    if (error.name !== "AbortError") loadError.value = error.message;
  } finally {
    if (activeRequest === request) loading.value = false;
  }
}

function openProjectFile(file) {
  closeExplorerOnNarrow();
  loadProjectMesh(file.relative, file.name);
}

function selectPreset() {
  const preset = presets.find((item) => item.id === selectedPreset.value);
  if (preset) loadProjectMesh(preset.path, preset.label);
}

async function loadLocalFile(file) {
  if (!file) return;
  if (!SUPPORTED_EXTENSIONS.has(fileExtension(file.name))) {
    loadError.value = "The 3D viewer accepts OBJ and PLY files.";
    return;
  }

  activeRequest?.abort();
  loading.value = true;
  loadError.value = "";
  selectedPreset.value = "custom";

  try {
    const buffer = await file.arrayBuffer();
    showMeshBuffer(buffer, file.name, file.name);
    const recent = rememberLocalMesh(file);
    activePath.value = recent.activePath;
  } catch (error) {
    loadError.value = error.message || `Could not parse ${file.name}.`;
  } finally {
    loading.value = false;
  }
}

function showMeshBuffer(buffer, pathValue, label) {
  const extension = fileExtension(pathValue);
  let object;
  if (extension === ".ply") {
    const geometry = new PLYLoader().parse(buffer);
    object = geometry.index?.count
      ? new THREE.Mesh(geometry, new THREE.MeshBasicMaterial())
      : new THREE.Points(geometry, new THREE.PointsMaterial());
  } else if (extension === ".obj") {
    object = new OBJLoader().parse(new TextDecoder().decode(buffer));
  } else {
    throw new Error(`Unsupported mesh format: ${extension || "unknown"}.`);
  }
  showObject(object, label, extension.slice(1).toUpperCase());
}

function showObject(object, label, format) {
  const summary = summarizeObject(object);
  if (!summary.vertices) {
    disposeObjectResources(object);
    throw new Error(`The ${format} file does not contain any renderable vertices.`);
  }

  object.updateMatrixWorld(true);
  const bounds = new THREE.Box3().setFromObject(object);
  if (bounds.isEmpty()) {
    disposeObjectResources(object);
    throw new Error(`The ${format} file has invalid geometry bounds.`);
  }

  disposeDisplayedObject();
  const center = bounds.getCenter(new THREE.Vector3());
  object.position.sub(center);
  object.updateMatrixWorld(true);
  const centeredBounds = new THREE.Box3().setFromObject(object);
  const sphere = centeredBounds.getBoundingSphere(new THREE.Sphere());
  modelRadius = Math.max(sphere.radius || 1, 0.001);
  modelBounds = centeredBounds.clone();
  controls?.setBounds(modelBounds, 0.3, 12);

  displayedObject = object;
  scene.add(displayedObject);
  applyViewerMaterials();

  stats.value = summary;
  currentFormat.value = format;
  displayName.value = label;
  resetView();
}

function summarizeObject(object) {
  const summary = { vertices: 0, faces: 0, hasColors: false };
  object.traverse((child) => {
    if (!child.isMesh && !child.isPoints) return;
    const geometry = child.geometry;
    const positions = geometry?.getAttribute("position");
    if (!positions?.count) return;
    summary.vertices += positions.count;
    if (child.isMesh) {
      if (!geometry.getAttribute("normal")) geometry.computeVertexNormals();
      summary.faces += geometry.index ? Math.floor(geometry.index.count / 3) : Math.floor(positions.count / 3);
    }
    if (geometry.getAttribute("color")) summary.hasColors = true;
  });
  return summary;
}

function createMaterial(renderable) {
  const vertexColors = Boolean(renderable.geometry?.getAttribute("color"));
  if (renderable.isPoints) {
    return new THREE.PointsMaterial({
      color: vertexColors ? 0xffffff : 0xaac2c2,
      size: modelRadius * 0.009,
      sizeAttenuation: true,
      vertexColors,
    });
  }
  if (renderMode.value === "lit" || !vertexColors) {
    return new THREE.MeshStandardMaterial({
      color: vertexColors ? 0xffffff : 0xaac2c2,
      metalness: 0,
      roughness: 0.72,
      side: THREE.DoubleSide,
      vertexColors,
    });
  }
  return new THREE.MeshBasicMaterial({
    color: 0xffffff,
    side: THREE.DoubleSide,
    vertexColors: true,
  });
}

function applyViewerMaterials() {
  if (!displayedObject) return;
  const oldMaterials = new Set();
  displayedObject.traverse((child) => {
    if (!child.isMesh && !child.isPoints) return;
    for (const material of arrayValue(child.material)) oldMaterials.add(material);
    child.material = createMaterial(child);
  });
  for (const material of oldMaterials) material?.dispose();
}

function updateMaterial() {
  applyViewerMaterials();
  requestRender();
}

function disposeDisplayedObject() {
  if (!displayedObject) return;
  scene?.remove(displayedObject);
  disposeObjectResources(displayedObject);
  displayedObject = null;
  modelBounds = undefined;
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

function arrayValue(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function resetView() {
  if (!controls || !modelBounds) return;
  controls.fit(modelBounds);
  requestRender();
}

function toggleSpin() {
  isSpinning.value = !isSpinning.value;
  lastFrameTime = undefined;
  requestRender();
}

async function toggleFullscreen() {
  if (!viewerFrame.value) return;
  if (document.fullscreenElement === viewerFrame.value) {
    await document.exitFullscreen();
  } else {
    await viewerFrame.value.requestFullscreen();
  }
}

function syncFullscreenState() {
  isFullscreen.value = document.fullscreenElement === viewerFrame.value;
  requestAnimationFrame(resizeViewport);
}

function saveSnapshot() {
  if (!renderer || !scene || !camera3d) return;
  renderScene();
  const link = document.createElement("a");
  const safeName = displayName.value.replace(/[^a-z0-9_-]+/gi, "_").replace(/^_+|_+$/g, "");
  link.download = `${safeName || "pca_mesh"}_view.png`;
  link.href = renderer.domElement.toDataURL("image/png");
  link.click();
  requestRender();
}

function chooseLocalFile(event) {
  const file = event.target.files?.[0];
  loadLocalFile(file);
  event.target.value = "";
}

function onDragEnter() {
  dragDepth += 1;
  draggingFile.value = true;
}

function onDragLeave() {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) draggingFile.value = false;
}

function onDrop(event) {
  dragDepth = 0;
  draggingFile.value = false;
  loadLocalFile(event.dataTransfer?.files?.[0]);
}

function rememberProjectMesh(pathValue) {
  const normalized = normalizePath(pathValue);
  const item = {
    id: `project:${normalized.toLowerCase()}`,
    source: "project",
    path: normalized,
    activePath: normalized,
    name: baseName(normalized),
    openedAt: Date.now(),
  };
  projectRecents.value = [item, ...projectRecents.value.filter((recent) => recent.id !== item.id)].slice(0, 10);
  try {
    localStorage.setItem(RECENT_STORAGE_KEY, JSON.stringify(projectRecents.value));
  } catch {
    // The viewer still works when browser storage is unavailable.
  }
  return item;
}

function rememberLocalMesh(file) {
  const id = `local:${file.name}:${file.size}:${file.lastModified}`;
  localFileCache.set(id, file);
  const item = {
    id,
    source: "local",
    path: file.name,
    activePath: id,
    name: file.name,
    openedAt: Date.now(),
  };
  sessionRecents.value = [item, ...sessionRecents.value.filter((recent) => recent.id !== id)].slice(0, 10);
  return item;
}

function restoreRecentMeshes() {
  try {
    const stored = JSON.parse(localStorage.getItem(RECENT_STORAGE_KEY) || "[]");
    projectRecents.value = Array.isArray(stored)
      ? stored.filter((item) => item?.source === "project" && SUPPORTED_EXTENSIONS.has(fileExtension(item.path))).slice(0, 10)
      : [];
  } catch {
    projectRecents.value = [];
  }
}

function openRecentMesh(item) {
  closeExplorerOnNarrow();
  if (item.source === "project") {
    loadProjectMesh(item.path, fileLabel(item.path));
    return;
  }
  const file = localFileCache.get(item.id);
  if (file) loadLocalFile(file);
  else loadError.value = "Drop this local file again to reopen it.";
}

function clearRecentMeshes() {
  projectRecents.value = [];
  sessionRecents.value = [];
  localFileCache.clear();
  try {
    localStorage.removeItem(RECENT_STORAGE_KEY);
  } catch {
    // Browser storage can be unavailable in restricted contexts.
  }
}

function isNarrowViewport() {
  return window.matchMedia(NARROW_VIEW_QUERY).matches;
}

function closeExplorerOnNarrow() {
  if (isNarrowViewport()) explorerOpen.value = false;
}

function toggleExplorer() {
  explorerOpen.value = !explorerOpen.value;
  nextTick(resizeViewport);
}

function fileLabel(pathValue) {
  return baseName(pathValue).replace(/\.(obj|ply)$/i, "") || "3D mesh";
}

function baseName(pathValue) {
  return normalizePath(pathValue).split("/").pop() || "3D mesh";
}

function fileExtension(pathValue) {
  const match = baseName(pathValue).toLowerCase().match(/\.[^.]+$/);
  return match?.[0] || "";
}

function normalizePath(pathValue) {
  return String(pathValue || "").replaceAll("\\", "/");
}

function formatCount(value) {
  return new Intl.NumberFormat().format(value || 0);
}
</script>

<template>
  <article ref="viewerFrame" class="pca-viewer" :class="{ fullscreen: isFullscreen }">
    <div class="viewer-header">
      <div class="viewer-heading">
        <button
          class="explorer-toggle"
          type="button"
          aria-controls="mesh-explorer"
          :aria-expanded="explorerOpen"
          :title="explorerOpen ? 'Hide mesh explorer' : 'Show mesh explorer'"
          @click="toggleExplorer"
        >
          <PanelLeftClose v-if="explorerOpen" :size="18" />
          <PanelLeftOpen v-else :size="18" />
        </button>
        <div>
          <span class="viewer-kicker"><Palette :size="15" />{{ currentFormat }} mesh viewer</span>
          <h2>{{ displayName }}</h2>
        </div>
      </div>
      <div class="viewer-stats" aria-live="polite">
        <span>{{ formatCount(stats.vertices) }} vertices</span>
        <span>{{ formatCount(stats.faces) }} faces</span>
        <span :class="{ warning: stats.vertices && !stats.hasColors }">
          {{ stats.hasColors ? "RGB present" : "No RGB" }}
        </span>
      </div>
    </div>

    <div
      class="viewer-workspace"
      :class="{ 'explorer-closed': !explorerOpen }"
    >
      <MeshExplorer
        id="mesh-explorer"
        v-show="explorerOpen"
        :files="projectFiles"
        :active-path="activePath"
        :loading="explorerLoading"
        :error="explorerError"
        :recents="recentMeshes"
        @open="openProjectFile"
        @refresh="refreshProjectFiles"
        @open-recent="openRecentMesh"
        @clear-recent="clearRecentMeshes"
      />

      <div class="viewer-main">
        <div class="viewer-toolbar" aria-label="Viewer controls">
          <div class="viewer-modes" aria-label="Color rendering mode">
            <button
              type="button"
              :class="{ active: renderMode === 'exact' }"
              :aria-pressed="renderMode === 'exact'"
              title="Exact unlit vertex colors"
              @click="renderMode = 'exact'"
            >
              <Palette :size="15" />PCA RGB
            </button>
            <button
              type="button"
              :class="{ active: renderMode === 'lit' }"
              :aria-pressed="renderMode === 'lit'"
              title="Vertex colors with surface lighting"
              @click="renderMode = 'lit'"
            >
              <Sun :size="15" />Lit
            </button>
          </div>

          <select
            v-model="selectedPreset"
            class="viewer-preset"
            aria-label="Mesh preset"
            @change="selectPreset"
          >
            <option v-for="preset in presets" :key="preset.id" :value="preset.id">{{ preset.label }}</option>
            <option value="custom">Custom / Explorer</option>
          </select>

          <label class="viewer-speed">
            <span>Spin {{ Number(spinSpeed).toFixed(2) }}x</span>
            <input v-model="spinSpeed" type="range" min="0.15" max="2" step="0.05" />
          </label>

          <div class="viewer-actions">
            <button type="button" :title="isSpinning ? 'Pause rotation' : 'Resume rotation'" @click="toggleSpin">
              <Pause v-if="isSpinning" :size="18" />
              <Play v-else :size="18" />
            </button>
            <button type="button" title="Fit mesh in view (F)" @click="resetView"><RotateCcw :size="18" /></button>
            <button type="button" title="Open a local OBJ or PLY" @click="fileInput?.click()"><FolderOpen :size="18" /></button>
            <button type="button" title="Save current view as PNG" @click="saveSnapshot"><Camera :size="18" /></button>
            <button type="button" :title="isFullscreen ? 'Exit fullscreen' : 'Enter fullscreen'" @click="toggleFullscreen">
              <Minimize2 v-if="isFullscreen" :size="18" />
              <Expand v-else :size="18" />
            </button>
          </div>
          <input ref="fileInput" class="visually-hidden" type="file" accept=".obj,.ply" @change="chooseLocalFile" />
        </div>

        <div
          ref="stage"
          class="viewer-stage"
          title="Left-drag to rotate · Shift-left, middle, or right-drag to pan · Scroll to zoom · F to fit"
          @dragenter.prevent="onDragEnter"
          @dragleave.prevent="onDragLeave"
          @dragover.prevent
          @drop.prevent="onDrop"
          @keydown.space.prevent="toggleSpin"
        >
          <div v-if="loading" class="viewer-state"><Loader2 class="spin" :size="26" />Loading mesh</div>
          <div v-else-if="loadError" class="viewer-state error">{{ loadError }}</div>
          <div v-if="draggingFile" class="viewer-drop">Drop OBJ or PLY</div>
        </div>

      </div>
    </div>
  </article>
</template>

<style scoped>
.pca-viewer {
  background: #ffffff;
  border: 1px solid var(--line);
  box-shadow: var(--shadow);
  min-width: 0;
}

.viewer-header {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 18px;
  justify-content: space-between;
  min-height: 72px;
  padding: 12px 16px;
}

.viewer-heading {
  align-items: center;
  display: flex;
  gap: 10px;
  min-width: 0;
}

.viewer-heading > div {
  min-width: 0;
}

.explorer-toggle {
  align-items: center;
  background: #f7f9f8;
  border: 1px solid var(--line);
  color: var(--ink);
  display: inline-flex;
  flex: 0 0 36px;
  height: 36px;
  justify-content: center;
  padding: 0;
  width: 36px;
}

.explorer-toggle:hover {
  background: var(--panel-soft);
  color: var(--teal-dark);
}

.viewer-header h2 {
  font-size: 17px;
  font-weight: 720;
  margin: 4px 0 0;
  overflow-wrap: anywhere;
}

.viewer-kicker {
  align-items: center;
  color: var(--teal);
  display: flex;
  font-size: 11px;
  font-weight: 700;
  gap: 6px;
  text-transform: uppercase;
}

.viewer-stats {
  align-items: center;
  color: var(--muted);
  display: flex;
  flex-wrap: wrap;
  font-size: 11px;
  gap: 6px;
  justify-content: flex-end;
}

.viewer-stats span {
  border: 1px solid var(--line);
  padding: 4px 7px;
}

.viewer-stats .warning {
  border-color: #d5ab80;
  color: var(--amber);
}

.viewer-workspace {
  display: grid;
  grid-template-columns: clamp(236px, 22vw, 286px) minmax(0, 1fr);
  height: calc(clamp(460px, 68vh, 720px) + 50px);
  min-width: 0;
}

.viewer-workspace.explorer-closed {
  grid-template-columns: minmax(0, 1fr);
}

.viewer-main {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  height: 100%;
  min-width: 0;
}

.viewer-toolbar {
  align-items: center;
  background: #182124;
  border-bottom: 1px solid #354246;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  min-height: 50px;
  padding: 6px 10px;
}

.viewer-stage {
  background: #111719;
  height: auto;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
  position: relative;
  touch-action: none;
  width: 100%;
}

.viewer-stage :deep(canvas:focus-visible) {
  outline: 3px solid rgba(31, 118, 118, 0.8);
  outline-offset: -3px;
}

.viewer-stage :deep(canvas) {
  display: block;
  height: 100%;
  width: 100%;
}

.viewer-actions {
  display: flex;
  gap: 6px;
  margin-left: auto;
}

.viewer-actions button,
.viewer-preset {
  align-items: center;
  background: #263235;
  border: 1px solid #3c4a4d;
  color: #e7efee;
}

.viewer-actions button {
  display: inline-flex;
  height: 38px;
  justify-content: center;
  padding: 0;
  width: 38px;
}

.viewer-actions button:hover,
.viewer-preset:hover {
  background: #314043;
  border-color: #526164;
}

.viewer-preset {
  flex: 0 1 220px;
  height: 38px;
  min-width: 160px;
  padding: 0 30px 0 10px;
  width: 220px;
}

.viewer-modes {
  background: #243033;
  border: 1px solid #3c4a4d;
  display: grid;
  grid-template-columns: auto auto;
}

.viewer-modes button {
  align-items: center;
  background: transparent;
  border: 0;
  color: #bfcaca;
  display: inline-flex;
  gap: 6px;
  min-height: 36px;
  padding: 7px 10px;
}

.viewer-modes button.active {
  background: #edf4f2;
  color: #1d292c;
}

.viewer-speed {
  align-items: center;
  color: #bdc9c7;
  display: grid;
  font-size: 11px;
  gap: 8px;
  grid-template-columns: auto minmax(90px, 130px);
  white-space: nowrap;
}

.viewer-speed input {
  accent-color: #58b4a9;
  background: transparent;
  border: 0;
  box-shadow: none;
  height: 14px;
  min-height: 14px;
  padding: 0;
}

.viewer-state,
.viewer-drop {
  align-items: center;
  background: rgba(17, 23, 25, 0.78);
  color: #f4f8f7;
  display: flex;
  gap: 9px;
  inset: 0;
  justify-content: center;
  position: absolute;
  z-index: 2;
}

.viewer-state.error {
  color: #ffd2cd;
  padding: 24px;
  text-align: center;
}

.viewer-drop {
  border: 2px dashed #75c2b8;
  inset: 12px;
  z-index: 4;
}

.visually-hidden {
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
  clip: rect(0, 0, 0, 0);
}

.pca-viewer.fullscreen {
  background: #111719;
  border: 0;
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
  height: 100vh;
  width: 100vw;
}

.fullscreen .viewer-header {
  background: #ffffff;
}

.fullscreen .viewer-stage {
  height: 100%;
}

.fullscreen .viewer-workspace,
.fullscreen .viewer-main {
  height: auto;
  min-height: 0;
}

.fullscreen .viewer-main {
  display: grid;
  grid-template-rows: auto minmax(0, 1fr);
}

@media (max-width: 760px) {
  .viewer-workspace {
    grid-template-columns: minmax(0, 1fr);
    height: auto;
  }

  .viewer-workspace :deep(.mesh-explorer) {
    border-bottom: 1px solid #354246;
    border-right: 0;
    height: min(36vh, 260px);
  }

  .fullscreen .viewer-workspace :deep(.mesh-explorer) {
    height: min(30vh, 240px);
  }
}

@media (max-width: 760px) {
  .viewer-header {
    align-items: flex-start;
    display: grid;
  }

  .viewer-stats {
    justify-content: flex-start;
  }

  .viewer-stage {
    height: clamp(390px, 62vh, 560px);
    min-height: 390px;
  }

  .viewer-main {
    height: auto;
  }

  .viewer-toolbar {
    align-items: stretch;
  }

  .viewer-speed {
    flex: 1 1 180px;
  }

  .viewer-preset {
    flex: 1 1 190px;
    min-width: 0;
  }
}
</style>
