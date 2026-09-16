import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright";

const baseUrl = process.env.PCA_VIEWER_URL || "http://127.0.0.1:5173";
const screenshotDir = process.env.PCA_VIEWER_SCREENSHOTS || path.join(os.tmpdir(), "bugnist-pca-viewer-smoke");
const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

fs.mkdirSync(screenshotDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: edgePath,
  headless: true,
  args: [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
  ],
});

try {
  await checkViewer({ width: 1440, height: 1000 }, "desktop", true);
  await checkViewer({ width: 390, height: 844 }, "mobile", false);
  console.log("PCA viewer smoke test passed at desktop and mobile sizes.");
} finally {
  await browser.close();
}

async function checkViewer(viewport, name, testInteraction) {
  const page = await browser.newPage({ viewport });
  const pageErrors = [];
  const failedResponses = [];
  page.on("pageerror", (error) => pageErrors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().startsWith("Failed to load resource:")) {
      pageErrors.push(message.text());
    }
  });
  page.on("response", (response) => {
    if (response.status() >= 400 && !response.url().endsWith("/favicon.ico")) {
      failedResponses.push(`${response.status()} ${response.url()}`);
    }
  });

  await page.goto(baseUrl, { waitUntil: "networkidle" });
  await page.getByRole("button", { name: "Visualize" }).click();
  assert.equal(new URL(page.url()).hash, "#/visualize/mesh", `${name}: mesh route was not saved in the URL`);
  const canvas = page.locator(".viewer-stage canvas");
  try {
    await canvas.waitFor({ state: "visible" });
  } catch (error) {
    throw new Error(`${name}: canvas did not appear. Browser errors: ${pageErrors.join(" | ") || "none"}`, {
      cause: error,
    });
  }
  await page.waitForFunction(() =>
    document.querySelector(".viewer-stats")?.textContent.replace(/\D/g, "").includes("24683"),
  );
  await page.waitForTimeout(700);

  const explorer = page.locator(".mesh-explorer");
  assert.equal(await page.locator(".viewer-source").count(), 0, `${name}: deprecated viewer footer is still present`);
  if (viewport.width <= 720 && await explorer.isHidden()) {
    await page.locator(".explorer-toggle").click();
  }
  await explorer.waitFor({ state: "visible" });
  assert.ok((await explorer.locator(".tree-row.file-row").count()) > 0, `${name}: explorer has no visible mesh files`);
  assert.ok((await explorer.locator(".recent-item").count()) > 0, `${name}: initial mesh was not added to Recent`);

  const pixels = await canvasPixels(page);
  assert.ok(pixels.nonBackgroundRatio > 0.02, `${name}: rendered mesh occupies too little of the canvas`);
  assert.ok(pixels.colorfulRatio > 0.01, `${name}: canvas does not contain enough PCA-colored pixels`);

  const layout = await page.evaluate(() => {
    const stage = document.querySelector(".viewer-stage").getBoundingClientRect();
    return {
      bodyOverflow: document.documentElement.scrollWidth - window.innerWidth,
      stageLeft: stage.left,
      stageRight: stage.right,
      viewportWidth: window.innerWidth,
    };
  });
  assert.ok(layout.bodyOverflow <= 1, `${name}: page overflows horizontally by ${layout.bodyOverflow}px`);
  assert.ok(layout.stageLeft >= -1 && layout.stageRight <= layout.viewportWidth + 1, `${name}: viewport is clipped`);

  if (testInteraction) {
    await page.getByTitle("Resume rotation").click();
    await page.waitForTimeout(100);
    const spinningHash = (await canvasPixels(page)).hash;
    await page.waitForTimeout(850);
    const rotatedHash = (await canvasPixels(page)).hash;
    assert.notEqual(rotatedHash, spinningHash, "auto-rotation did not change the rendered view");

    await page.getByTitle("Pause rotation").click();
    await page.waitForTimeout(250);
    const pausedHash = (await canvasPixels(page)).hash;
    await page.waitForTimeout(450);
    assert.equal((await canvasPixels(page)).hash, pausedHash, "paused viewer continued rotating");

    const bounds = await canvas.boundingBox();
    assert.ok(bounds, "viewer canvas has no bounds");
    const startX = bounds.x + bounds.width * 0.5;
    const startY = bounds.y + bounds.height * 0.5;

    const untouchedControls = await controlsState(page);
    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + 3, startY);
    await page.mouse.up();
    assert.deepEqual(await controlsState(page), untouchedControls, "sub-4px pointer jitter moved the camera");

    await page.mouse.move(startX, startY);
    await page.mouse.down();
    await page.mouse.move(startX + 95, startY + 25, { steps: 8 });
    await page.mouse.up();
    await page.waitForTimeout(450);
    assert.notEqual((await canvasPixels(page)).hash, pausedHash, "pointer orbit did not change the rendered view");
    const rotatedControls = await controlsState(page);
    assert.notEqual(rotatedControls.yaw, untouchedControls.yaw, "left-drag did not rotate yaw");
    assert.notEqual(rotatedControls.pitch, untouchedControls.pitch, "left-drag did not rotate pitch");
    assert.deepEqual(rotatedControls.pan, untouchedControls.pan, "left-drag unexpectedly panned");
    assert.equal(rotatedControls.distance, untouchedControls.distance, "left-drag unexpectedly zoomed");

    await page.mouse.move(startX, startY);
    await page.keyboard.down("Shift");
    await page.mouse.down();
    await page.mouse.move(startX + 70, startY - 30, { steps: 6 });
    await page.mouse.up();
    await page.keyboard.up("Shift");
    const shiftPannedControls = await controlsState(page);
    assert.equal(shiftPannedControls.yaw, rotatedControls.yaw, "Shift-left drag rotated instead of panning");
    assert.equal(shiftPannedControls.pitch, rotatedControls.pitch, "Shift-left drag rotated instead of panning");
    assert.notDeepEqual(shiftPannedControls.pan, rotatedControls.pan, "Shift-left drag did not pan");

    await page.mouse.move(startX, startY);
    await page.mouse.down({ button: "right" });
    await page.mouse.move(startX - 45, startY + 20, { steps: 5 });
    await page.mouse.up({ button: "right" });
    const rightPannedControls = await controlsState(page);
    assert.equal(rightPannedControls.yaw, shiftPannedControls.yaw, "right-drag rotated instead of panning");
    assert.notDeepEqual(rightPannedControls.pan, shiftPannedControls.pan, "right-drag did not pan");

    await page.mouse.move(startX, startY);
    const beforeWheelControls = await controlsState(page);
    await page.mouse.wheel(0, -320);
    const zoomedControls = await controlsState(page);
    assert.ok(zoomedControls.distance < beforeWheelControls.distance, "wheel-up did not zoom toward the mesh");

    await page.keyboard.press("f");
    const fittedControls = await controlsState(page);
    assert.deepEqual(fittedControls.pan, [0, 0], "F did not clear the camera pan while fitting");

    await explorer.locator(".explorer-search input").fill("BrownCricket_rotated.obj");
    const objRow = explorer.locator(".tree-row.file-row", { hasText: "BrownCricket_rotated.obj" }).first();
    await objRow.waitFor({ state: "visible" });
    await objRow.click();
    await page.waitForFunction(() =>
      document.querySelector(".viewer-kicker")?.textContent.includes("OBJ")
      && document.querySelector(".viewer-header h2")?.textContent.includes("BrownCricket_rotated.obj"),
    );
    await page.waitForTimeout(300);
    assert.ok((await canvasPixels(page)).nonBackgroundRatio > 0.02, "OBJ selected from explorer did not render");
    assert.ok(
      await explorer.locator(".recent-item", { hasText: "BrownCricket_rotated.obj" }).isVisible(),
      "OBJ selected from explorer was not added to Recent",
    );
    await explorer.locator(".explorer-search input").fill("");

    await page.locator(".viewer-preset").selectOption("black");
    await page.waitForFunction(() =>
      document.querySelector(".viewer-stats")?.textContent.replace(/\D/g, "").includes("24761"),
    );
    await page.waitForTimeout(300);
    const secondMeshPixels = await canvasPixels(page);
    assert.ok(secondMeshPixels.colorfulRatio > 0.01, "black cricket PCA mesh did not render colored pixels");

    await page.getByTitle("Enter fullscreen").click();
    await page.waitForFunction(() => document.fullscreenElement?.classList.contains("pca-viewer"));
    await page.getByTitle("Exit fullscreen").waitFor({ state: "visible" });
    await page.getByTitle("Exit fullscreen").click();
    await page.waitForFunction(() => !document.fullscreenElement);
  }

  await page.locator(".pca-viewer").screenshot({ path: path.join(screenshotDir, `pca-viewer-${name}.png`) });

  assert.deepEqual(pageErrors, [], `${name}: browser errors: ${pageErrors.join(" | ")}`);
  assert.deepEqual(failedResponses, [], `${name}: failed requests: ${failedResponses.join(" | ")}`);
  await page.close();
}

async function controlsState(page) {
  return page.evaluate(() => window.__BUGNIST_PCA_VIEWER_DEBUG__?.getControlsState?.());
}

async function canvasPixels(page) {
  return page.locator(".viewer-stage canvas").evaluate((canvas) => {
    window.__BUGNIST_PCA_VIEWER_DEBUG__?.renderNow?.();
    const gl = canvas.getContext("webgl2") || canvas.getContext("webgl");
    if (!gl) throw new Error("WebGL context is unavailable");

    const width = gl.drawingBufferWidth;
    const height = gl.drawingBufferHeight;
    const pixels = new Uint8Array(width * height * 4);
    gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

    const background = [pixels[0], pixels[1], pixels[2]];
    const stride = Math.max(1, Math.floor((width * height) / 160000));
    let sampled = 0;
    let nonBackground = 0;
    let colorful = 0;
    let hash = 2166136261;

    for (let pixel = 0; pixel < width * height; pixel += stride) {
      const i = pixel * 4;
      const r = pixels[i];
      const g = pixels[i + 1];
      const b = pixels[i + 2];
      const delta = Math.abs(r - background[0]) + Math.abs(g - background[1]) + Math.abs(b - background[2]);
      if (delta > 18) nonBackground += 1;
      if (Math.max(r, g, b) - Math.min(r, g, b) > 25 && delta > 18) colorful += 1;
      hash ^= r + g * 257 + b * 65537 + pixel;
      hash = Math.imul(hash, 16777619);
      sampled += 1;
    }

    return {
      colorfulRatio: colorful / sampled,
      hash: hash >>> 0,
      nonBackgroundRatio: nonBackground / sampled,
    };
  });
}
