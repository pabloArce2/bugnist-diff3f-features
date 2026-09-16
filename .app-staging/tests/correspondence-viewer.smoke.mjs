import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { chromium } from "playwright";

const baseUrl = process.env.CORRESPONDENCE_VIEWER_URL || "http://127.0.0.1:5173";
const screenshotDir = process.env.PCA_VIEWER_SCREENSHOTS || path.join(os.tmpdir(), "bugnist-pca-viewer-smoke");
const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
fs.mkdirSync(screenshotDir, { recursive: true });

const browser = await chromium.launch({
  executablePath: edgePath,
  headless: true,
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--enable-webgl"],
});

try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
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
  await page.getByRole("tab", { name: /Correspondences/ }).click();
  const viewer = page.locator(".correspondence-viewer");
  await viewer.waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector(".dataset-meta")?.textContent.includes("9 landmarks"));
  await page.locator(".stage-state").waitFor({ state: "hidden" });
  assert.equal(new URL(page.url()).hash, "#/visualize/correspondences", "correspondence route was not saved");

  await page.reload({ waitUntil: "networkidle" });
  await viewer.waitFor({ state: "visible" });
  await page.waitForFunction(() => document.querySelector(".dataset-meta")?.textContent.includes("9 landmarks"));
  await page.locator(".stage-state").waitFor({ state: "hidden" });
  assert.equal(new URL(page.url()).hash, "#/visualize/correspondences", "reload did not preserve the correspondence route");

  assert.equal(await viewer.locator(".match-list button").count(), 9, "expected all nine landmark rows");
  assert.equal(await viewer.locator(".viewer-error").count(), 0, "viewer reported a load or validation error");
  assert.equal(await viewer.locator(".dataset-bar option").count(), 2, "expected both benchmark directions");

  const stage = viewer.locator(".viewer-stage");
  const canvas = viewer.getByLabel("Interactive point correspondence viewport");
  const canvasBox = await canvas.boundingBox();
  assert.ok(canvasBox, "correspondence canvas has no layout box");
  const pickX = canvasBox.x + canvasBox.width * 0.28;
  const pickY = canvasBox.y + canvasBox.height * 0.53;
  const initialControls = await controlsState(page);
  await page.mouse.move(pickX, pickY);
  await page.mouse.down();
  await page.mouse.move(pickX + 3, pickY);
  await page.mouse.up();
  await viewer.locator(".inspection-card").waitFor({ state: "visible" });
  assert.match(await viewer.locator(".inspection-heading strong").textContent(), /^#\d+$/);
  assert.deepEqual(await controlsState(page), initialControls, "sub-4px inspection jitter moved the camera");

  await page.keyboard.press("Escape");
  await viewer.locator(".inspection-card").waitFor({ state: "hidden" });

  const dragX = canvasBox.x + canvasBox.width * 0.55;
  const dragY = canvasBox.y + canvasBox.height * 0.5;
  await page.mouse.move(dragX, dragY);
  await page.mouse.down();
  await page.mouse.move(dragX + 80, dragY + 25, { steps: 7 });
  await page.mouse.up();
  const rotatedControls = await controlsState(page);
  assert.notEqual(rotatedControls.yaw, initialControls.yaw, "correspondence left-drag did not rotate");
  assert.deepEqual(rotatedControls.pan, initialControls.pan, "correspondence left-drag unexpectedly panned");
  assert.equal(await viewer.locator(".inspection-card").count(), 0, "camera drag was mistaken for a surface click");

  await page.mouse.move(dragX, dragY);
  await page.keyboard.down("Shift");
  await page.mouse.down();
  await page.mouse.move(dragX + 60, dragY - 30, { steps: 6 });
  await page.mouse.up();
  await page.keyboard.up("Shift");
  const pannedControls = await controlsState(page);
  assert.equal(pannedControls.yaw, rotatedControls.yaw, "Shift-left drag rotated instead of panning");
  assert.notDeepEqual(pannedControls.pan, rotatedControls.pan, "Shift-left drag did not pan");

  const beforeWheel = await controlsState(page);
  await page.mouse.wheel(0, -280);
  assert.ok((await controlsState(page)).distance < beforeWheel.distance, "correspondence wheel-up did not zoom in");

  await viewer.getByRole("button", { name: "Front" }).click();
  assert.equal((await controlsState(page)).yaw, 0, "Front view did not set the HTML viewer angle");
  await viewer.getByRole("button", { name: "Side" }).click();
  assert.equal((await controlsState(page)).yaw, Math.PI / 2, "Side view did not set the HTML viewer angle");

  await viewer.getByTitle("Inspect a random vertex").click();
  await viewer.locator(".inspection-card").waitFor({ state: "visible" });
  assert.match(await viewer.locator(".inspection-heading strong").textContent(), /^#\d+$/);

  const layout = await page.evaluate(() => {
    const stage = document.querySelector(".correspondence-viewer .viewer-stage").getBoundingClientRect();
    const inspector = document.querySelector(".correspondence-viewer .inspector").getBoundingClientRect();
    return {
      bodyOverflow: document.documentElement.scrollWidth - window.innerWidth,
      disjoint: stage.right <= inspector.left + 1 || stage.bottom <= inspector.top + 1,
      canvasWidth: document.querySelector(".correspondence-viewer canvas").width,
      canvasHeight: document.querySelector(".correspondence-viewer canvas").height,
    };
  });
  assert.ok(layout.bodyOverflow <= 1, `page overflows horizontally by ${layout.bodyOverflow}px`);
  assert.equal(layout.disjoint, true, "inspector overlaps the WebGL stage");
  assert.ok(layout.canvasWidth > 400 && layout.canvasHeight > 400, "WebGL canvas is unexpectedly small");

  await viewer.getByTitle("Enter fullscreen").click();
  await page.waitForFunction(() => document.fullscreenElement?.classList.contains("correspondence-viewer"));
  assert.equal(await viewer.getByTitle("Exit fullscreen").isVisible(), true, "fullscreen controls did not update");
  await viewer.getByTitle("Exit fullscreen").click();
  await page.waitForFunction(() => !document.fullscreenElement);

  await viewer.locator(".dataset-bar select").selectOption({ label: "Brown cricket → Black cricket" });
  await page.waitForFunction(() => document.querySelector(".correspondence-viewer h2")?.textContent.includes("Brown cricket"));
  await page.locator(".stage-state").waitFor({ state: "hidden" });
  assert.equal(await viewer.locator(".viewer-error").count(), 0, "reverse direction failed to load");

  await viewer.screenshot({ path: path.join(screenshotDir, "correspondence-viewer-desktop.png") });
  assert.deepEqual(pageErrors, [], `browser errors: ${pageErrors.join(" | ")}`);
  assert.deepEqual(failedResponses, [], `failed responses: ${failedResponses.join(" | ")}`);
  console.log(`Correspondence viewer smoke test passed. Screenshot: ${screenshotDir}`);
  await page.close();
} finally {
  await browser.close();
}

async function controlsState(page) {
  return page.evaluate(() => window.__BUGNIST_CORRESPONDENCE_VIEWER_DEBUG__?.getControlsState?.());
}
