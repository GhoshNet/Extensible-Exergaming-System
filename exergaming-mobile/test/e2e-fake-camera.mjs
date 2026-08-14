// End-to-end smoke test: launches real Chromium with a real video file fed
// in as a fake camera device, loads the actual app over a local static
// server, selects an exercise, clicks Start, and reads the on-screen rep
// counter after the clip finishes. Exercises the full real pipeline
// (camera capture -> MediaPipe WASM pose inference -> landmark conversion
// -> GenericExercise state machine -> DOM) exactly as it runs on a phone --
// something otherwise untestable in this sandbox, which has no camera.
//
// Usage: node e2e-fake-camera.mjs <y4m-path> <exercise-index> <expected-reps> [config-query] [wait-ms]
//   exercise-index matches EXERCISES order in js/app.js: 0=squat, 1=push-up, 2=bicep curl
import { chromium } from 'playwright';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { join, extname, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const PORT = 8743;

const MIME = {
  '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
  '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm',
  '.task': 'application/octet-stream', '.png': 'image/png',
};

function startServer() {
  const server = createServer(async (req, res) => {
    try {
      let path = decodeURIComponent(req.url.split('?')[0]);
      if (path === '/') path = '/index.html';
      const filePath = join(ROOT, path);
      const data = await readFile(filePath);
      res.writeHead(200, { 'Content-Type': MIME[extname(filePath)] || 'application/octet-stream' });
      res.end(data);
    } catch (err) {
      res.writeHead(404);
      res.end('not found');
    }
  });
  return new Promise((resolve) => server.listen(PORT, () => resolve(server)));
}

async function main() {
  const [y4mArg, exIdxArg, expectedArg, configQuery = '', waitMsArg] = process.argv.slice(2);
  const y4mPath = join(__dirname, y4mArg);
  const exerciseIndex = Number(exIdxArg ?? 0);
  const expectedReps = expectedArg;
  const waitMs = Number(waitMsArg ?? 20000);

  const server = await startServer();
  console.log(`Serving ${ROOT} on http://localhost:${PORT}`);
  console.log(`Video: ${y4mArg} | exerciseIndex: ${exerciseIndex} | expected: ${expectedReps} | config: ${configQuery || '(default)'}`);

  const browser = await chromium.launch({
    args: [
      '--use-fake-device-for-media-stream',
      '--use-fake-ui-for-media-stream',
      `--use-file-for-fake-video-capture=${y4mPath}`,
    ],
  });
  const page = await browser.newPage();
  page.on('console', (msg) => console.log(`[browser:${msg.type()}]`, msg.text()));
  page.on('pageerror', (err) => console.error('[browser:pageerror]', err));

  await page.goto(`http://localhost:${PORT}/index.html${configQuery}`);

  console.log('Waiting for pose model to load...');
  await page.waitForFunction(
    () => document.getElementById('loadingOverlay').style.display === 'none',
    { timeout: 60000 }
  );
  console.log('Model loaded, camera should be live.');

  if (exerciseIndex !== 0) {
    await page.selectOption('#exerciseSelect', { index: exerciseIndex });
    await page.waitForTimeout(300); // let the definition fetch/load resolve
  }

  await page.click('#startBtn');
  console.log(`Clicked Start. Letting the clip play through (~${waitMs / 1000}s)...`);
  await page.waitForTimeout(waitMs);

  const repCount = await page.textContent('#repCount');
  const state = await page.textContent('#stateBadge');
  const fps = await page.textContent('#fpsBadge');
  console.log(`\nFINAL: reps=${repCount} state=${state} fps=${fps}`);
  console.log(
    repCount === expectedReps
      ? `MATCHES expected ground truth (${expectedReps})`
      : `DOES NOT MATCH expected ground truth (${expectedReps}), got ${repCount}`
  );

  await browser.close();
  server.close();
}

main().catch((err) => { console.error(err); process.exit(1); });
