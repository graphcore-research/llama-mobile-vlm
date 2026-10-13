import { spawn } from 'node:child_process';
import { promises as fs } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const today = new Date();
const date = [today.getFullYear(), String(today.getMonth() + 1).padStart(2, '0'), String(today.getDate()).padStart(2, '0')].join('-');
const output = path.join(root, 'out', `${date}-llama-mobile.pdf`);
const profile = await fs.mkdtemp(path.join(tmpdir(), 'llama-slides-chrome-'));
const server = spawn('python3', ['tools/serve.py'], { cwd: root, stdio: ['ignore', 'pipe', 'inherit'] });
let chrome;
let socket;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

try {
  const url = await Promise.race([
    new Promise((resolve, reject) => {
      let output = '';
      server.stdout.on('data', chunk => {
        output += chunk;
        const match = output.match(/Slides: (http:\/\/127\.0\.0\.1:\d+)/);
        if (match) resolve(match[1]);
      });
      server.on('error', reject);
      server.on('exit', code => reject(new Error(`Slide server exited with code ${code}`)));
    }),
    sleep(10000).then(() => { throw new Error('Slide server did not start'); })
  ]);

  chrome = spawn('google-chrome', [
    '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
    `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank'
  ], { stdio: 'ignore' });
  chrome.on('error', error => { throw error; });

  let port;
  for (let i = 0; i < 100; i++) {
    try {
      port = Number((await fs.readFile(path.join(profile, 'DevToolsActivePort'), 'utf8')).split('\n')[0]);
      break;
    } catch { await sleep(100); }
  }
  if (!port) throw new Error('Chrome did not start');

  const tabs = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
  const tab = tabs.find(item => item.type === 'page');
  socket = new WebSocket(tab.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.addEventListener('open', resolve, { once: true });
    socket.addEventListener('error', reject, { once: true });
  });

  let nextId = 0;
  const pending = new Map();
  socket.addEventListener('message', event => {
    const message = JSON.parse(event.data);
    if (!message.id) return;
    const request = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) request.reject(new Error(message.error.message));
    else request.resolve(message.result);
  });
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });

  await call('Page.enable');
  await call('Page.navigate', { url: `${url}/?print-pdf` });
  let ready = false;
  for (let i = 0; i < 200; i++) {
    const result = await call('Runtime.evaluate', {
      expression: 'window.slidesReady === true && document.querySelectorAll(".centroid-cloud canvas").length === 3',
      returnByValue: true
    });
    if (result.result.value === true) { ready = true; break; }
    await sleep(100);
  }
  if (!ready) throw new Error('Slides did not finish rendering');

  const pdf = await call('Page.printToPDF', { printBackground: true, preferCSSPageSize: true });
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, Buffer.from(pdf.data, 'base64'));
  console.log(output);
} finally {
  socket?.close();
  chrome?.kill();
  server.kill();
  await fs.rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 100 });
}
