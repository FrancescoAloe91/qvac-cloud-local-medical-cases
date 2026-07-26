/**
 * Minimal HTTP sidecar for QVAC SDK MedPsy inference.
 * Python benchmark engine calls POST /generate { "prompt": "..." }.
 *
 * Env:
 *   QVAC_SIDECAR_PORT   default 8787
 *   QVAC_MODEL_PATH     path to MedPsy GGUF
 *   QVAC_DEVICE         "gpu" (default) | "cpu"
 *   QVAC_GPU_LAYERS     default 99 (0 = CPU-only layers)
 *   QVAC_CTX_SIZE       default 4096
 *   QVAC_PREDICT        max new tokens, default 3000 (aligned with cloud candidates)
 *   QVAC_MAIN_GPU       optional: "dedicated" | "integrated" | "0"
 *   QVAC_WARM_LOAD      "1" (default) preload model at startup
 */
import http from "node:http";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";
import fs from "node:fs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.QVAC_SIDECAR_PORT || 8787);
const HOST = process.env.QVAC_SIDECAR_HOST || "127.0.0.1";

function spaceFreePath(absPath) {
  // QVAC SDK worker fails on file:// URLs that contain spaces (%20). Prefer a
  // space-free symlink under ~/.local when the path has spaces.
  const resolved = path.resolve(absPath);
  if (!resolved.includes(" ") && fs.existsSync(resolved)) {
    return resolved;
  }
  const linkDir = path.join(
    process.env.HOME || "/tmp",
    ".local",
    "qvac-models"
  );
  const linkPath = path.join(linkDir, path.basename(resolved));
  try {
    fs.mkdirSync(linkDir, { recursive: true });
    if (fs.existsSync(resolved)) {
      try {
        fs.unlinkSync(linkPath);
      } catch {
        /* ignore */
      }
      fs.symlinkSync(resolved, linkPath);
      return linkPath;
    }
  } catch (err) {
    console.warn(
      `[qvac-sidecar] could not create space-free model link: ${err?.message || err}`
    );
  }
  return resolved;
}

function resolveModelPath() {
  if (process.env.QVAC_MODEL_PATH) {
    return spaceFreePath(process.env.QVAC_MODEL_PATH);
  }
  const repoModel = path.resolve(
    __dirname,
    "..",
    "models",
    "medpsy-4b-q4_k_m-imat.gguf"
  );
  return spaceFreePath(repoModel);
}

let MODEL_PATH = resolveModelPath();

const DEVICE = (process.env.QVAC_DEVICE || "gpu").toLowerCase();
const GPU_LAYERS = Number(
  process.env.QVAC_GPU_LAYERS ?? (DEVICE === "cpu" ? 0 : 99)
);
const CTX_SIZE = Number(process.env.QVAC_CTX_SIZE || 4096);
const PREDICT = Number(process.env.QVAC_PREDICT || 3000);
const WARM_LOAD = (process.env.QVAC_WARM_LOAD || "1") !== "0";

function buildModelConfig() {
  const cfg = {
    device: DEVICE === "cpu" ? "cpu" : "gpu",
    gpu_layers: GPU_LAYERS,
    ctx_size: CTX_SIZE,
    predict: PREDICT,
    // Keep stock-ish sampling; cap length for demo latency
    temp: 0.6,
    top_k: 20,
    top_p: 0.95,
  };
  const mainGpu = process.env.QVAC_MAIN_GPU;
  if (mainGpu === "dedicated" || mainGpu === "integrated") {
    cfg["main-gpu"] = mainGpu;
  } else if (mainGpu != null && mainGpu !== "" && !Number.isNaN(Number(mainGpu))) {
    cfg["main-gpu"] = Number(mainGpu);
  }
  return cfg;
}

let sdk = null;
let modelId = null;
let loadError = null;
let activeConfig = null;
let lastRamMb = null;

/** RSS of this Node process + descendant workers (llama/Metal), in MB. */
function sampleRamMb() {
  try {
    const out = execFileSync("ps", ["-axo", "pid=,ppid=,rss="], {
      encoding: "utf8",
      maxBuffer: 8 * 1024 * 1024,
    });
    const rows = [];
    for (const line of out.split("\n")) {
      const parts = line.trim().split(/\s+/);
      if (parts.length < 3) continue;
      const pid = Number(parts[0]);
      const ppid = Number(parts[1]);
      const rssKb = Number(parts[2]);
      if (!Number.isFinite(pid) || !Number.isFinite(ppid) || !Number.isFinite(rssKb)) {
        continue;
      }
      rows.push({ pid, ppid, rssKb });
    }
    const byPid = new Map(rows.map((r) => [r.pid, r]));
    const kids = new Set([process.pid]);
    let grew = true;
    while (grew) {
      grew = false;
      for (const r of rows) {
        if (kids.has(r.ppid) && !kids.has(r.pid)) {
          kids.add(r.pid);
          grew = true;
        }
      }
    }
    let totalKb = 0;
    for (const pid of kids) {
      const r = byPid.get(pid);
      if (r) totalKb += r.rssKb;
    }
    if (totalKb > 0) {
      return Math.round((totalKb / 1024) * 10) / 10;
    }
  } catch {
    /* fall through */
  }
  return Math.round((process.memoryUsage().rss / (1024 * 1024)) * 10) / 10;
}

function ggufSizeMb(p) {
  try {
    if (!p || !fs.existsSync(p)) return null;
    return Math.round((fs.statSync(p).size / (1024 * 1024)) * 10) / 10;
  } catch {
    return null;
  }
}

function refreshRam() {
  lastRamMb = sampleRamMb();
  return lastRamMb;
}

function modelTagFromPath(p) {
  const base = path.basename(p || "").replace(/\.gguf$/i, "");
  return base || "medpsy";
}

async function unloadCurrent() {
  if (!modelId) return;
  const id = modelId;
  try {
    if (sdk?.unloadModel) {
      // Keep RPC alive so the next /load can reuse the worker (Node auto-closes by default).
      await sdk.unloadModel({ modelId: id, autoClose: false });
      console.log(`[qvac-sidecar] unloaded ${modelTagFromPath(MODEL_PATH)}`);
    }
  } catch (err) {
    console.warn(
      `[qvac-sidecar] unload warning: ${err?.message || err}`
    );
  }
  modelId = null;
  activeConfig = null;
  lastRamMb = null;
}

async function ensureModel() {
  if (modelId) return modelId;
  // Always allow a fresh load attempt (warm-load / prior RPC timeouts must not latch forever).
  loadError = null;
  try {
    sdk = await import("@qvac/sdk");
  } catch (err) {
    loadError = new Error(
      `Cannot import @qvac/sdk. Run: cd sidecar && npm install\n${err}`
    );
    throw loadError;
  }
  if (!fs.existsSync(MODEL_PATH)) {
    loadError = new Error(
      `MedPsy GGUF not found at ${MODEL_PATH}. Set QVAC_MODEL_PATH or download the model.`
    );
    throw loadError;
  }
  // Prefer absolute filesystem path (file:// with %20 spaces breaks the worker).
  const modelSrc = MODEL_PATH;
  // Weight load can exceed the default RPC call timeout; worker IPC init is separate (30s).
  const LOAD_TIMEOUT_MS = Number(process.env.QVAC_LOAD_TIMEOUT_MS || 300000);
  const tryLoad = async (modelConfig) => {
    console.log(
      `[qvac-sidecar] loading ${modelTagFromPath(MODEL_PATH)} · device=${modelConfig.device} · gpu_layers=${modelConfig.gpu_layers} · ctx=${modelConfig.ctx_size} · predict=${modelConfig.predict}`
    );
    return sdk.loadModel(
      {
        modelSrc,
        modelType: "llamacpp-completion",
        modelConfig,
        onProgress: (p) => {
          if (p?.percentage != null) {
            process.stderr.write(`\r[qvac] load ${p.percentage.toFixed(0)}%`);
            if (p.percentage >= 100) process.stderr.write("\n");
          }
        },
      },
      { timeout: LOAD_TIMEOUT_MS }
    );
  };

  let cfg = buildModelConfig();
  try {
    modelId = await tryLoad(cfg);
    activeConfig = cfg;
  } catch (err) {
    // Apple Silicon / some GGUF builds hit Metal crashes — fall back to CPU once.
    if (cfg.device === "gpu") {
      console.warn(
        `[qvac-sidecar] GPU load failed (${err?.message || err}); retrying on CPU…`
      );
      cfg = {
        ...cfg,
        device: "cpu",
        gpu_layers: 0,
      };
      try {
        modelId = await tryLoad(cfg);
        activeConfig = cfg;
      } catch (err2) {
        loadError = err2;
        throw err2;
      }
    } else {
      loadError = err;
      throw err;
    }
  }
  refreshRam();
  console.log(
    `[qvac-sidecar] model ready · ${modelTagFromPath(MODEL_PATH)} · device=${activeConfig?.device} · gpu_layers=${activeConfig?.gpu_layers} · ram≈${lastRamMb} MB`
  );
  return modelId;
}

/** Hot-swap GGUF for multi-QVAC compare (unload + load). */
async function loadFromPath(requestedPath) {
  if (!requestedPath || !String(requestedPath).trim()) {
    throw new Error("Missing model_path");
  }
  const next = spaceFreePath(String(requestedPath).trim());
  if (!fs.existsSync(next)) {
    throw new Error(`MedPsy GGUF not found at ${next}`);
  }
  if (modelId && MODEL_PATH === next) {
    refreshRam();
    return {
      ok: true,
      modelLoaded: true,
      modelPath: MODEL_PATH,
      model: modelTagFromPath(MODEL_PATH),
      reused: true,
      device: activeConfig?.device || DEVICE,
      gpu_layers: activeConfig?.gpu_layers ?? GPU_LAYERS,
      ram_mb: lastRamMb,
      gguf_mb: ggufSizeMb(MODEL_PATH),
    };
  }
  await unloadCurrent();
  MODEL_PATH = next;
  loadError = null;
  await ensureModel();
  return {
    ok: true,
    modelLoaded: true,
    modelPath: MODEL_PATH,
    model: modelTagFromPath(MODEL_PATH),
    reused: false,
    device: activeConfig?.device || DEVICE,
    gpu_layers: activeConfig?.gpu_layers ?? GPU_LAYERS,
    ram_mb: lastRamMb,
    gguf_mb: ggufSizeMb(MODEL_PATH),
  };
}

/** Extract plain text from SDK token chunks (never emit "[object Object]"). */
function extractTokenText(item) {
  if (item == null) return "";
  if (typeof item === "string") return item;
  if (typeof item === "number" || typeof item === "boolean") return String(item);
  if (typeof item !== "object") return "";
  if (item.__done) return "";
  const keys = ["token", "text", "content", "delta", "value"];
  for (const k of keys) {
    const v = item[k];
    if (typeof v === "string" && v) return v;
  }
  if (item.message && typeof item.message.content === "string") {
    return item.message.content;
  }
  return "";
}

async function* tokenStream(prompt) {
  const id = await ensureModel();
  const history = [{ role: "user", content: prompt }];
  const t0 = Date.now();
  let ttftMs = null;
  let content = "";
  let tokenCount = 0;

  const result = sdk.completion({ modelId: id, history, stream: true });

  const mark = (tok) => {
    const text = extractTokenText(tok);
    if (!text) return null;
    if (ttftMs == null) ttftMs = Date.now() - t0;
    content += text;
    tokenCount += 1;
    return text;
  };

  if (typeof result === "string") {
    const t = mark(result);
    if (t) yield t;
  } else if (result && typeof result.then === "function") {
    const resolved = await result;
    if (typeof resolved === "string") {
      const t = mark(resolved);
      if (t) yield t;
    } else if (resolved?.tokenStream) {
      for await (const tok of resolved.tokenStream) {
        const t = mark(tok);
        if (t) yield t;
      }
    } else {
      const t = mark(
        String(resolved?.text || resolved?.content || resolved?.message?.content || "")
      );
      if (t) yield t;
    }
  } else if (result?.tokenStream) {
    for await (const tok of result.tokenStream) {
      const t = mark(tok);
      if (t) yield t;
    }
  } else {
    const t = mark(String(result?.content || result?.text || ""));
    if (t) yield t;
  }

  const latencyS = (Date.now() - t0) / 1000;
  const ttftS = ttftMs != null ? ttftMs / 1000 : null;
  const genS = ttftS != null ? Math.max(latencyS - ttftS, 0.05) : latencyS;
  const approxTokens =
    tokenCount > 1 ? tokenCount : Math.max(1, content.split(/\s+/).length);
  const tps = genS > 0 ? Math.round((approxTokens / genS) * 10) / 10 : null;
  refreshRam();

  yield {
    __done: true,
    content,
    model: modelTagFromPath(MODEL_PATH),
    device: activeConfig?.device || DEVICE,
    gpu_layers: activeConfig?.gpu_layers ?? GPU_LAYERS,
    latency_s: Math.round(latencyS * 1000) / 1000,
    ttft_s: ttftS != null ? Math.round(ttftS * 1000) / 1000 : null,
    tps,
    completion_tokens: approxTokens,
    prompt_tokens: 0,
    cost_usd: 0,
    ram_mb: lastRamMb,
    gguf_mb: ggufSizeMb(MODEL_PATH),
  };
}

async function generate(prompt) {
  let meta = null;
  for await (const item of tokenStream(prompt)) {
    if (item && typeof item === "object" && item.__done) meta = item;
  }
  if (!meta) {
    throw new Error("Empty generation");
  }
  const { __done, ...out } = meta;
  return out;
}

function formatErr(err) {
  const parts = [];
  const push = (s) => {
    const t = String(s || "").trim();
    if (t && !parts.some((p) => p.includes(t.slice(0, 48)))) parts.push(t);
  };
  push(err?.message || err);
  let cur = err?.cause;
  for (let i = 0; i < 4 && cur; i += 1) {
    push(cur?.message || cur);
    const stderr = cur?.stderr || cur?.cause?.stderr;
    if (stderr) push(String(stderr).split("\n").slice(0, 4).join(" "));
    cur = cur?.cause;
  }
  const blob = parts.join(" ");
  if (/libssl\.3|openssl@3/i.test(blob)) {
    push(
      "Fix: install Homebrew OpenSSL 3 (`brew install openssl@3`), then restart the sidecar"
    );
  }
  return parts.join(" — ");
}

function send(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, {
    "Content-Type": "application/json",
    "Content-Length": Buffer.byteLength(body),
  });
  res.end(body);
}

function writeNdjson(res, obj) {
  res.write(JSON.stringify(obj) + "\n");
}

const server = http.createServer(async (req, res) => {
  const urlPath = (req.url || "").split("?")[0];
  if (req.method === "GET" && urlPath === "/health") {
    if (modelId) refreshRam();
    return send(res, 200, {
      ok: true,
      modelLoaded: Boolean(modelId),
      modelPath: MODEL_PATH,
      model: modelId ? modelTagFromPath(MODEL_PATH) : null,
      device: activeConfig?.device || (modelId ? DEVICE : null),
      gpu_layers: activeConfig?.gpu_layers ?? (modelId ? GPU_LAYERS : null),
      ctx_size: CTX_SIZE,
      predict: PREDICT,
      stream: true,
      ram_mb: modelId ? lastRamMb : null,
      gguf_mb: modelId ? ggufSizeMb(MODEL_PATH) : null,
      lastError: loadError ? formatErr(loadError) : null,
    });
  }

  if (req.method === "POST" && urlPath === "/load") {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    let body = {};
    try {
      body = JSON.parse(raw || "{}");
    } catch {
      return send(res, 400, { error: "Invalid JSON body" });
    }
    try {
      const out = await loadFromPath(body.model_path || body.path || "");
      return send(res, 200, out);
    } catch (err) {
      loadError = err;
      return send(res, 500, { error: formatErr(err), modelLoaded: false });
    }
  }

  if (
    req.method === "POST" &&
    (urlPath === "/generate" || urlPath === "/generate/stream")
  ) {
    let raw = "";
    for await (const chunk of req) raw += chunk;
    let body = {};
    try {
      body = JSON.parse(raw || "{}");
    } catch {
      return send(res, 400, { error: "Invalid JSON body" });
    }
    const prompt = body.prompt || "";
    if (!prompt.trim()) {
      return send(res, 400, { error: "Missing prompt" });
    }
    const wantStream =
      urlPath === "/generate/stream" || body.stream === true;

    if (wantStream) {
      res.writeHead(200, {
        "Content-Type": "application/x-ndjson; charset=utf-8",
        "Cache-Control": "no-cache",
        Connection: "keep-alive",
      });
      try {
        for await (const item of tokenStream(prompt)) {
          if (item && typeof item === "object" && item.__done) {
            const { __done, ...meta } = item;
            writeNdjson(res, { type: "done", ...meta });
          } else {
            const tok = extractTokenText(item);
            if (tok) writeNdjson(res, { type: "token", token: tok });
          }
        }
      } catch (err) {
        writeNdjson(res, { type: "error", error: formatErr(err) });
      }
      return res.end();
    }

    try {
      const out = await generate(prompt);
      return send(res, 200, out);
    } catch (err) {
      return send(res, 500, { error: formatErr(err) });
    }
  }
  send(res, 404, { error: "Not found" });
});

server.listen(PORT, HOST, async () => {
  console.log(`[qvac-sidecar] http://${HOST}:${PORT}`);
  console.log(`[qvac-sidecar] model path: ${MODEL_PATH}`);
  console.log(
    `[qvac-sidecar] prefer device=${DEVICE} gpu_layers=${GPU_LAYERS} (SDK default is GPU/Metal on Mac)`
  );
  console.log(`[qvac-sidecar] GET /health  POST /generate  POST /load`);
  if (WARM_LOAD) {
    try {
      await ensureModel();
    } catch (err) {
      // Do not permanently latch loadError from warm-load — allow first /generate to retry.
      console.error(`[qvac-sidecar] warm load failed: ${formatErr(err)}`);
      loadError = null;
      modelId = null;
      activeConfig = null;
    }
  }
});
