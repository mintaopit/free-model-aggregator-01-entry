// main.ts — the free AI model aggregator.
//
// One server, two client APIs (OpenAI Chat Completions and Anthropic
// Messages, JSON and server-sent events), over the ten simulated free-tier
// providers the trial's brief names. For each request it picks the best free
// model that meets the request's requirements, speaks that provider's wire,
// and fails over to the next best free model when a provider refuses with a
// spent budget (402), a rate limit (429) or an outage — because a provider
// that is out takes every request routed to it down with it unless the
// server moves.
//
// Routing, in one paragraph: candidates are every free model meeting the
// request, ranked by the quality table's rating, then by whether their
// provider refused recently (a provider that just failed is likelier to
// fail again on its next call), then by how few calls their provider has
// taken so far (which spreads the load over same-rated providers), then by
// provider and model id for a deterministic order. A provider that refuses
// twice in a row is treated as spent — a scheduled failure is a lone
// refusal, a spent budget or an outage refuses every call after it — and is
// skipped for the rest of the trial. Every choice is a function of the brief, the requests and the providers' answers so far:
// no wall clock and no unseeded randomness, so a replay leaves the same
// state export.
//
// Run it:
//   node main.ts            serve the trial (needs MAKEFUN_ENV_URL and
//                           MAKEFUN_PROMPT_FILE, as the trial sets them)
//   node main.ts --list     print the catalog document and exit 0
//
// Files:
//   main.ts            this server: the brief, the routing, the routes,
//                      the register and the poll.
//   lib/brief.ts       the brief's documents, the catalog probe, the catalog.
//   lib/translate.ts   the six client-wire x upstream-wire translations.
//   lib/upstream.ts    one inference call to a provider, in its own wire.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  isRecord,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingRowOf,
  readBrief,
  type Brief,
  type CatalogReport,
  type Json,
  type ListedModel,
  type Probe,
  type ProviderEntry,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  answerToAnthropicJson,
  answerToAnthropicStream,
  answerToOpenAIJson,
  answerToOpenAIStream,
  openAIError,
  openAIModelList,
  type UpstreamAnswer,
} from './lib/translate.ts';
import { callProvider, isFailoverStatus, type UpstreamResult } from './lib/upstream.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Two refusals in a row mean a spent budget or an outage, not one bad call. */
const EXHAUSTED_AFTER_FAILURES = 2;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief: Brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order. */
let manifestIndex = 0;

/** Per-provider call state, for budget-aware routing and exhaustion. */
interface ProviderState {
  calls: number;
  consecutiveFailures: number;
  exhausted: boolean;
}
const providerState = new Map<string, ProviderState>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, consecutiveFailures: 0, exhausted: false };
    providerState.set(providerId, state);
  }
  return state;
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
  for (const probe of probes) stateOf(probe.provider.id);
}

interface Candidate {
  provider: ProviderEntry;
  model: ListedModel;
  rating: number;
}

/** The free models across the providers that answered, each id once: what GET /v1/models lists. */
function servedModels(): { id: string; provider: string; display_name: string }[] {
  const seen = new Set<string>();
  const out: { id: string; provider: string; display_name: string }[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0))) {
      if (seen.has(model.id)) continue;
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out.sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Every free model meeting `requires`, best first: the quality table's
 * rating, then a provider with no recent refusal (a scheduled failure is a
 * lone refusal, so a provider that just failed is more likely to fail
 * again on its next call than one that has not), then the provider with
 * the fewest calls so far (load spread over same-rated providers), then
 * provider and model id for a fixed order.
 */
function candidatesFor(requires: Requirement): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      const row = ratingRowOf(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ provider: probe.provider, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => {
    if (b.rating !== a.rating) return b.rating - a.rating;
    const failedA = stateOf(a.provider.id).consecutiveFailures > 0 ? 1 : 0;
    const failedB = stateOf(b.provider.id).consecutiveFailures > 0 ? 1 : 0;
    if (failedA !== failedB) return failedA - failedB;
    const callsA = stateOf(a.provider.id).calls;
    const callsB = stateOf(b.provider.id).calls;
    if (callsA !== callsB) return callsA - callsB;
    if (a.provider.id !== b.provider.id) return a.provider.id < b.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    tools: Array.isArray(body.tools) && body.tools.length > 0,
    vision,
    json: format === 'json_object' || format === 'json_schema',
  };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendText(res: ServerResponse, status: number, body: string, contentType: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface ServedAnswer {
  answer: UpstreamAnswer;
  model: string;
  provider: string;
}

/**
 * Serve one client request: try the candidates best first, failing over on
 * a refusal, until one answers. A provider that refuses twice running is
 * spent for the trial and is skipped from then on. Returns null when no
 * candidate answered.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement): Promise<ServedAnswer | null> {
  const candidates = candidatesFor(requires);
  let lastRefusal: string | null = null;
  for (const candidate of candidates) {
    const state = stateOf(candidate.provider.id);
    if (state.exhausted) continue;
    state.calls += 1;
    let result: UpstreamResult;
    try {
      result = await callProvider(ENV_URL, candidate.provider, candidate.model.id, clientApi, body);
    } catch (err) {
      log(`call to ${candidate.provider.id}/${candidate.model.id} threw: ${(err as Error).message}`);
      result = { ok: false, status: 0, code: null, detail: (err as Error).message, retryable: false };
    }
    if (result.ok) {
      state.consecutiveFailures = 0;
      return { answer: result.answer, model: candidate.model.id, provider: candidate.provider.id };
    }
    lastRefusal = `${candidate.provider.id}: ${result.detail}`;
    log(`refused by ${candidate.provider.id}/${candidate.model.id}: HTTP ${result.status} ${result.code ?? ''} (${result.detail})`);
    if (isFailoverStatus(result.status)) {
      state.consecutiveFailures += 1;
      if (state.consecutiveFailures >= EXHAUSTED_AFTER_FAILURES) {
        state.exhausted = true;
        log(`${candidate.provider.id} looks spent (refused ${state.consecutiveFailures} running): skipping it`);
      }
    }
  }
  if (lastRefusal !== null) log(`no candidate answered: last refusal ${lastRefusal}`);
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered on any upstream wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('openai', body, requirementsFor(body));
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_unavailable', 'no free model on the directory answered this request');
  }
  if (body.stream === true) {
    sendText(res, 200, answerToOpenAIStream(served.answer, served.model), 'text/event-stream; charset=utf-8');
    return;
  }
  sendJson(res, 200, answerToOpenAIJson(served.answer, served.model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered on any upstream wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const served = await serve('anthropic', body, requirementsFor(body));
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model on the directory answered this request');
  }
  if (body.stream === true) {
    sendText(res, 200, answerToAnthropicStream(served.answer, served.model), 'text/event-stream; charset=utf-8');
    return;
  }
  sendJson(res, 200, answerToAnthropicJson(served.answer, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the
    // one built object and stop before any server starts. With no brief and
    // no env to probe, the catalog is empty, as the starter's is.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
