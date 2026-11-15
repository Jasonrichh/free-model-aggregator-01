// free-model-aggregator-01: one server, two client wires, ten simulated free providers.
//
// The trial gives this server a brief (the provider directory, the quality table and the
// workload manifest) and an env that serves the providers. This server reads the brief, probes
// each provider's catalog route in the provider's own wire, builds one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it
//      got: x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and
//      POST /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
// register body's catalog must be that same document, byte for byte.
//
// ROUTING
//   Every request names model "auto", so the model is this server's to choose. For each request it
//   ranks the free models that meet the request's requirements by the quality table's rating_milli
//   (one candidate a provider: the provider's best model that meets the request; a budget is spent
//   a provider at a time, so a second model on the same provider is no fallback). It tries them in
//   that order and answers with the first provider that serves the request: exactly one successful
//   provider call answers a request, and a call that succeeds is always the answer (a second
//   success would fail the trial's one-provider check).
//
//   The budgets are never published, so the router learns them from the providers' answers:
//   - a 402 or 429 is counted against its provider, and the request moves to the next candidate;
//   - a failure that carried a retry-after is respected by not hammering that provider again at
//     once (no immediate retry); a failure without one, on the top candidate's first failure in a
//     while, earns one immediate retry, because such a failure is usually the provider's own
//     failure schedule and its next call serves;
//   - two failures in a row mark a provider spent (its budget is gone or it is out): it is
//     skipped while any other candidate remains, and is tried again only when nothing else is left.
//   - when every candidate has failed once, the request is tried again on the same order: a
//     failure on a provider's schedule passes, and the retry may serve where the first call did not.
//
// WIRES
//   The client speaks OpenAI or Anthropic; a provider speaks openai, anthropic or gemini. The
//   request is respelled into the provider's wire with its system prompt, user text (images
//   included), tools, stop sequences, max_tokens and JSON mode untouched, and the provider's
//   answer is respelled back into the client's wire with its text whole, its usage counts as that
//   wire reports them, and its stop reason mapped. A JSON-mode request from an OpenAI client is
//   never sent to an anthropic-wire provider: that wire cannot carry response_format, so its echo
//   would say JSON mode was absent. Same-wire answers pass through byte for byte; cross-wire
//   answers are parsed and re-emitted (streams are buffered first, so the usage the provider
//   reports last can be named first, as the Anthropic wire wants it).

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type RatingRow,
  type Requirement,
} from './lib/providers.ts';
import {
  answerToAnthropicJson,
  answerToAnthropicSse,
  answerToOpenAIJson,
  answerToOpenAISse,
  anthropicError,
  anthropicModelList,
  anthropicToGemini,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropic,
  openAIToGemini,
  parseAnthropicJson,
  parseAnthropicSse,
  parseGeminiJson,
  parseGeminiSse,
  parseOpenAIJson,
  parseOpenAISse,
  type ProviderAnswer,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const CALL_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive failures after which a provider counts as spent (its budget gone or it out). */
const SPENT_AFTER_CONSECUTIVE_FAILURES = 2;
/** Retry the top candidate once on its first failure in a while, unless it sent retry-after. */
const IMMEDIATE_RETRY_TOP = true;
/** How many times POST /register is attempted before the trial is given up. */
const REGISTER_ATTEMPTS = 10;
/** Milliseconds between register attempts. */
const REGISTER_RETRY_MS = 200;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Routing: what the router knows about each provider, learned from its answers
// ---------------------------------------------------------------------------

interface ProviderState {
  /** Inference calls attempted, successes and refusals alike. */
  calls: number;
  /** Inference calls refused (a 402, a 429, a 5xx, or unreachable). */
  fails: number;
  /** Refusals in a row, since the provider last served a request. */
  consecFails: number;
}

const providerState = new Map<string, ProviderState>();

function stateOf(providerId: string): ProviderState {
  let state = providerState.get(providerId);
  if (state === undefined) {
    state = { calls: 0, fails: 0, consecFails: 0 };
    providerState.set(providerId, state);
  }
  return state;
}

function recordSuccess(providerId: string): void {
  const state = stateOf(providerId);
  state.calls += 1;
  state.consecFails = 0;
}

function recordFailure(providerId: string): void {
  const state = stateOf(providerId);
  state.calls += 1;
  state.fails += 1;
  state.consecFails += 1;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  row: RatingRow | null;
  rating: number;
}

/**
 * The candidates for one request: one a provider, its best free model that meets the
 * requirements, ranked by the quality table's rating, then by provider and model id so the order
 * is the same on a replay. An OpenAI JSON-mode request never goes to an anthropic-wire provider:
 * that wire cannot carry response_format, so its echo would say JSON mode was absent.
 */
function candidatesFor(requires: Requirement, jsonModeOpenAI: boolean): Candidate[] {
  const byProvider = new Map<string, Candidate>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonModeOpenAI && probe.provider.wire === 'anthropic') continue;
    let best: Candidate | null = null;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      const rating = row?.rating_milli ?? 0;
      const candidate: Candidate = { probe, model, row, rating };
      if (best === null || rating > best.rating || (rating === best.rating && model.id < best.model.id)) best = candidate;
    }
    if (best !== null) byProvider.set(probe.provider.id, best);
  }
  return [...byProvider.values()].sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0));
}

/** Whether a provider counts as spent: two refusals in a row means its budget is gone or it is out. */
function isSpent(providerId: string): boolean {
  return stateOf(providerId).consecFails >= SPENT_AFTER_CONSECUTIVE_FAILURES;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  const fromBody = requirementsFromBody(body);
  if (entry === undefined) return fromBody;
  // The manifest is the declared source of each request's requirements; the body is the ground
  // truth for what it carries, so the two are unioned (a model meeting the union meets either).
  return {
    tools: entry.requires.tools === true || fromBody.tools === true,
    vision: entry.requires.vision === true || fromBody.vision === true,
    json: entry.requires.json === true || fromBody.json === true,
    ...(entry.requires.min_context !== undefined ? { min_context: entry.requires.min_context } : {}),
  };
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
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether this OpenAI client request asks for JSON mode (its response_format says so). */
function jsonModeOpenAI(body: Json): boolean {
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return format === 'json_object' || format === 'json_schema';
}

// ---------------------------------------------------------------------------
// One provider call
// ---------------------------------------------------------------------------

interface CallResult {
  ok: boolean;
  status: number;
  retryAfter: boolean;
  /** The provider's answer body when the call served (JSON or SSE text). */
  text: string | null;
  /** The provider's answer stream when the call served and the client streams on the same wire. */
  stream: ReadableStream<Uint8Array> | null;
}

/** The upstream request for one candidate: its URL, headers and body, in the provider's wire. */
function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; payload: string } {
  const probe = candidate.probe;
  const base = joinBase(ENV_URL, probe.provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(probe.provider.auth, probe.provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  if (probe.provider.wire === 'openai') {
    const upstream = clientApi === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (upstream.stream === true) upstream.stream_options = { include_usage: true };
    return { url: `${base}/v1/chat/completions`, headers, payload: JSON.stringify(upstream) };
  }
  if (probe.provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const upstream = clientApi === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropic(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, payload: JSON.stringify(upstream) };
  }
  const built = clientApi === 'openai' ? openAIToGemini(body, candidate.model.id) : anthropicToGemini(body, candidate.model.id);
  const action = built.action === 'streamGenerateContent' ? `${built.action}?alt=sse` : built.action;
  return {
    url: `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${action}`,
    headers,
    payload: JSON.stringify(built.body),
  };
}

/** One inference call to one candidate; a refusal is a result, never a throw. */
async function callCandidate(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<CallResult> {
  const { url, headers, payload } = upstreamRequest(candidate, clientApi, body);
  let res: Response;
  try {
    res = await fetch(url, { method: 'POST', headers, body: payload, signal: AbortSignal.timeout(CALL_TIMEOUT_MS) });
  } catch {
    return { ok: false, status: 0, retryAfter: false, text: null, stream: null };
  }
  const retryAfter = res.headers.has('retry-after');
  if (!res.ok) {
    try {
      await res.text();
    } catch {
      // The refusal's body does not matter.
    }
    return { ok: false, status: res.status, retryAfter, text: null, stream: null };
  }
  if (stream && res.body !== null) return { ok: true, status: res.status, retryAfter, text: null, stream: res.body };
  return { ok: true, status: res.status, retryAfter, text: await res.text(), stream: null };
}

/** Parse one provider's answer into the wire-neutral shape, for a cross-wire reply. */
function parseAnswer(candidate: Candidate, result: CallResult, stream: boolean): ProviderAnswer | null {
  if (result.text === null) return null;
  const wire = candidate.probe.provider.wire;
  try {
    if (!stream) {
      const value: unknown = JSON.parse(result.text);
      if (!isRecord(value)) return null;
      if (wire === 'openai') return parseOpenAIJson(value);
      if (wire === 'anthropic') return parseAnthropicJson(value);
      return parseGeminiJson(value);
    }
    if (wire === 'openai') return parseOpenAISse(result.text);
    if (wire === 'anthropic') return parseAnthropicSse(result.text);
    return parseGeminiSse(result.text);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// HTTP plumbing
// ---------------------------------------------------------------------------

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

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * Answer one client request from the first provider that serves it. Returns true when an answer
 * was sent. The attempt order is: every candidate that does not count as spent, then the same
 * order again (a failure on a provider's schedule passes, and the retry may serve), then the
 * spent ones last. The top candidate earns one immediate retry on its first failure in a while,
 * unless that failure carried a retry-after.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<boolean> {
  const stream = body.stream === true;
  const requires = requirementsFor(body);
  const candidates = candidatesFor(requires, clientApi === 'openai' && jsonModeOpenAI(body));
  if (candidates.length === 0) {
    sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
    return false;
  }
  const fresh = candidates.filter((c) => !isSpent(c.probe.provider.id));
  const spent = candidates.filter((c) => isSpent(c.probe.provider.id));
  // Each entry is [candidate, round]: round 1 the first pass, round 2 the retry pass, round 3
  // the spent providers last.
  const order: [Candidate, number][] = [
    ...fresh.map((c): [Candidate, number] => [c, 1]),
    ...fresh.map((c): [Candidate, number] => [c, 2]),
    ...spent.map((c): [Candidate, number] => [c, 3]),
  ];
  for (let i = 0; i < order.length; i++) {
    const [candidate, round] = order[i];
    if (round < 3 && isSpent(candidate.probe.provider.id)) continue;
    const result = await callCandidate(candidate, clientApi, body, stream);
    if (result.ok) {
      recordSuccess(candidate.probe.provider.id);
      await reply(candidate, clientApi, body, result, stream, res);
      return true;
    }
    recordFailure(candidate.probe.provider.id);
    log(`request: ${candidate.probe.provider.id}/${candidate.model.id} refused HTTP ${result.status} (round ${round})`);
    // One immediate retry for the top candidate's first failure in a while, unless the provider
    // asked, with a retry-after, to be left alone.
    if (IMMEDIATE_RETRY_TOP && round === 1 && i === 0 && stateOf(candidate.probe.provider.id).consecFails === 1 && !result.retryAfter) {
      const retry = await callCandidate(candidate, clientApi, body, stream);
      if (retry.ok) {
        recordSuccess(candidate.probe.provider.id);
        await reply(candidate, clientApi, body, retry, stream, res);
        return true;
      }
      recordFailure(candidate.probe.provider.id);
      log(`request: ${candidate.probe.provider.id}/${candidate.model.id} refused again HTTP ${retry.status} (immediate retry)`);
    }
  }
  sendError(res, clientApi, 503, clientApi === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
  return false;
}

/**
 * Send one provider's answer back in the client's wire. Same-wire answers pass through byte for
 * byte; cross-wire answers are parsed and re-emitted, streams buffered first.
 */
async function reply(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json, result: CallResult, stream: boolean, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  const sameWire = (clientApi === 'openai' && wire === 'openai') || (clientApi === 'anthropic' && wire === 'anthropic');
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (sameWire && result.stream !== null) {
      await pipeStream(result.stream, res);
      return;
    }
    // Cross-wire: read the provider's stream whole, then emit the client's event order.
    const text = result.stream !== null ? await new Response(result.stream).text() : (result.text ?? '');
    const answer = parseAnswer(candidate, { ...result, text }, true);
    if (answer === null) {
      res.end();
      return;
    }
    res.end(clientApi === 'openai' ? answerToOpenAISse(answer, candidate.model.id) : answerToAnthropicSse(answer, candidate.model.id));
    return;
  }
  if (sameWire) {
    sendText(res, 200, 'application/json', result.text ?? '');
    return;
  }
  const answer = parseAnswer(candidate, result, false);
  if (answer === null) {
    sendError(res, clientApi, 502, clientApi === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered in a shape this server could not read');
    return;
  }
  sendJson(res, 200, clientApi === 'openai' ? answerToOpenAIJson(answer, candidate.model.id) : answerToAnthropicJson(answer, candidate.model.id));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the routed provider. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the routed provider. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
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

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: unknown = null;
  for (let attempt = 1; attempt <= REGISTER_ATTEMPTS; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err;
      log(`register attempt ${attempt} failed: ${(err as Error).message}`);
      if (attempt < REGISTER_ATTEMPTS) await sleep(REGISTER_RETRY_MS);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(String(lastError));
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
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
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
