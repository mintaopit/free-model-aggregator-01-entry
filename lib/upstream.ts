// lib/upstream.ts — one inference call to a provider, in its own wire.
//
// Each function takes the client request body (already in the client's
// wire), the model that serves it, and the provider's directory entry, and
// returns the provider's answer normalised across the three wires, or the
// refusal that came back. A refusal's status decides what happens next: 402,
// 429 and 5xx are a spent budget, a scheduled failure or an outage, and the
// caller retries another provider; anything else is a malformed call, which
// no other provider would answer either.
//
// One provider serves one call at a time in this trial, and the client sends
// one request at a time, so calls are strictly sequential here: no second
// call is ever in flight while a first runs.

import {
  anthropicToOpenAI,
  clientToGemini,
  openAIToAnthropic,
  parseAnthropicAnswer,
  parseGeminiAnswer,
  parseOpenAIAnswer,
  type UpstreamAnswer,
} from './translate.ts';
import { authHeaders, joinBase, type Json, type ProviderEntry } from './brief.ts';

export interface UpstreamOk {
  ok: true;
  answer: UpstreamAnswer;
}

export interface UpstreamRefused {
  ok: false;
  /** The HTTP status the provider answered, or 0 when unreachable. */
  status: number;
  /** The error code the provider named, when it named one. */
  code: string | null;
  /** Whether another provider is worth trying for this request. */
  retryable: boolean;
  detail: string;
}

export type UpstreamResult = UpstreamOk | UpstreamRefused;

/**
 * How long one provider call may take. The providers are loopback in a
 * trial and a healthy call answers in milliseconds; a long hang here risks
 * the client's 30 s request timeout when a request fails over across
 * several providers, so a slow provider is treated as a refusal instead.
 */
const CALL_TIMEOUT_MS = 5000;

/** Whether a refusal is a budget/schedule/outage failure worth failing over. */
export function isFailoverStatus(status: number): boolean {
  return status === 402 || status === 429 || status >= 500;
}

function refused(status: number, code: string | null, detail: string): UpstreamRefused {
  return { ok: false, status, code, detail, retryable: isFailoverStatus(status) };
}

/** The error code a provider's refusal body names, in its own wire's shape. */
function errorCodeOf(wire: string, text: string): string | null {
  try {
    const value: unknown = JSON.parse(text);
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return null;
    const body = value as Record<string, unknown>;
    if (wire === 'openai' && body.error !== null && typeof body.error === 'object' && !Array.isArray(body.error)) {
      const code = (body.error as Record<string, unknown>).code;
      return typeof code === 'string' ? code : null;
    }
    if (wire === 'anthropic' && body.error !== null && typeof body.error === 'object' && !Array.isArray(body.error)) {
      const type = (body.error as Record<string, unknown>).type;
      return typeof type === 'string' ? type : null;
    }
    if (wire === 'gemini' && body.error !== null && typeof body.error === 'object' && !Array.isArray(body.error)) {
      const status = (body.error as Record<string, unknown>).status;
      return typeof status === 'string' ? status : null;
    }
    return null;
  } catch {
    return null;
  }
}

async function postJson(url: string, headers: Record<string, string>, body: Json): Promise<Response> {
  return fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', ...headers },
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
  });
}

/**
 * Call an OpenAI-wire provider with a Chat Completions body. When the
 * client asked for a stream, include_usage is forced on: without it the
 * provider sends no usage chunk, and a streamed answer with no usage fails
 * the client's usage check.
 */
export async function callOpenAIWire(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const body = clientApi === 'openai' ? { ...clientBody, model } : anthropicToOpenAI(clientBody, model);
  if (body.stream === true) body.stream_options = { include_usage: true };
  let res: Response;
  try {
    res = await postJson(`${base}/v1/chat/completions`, authHeaders(provider.auth, provider.key), body);
  } catch (err) {
    return refused(0, null, `the provider could not be reached: ${(err as Error).message}`);
  }
  const text = await res.text();
  if (!res.ok) return refused(res.status, errorCodeOf('openai', text), `the provider answered HTTP ${res.status}`);
  const answer = parseOpenAIAnswer(text, body.stream === true);
  if (answer === null) return refused(502, null, 'the provider answered a body that is not one OpenAI answer');
  return { ok: true, answer };
}

/** Call an Anthropic-wire provider with a Messages body. */
export async function callAnthropicWire(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const body = clientApi === 'anthropic' ? { ...clientBody, model } : openAIToAnthropic(clientBody, model);
  let res: Response;
  try {
    res = await postJson(
      `${base}/v1/messages`,
      { 'anthropic-version': '2023-06-01', ...authHeaders(provider.auth, provider.key) },
      body,
    );
  } catch (err) {
    return refused(0, null, `the provider could not be reached: ${(err as Error).message}`);
  }
  const text = await res.text();
  if (!res.ok) return refused(res.status, errorCodeOf('anthropic', text), `the provider answered HTTP ${res.status}`);
  const answer = parseAnthropicAnswer(text, body.stream === true);
  if (answer === null) return refused(502, null, 'the provider answered a body that is not one Anthropic answer');
  return { ok: true, answer };
}

/**
 * Call a Gemini-wire provider. JSON mode is carried as
 * responseMimeType application/json, and only when the OpenAI client asked
 * for it: the Anthropic wire has no JSON mode to carry.
 */
export async function callGeminiWire(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<UpstreamResult> {
  const base = joinBase(envUrl, provider.base);
  const stream = clientBody.stream === true;
  const jsonMode =
    clientApi === 'openai' &&
    typeof clientBody.response_format === 'object' &&
    clientBody.response_format !== null &&
    ((clientBody.response_format as Json).type === 'json_object' || (clientBody.response_format as Json).type === 'json_schema');
  const body = clientToGemini(clientApi, clientBody, jsonMode);
  const path = stream ? ':streamGenerateContent' : ':generateContent';
  let res: Response;
  try {
    res = await postJson(
      `${base}/v1beta/models/${encodeURIComponent(model)}${path}`,
      authHeaders(provider.auth, provider.key),
      body,
    );
  } catch (err) {
    return refused(0, null, `the provider could not be reached: ${(err as Error).message}`);
  }
  const text = await res.text();
  if (!res.ok) return refused(res.status, errorCodeOf('gemini', text), `the provider answered HTTP ${res.status}`);
  const answer = parseGeminiAnswer(text, stream);
  if (answer === null) return refused(502, null, 'the provider answered a body that is not one Gemini answer');
  return { ok: true, answer };
}

/** Call a provider in its own wire, whichever of the three it speaks. */
export async function callProvider(
  envUrl: string,
  provider: ProviderEntry,
  model: string,
  clientApi: 'openai' | 'anthropic',
  clientBody: Json,
): Promise<UpstreamResult> {
  if (provider.wire === 'openai') return callOpenAIWire(envUrl, provider, model, clientApi, clientBody);
  if (provider.wire === 'anthropic') return callAnthropicWire(envUrl, provider, model, clientApi, clientBody);
  return callGeminiWire(envUrl, provider, model, clientApi, clientBody);
}
