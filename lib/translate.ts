// lib/translate.ts — the six client-wire × upstream-wire translations.
//
// The trial's client speaks OpenAI Chat Completions or Anthropic Messages;
// each provider speaks its own wire (OpenAI, Anthropic or Gemini). Every
// translation here is a pure function of the request body and the provider's
// answer: no clock, no randomness, so a replay replays byte for byte.
//
// What must survive a translation, because the env's client compares the
// provider's echo with what it sent: the system prompt, the user text, the
// images (by bytes), the tool names, max_tokens, the stop sequences and JSON
// mode. What the providers never do in this trial: answer with a tool call
// (their answers are text echoes), or stop for any reason but the end of the
// text — so every answer maps to the "stop" stop family on both client wires.
//
// Responses are buffered whole before they are re-emitted: the answers are
// short echoes, and buffering lets a streamed translation carry the usage
// the provider reports in its final frame from the very first event.

import { isRecord, type Json } from './brief.ts';

/** One provider answer, normalised across the three wires. */
export interface UpstreamAnswer {
  /** The answer's text, whole. */
  text: string;
  /** The provider's input token count. */
  inputTokens: number;
  /** The provider's output token count. */
  outputTokens: number;
  /** The stop family: 0 stop/end_turn/STOP, 1 length, 2 tool use, 3 refusal. */
  stopFamily: number;
  /** Tool calls the provider made, if any (never in this trial). */
  toolCalls: { id: string; name: string; arguments: Json }[];
}

/** The text of a string or of a block array's text blocks; '' when none. */
export function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content
    .flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : []))
    .join('\n');
}

/** A tool call's arguments as an object: the provider's JSON string, or {}. */
export function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

function openAIFinishReason(family: number): string {
  return family === 1 ? 'length' : family === 2 ? 'tool_calls' : family === 3 ? 'content_filter' : 'stop';
}

function anthropicStopReason(family: number): string {
  return family === 1 ? 'max_tokens' : family === 2 ? 'tool_use' : family === 3 ? 'refusal' : 'end_turn';
}

// ---------------------------------------------------------------------------
// Shared request pieces
// ---------------------------------------------------------------------------

/** One Anthropic content block as an OpenAI content part; null for the rest. */
function openAIPartOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI content part as an Anthropic content block; null for the rest. */
function anthropicBlockOf(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
    const url = part.image_url.url;
    const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
    if (data !== null) {
      return { type: 'image', source: { type: 'base64', media_type: data[1] === '' ? 'image/png' : data[1], data: data[2] } };
    }
    return { type: 'image', source: { type: 'url', url } };
  }
  return null;
}

/** Split a data: URL into its media type and base64 payload; null when not one. */
function dataUrlOf(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] === '' ? 'image/png' : match[1], data: match[2] };
}

// ---------------------------------------------------------------------------
// Anthropic client -> OpenAI upstream
// ---------------------------------------------------------------------------

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request naming
 * `model`: system prompt, images, tool definitions and results, tool choice,
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 * Anthropic has no response_format, so none is added.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = openAIPartOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') {
      out.tool_choice = { type: 'function', function: { name: choice.name } };
    }
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

// ---------------------------------------------------------------------------
// OpenAI client -> Anthropic upstream
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the system message becomes the system prompt, image_url parts
 * become image blocks, tool_calls become tool_use blocks, tool messages
 * become tool_result blocks, and the output cap, temperature, top_p, stop
 * sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    const content = raw.content;
    if (role === 'system') {
      const text = textOf(content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [
          {
            type: 'tool_result',
            tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
            content: typeof content === 'string' ? content : toolResultText(content),
          },
        ],
      });
      continue;
    }
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (Array.isArray(content)) {
      const blocks: Json[] = [];
      for (const part of content) {
        if (!isRecord(part)) continue;
        const block = anthropicBlockOf(part);
        if (block !== null) blocks.push(block);
      }
      if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
    }
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Gemini upstream: request builders
// ---------------------------------------------------------------------------

interface GeminiParts {
  systemText: string;
  contents: Json[];
  tools: Json[];
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
}

/** The shared Gemini body parts of an OpenAI or Anthropic client request. */
function geminiPartsOf(clientApi: 'openai' | 'anthropic', body: Json): GeminiParts {
  const systemParts: string[] = [];
  const contents: Json[] = [];
  const toolDecls: Json[] = [];
  // tool_call_id -> function name, for tool results that follow a tool call.
  const callNames = new Map<string, string>();

  const addTextImageParts = (content: unknown, parts: Json[]): void => {
    if (typeof content === 'string') {
      if (content !== '') parts.push({ text: content });
      return;
    }
    if (!Array.isArray(content)) return;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
      else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          parts.push({ fileData: { fileUri: source.url } });
        }
      } else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
        const url = block.image_url.url;
        const data = dataUrlOf(url);
        if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
        else parts.push({ fileData: { fileUri: url } });
      }
    }
  };

  if (clientApi === 'openai') {
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role;
      const content = raw.content;
      if (role === 'system') {
        const text = textOf(content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (role === 'assistant') {
        const parts: Json[] = [];
        addTextImageParts(content, parts);
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          const id = typeof call.id === 'string' ? call.id : '';
          const name = typeof call.function.name === 'string' ? call.function.name : '';
          if (id !== '' && name !== '') callNames.set(id, name);
          parts.push({ functionCall: { name, args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (role === 'tool') {
        const id = typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '';
        const name = callNames.get(id) ?? '';
        contents.push({
          role: 'user',
          parts: [{ functionResponse: { name, response: { result: typeof content === 'string' ? content : toolResultText(content) } } }],
        });
        continue;
      }
      const parts: Json[] = [];
      addTextImageParts(content, parts);
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        toolDecls.push({
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      }
    }
  } else {
    const system = textOf(body.system);
    if (system !== '') systemParts.push(system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              const id = typeof block.id === 'string' ? block.id : '';
              const name = typeof block.name === 'string' ? block.name : '';
              if (id !== '' && name !== '') callNames.set(id, name);
              parts.push({ functionCall: { name, args: isRecord(block.input) ? block.input : {} } });
            }
          }
        } else if (typeof content === 'string' && content !== '') {
          parts.push({ text: content });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      const parts: Json[] = [];
      if (Array.isArray(content)) {
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'tool_result') {
            const id = typeof block.tool_use_id === 'string' ? block.tool_use_id : '';
            const name = callNames.get(id) ?? '';
            parts.push({ functionResponse: { name, response: { result: toolResultText(block.content) } } });
          } else {
            addTextImageParts([block], parts);
          }
        }
      } else if (typeof content === 'string' && content !== '') {
        parts.push({ text: content });
      }
      if (parts.length > 0) contents.push({ role: 'user', parts });
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        toolDecls.push({
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }

  const stops =
    clientApi === 'openai'
      ? (Array.isArray(body.stop) ? body.stop.filter((s): s is string => typeof s === 'string') : [])
      : (Array.isArray(body.stop_sequences) ? body.stop_sequences.filter((s): s is string => typeof s === 'string') : []);
  const maxTokens =
    typeof body.max_tokens === 'number'
      ? body.max_tokens
      : typeof body.max_completion_tokens === 'number'
        ? body.max_completion_tokens
        : null;
  return {
    systemText: systemParts.join('\n'),
    contents,
    tools: toolDecls,
    maxTokens,
    temperature: typeof body.temperature === 'number' ? body.temperature : null,
    topP: typeof body.top_p === 'number' ? body.top_p : null,
    stops,
  };
}

/** An OpenAI or Anthropic client request as a Gemini generateContent body. */
export function clientToGemini(clientApi: 'openai' | 'anthropic', body: Json, jsonMode: boolean): Json {
  const parts = geminiPartsOf(clientApi, body);
  const out: Json = { contents: parts.contents };
  if (parts.systemText !== '') out.systemInstruction = { parts: [{ text: parts.systemText }] };
  if (parts.tools.length > 0) out.tools = [{ functionDeclarations: parts.tools }];
  const config: Json = {};
  if (parts.maxTokens !== null) config.maxOutputTokens = parts.maxTokens;
  if (parts.temperature !== null) config.temperature = parts.temperature;
  if (parts.topP !== null) config.topP = parts.topP;
  if (parts.stops.length > 0) config.stopSequences = parts.stops;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Upstream answers: parsing
// ---------------------------------------------------------------------------

/** One SSE frame's lines: its event name (null when unnamed) and its data. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split SSE text into frames, tolerating CRLF. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

/** An OpenAI Chat Completions answer (JSON or SSE) as a normalised answer. */
export function parseOpenAIAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value)) return null;
    const choices = Array.isArray(value.choices) ? value.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : {};
    const message = isRecord(choice.message) ? choice.message : {};
    const text = typeof message.content === 'string' ? message.content : textOf(message.content);
    const toolCalls: UpstreamAnswer['toolCalls'] = [];
    const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call) || !isRecord(call.function)) continue;
      toolCalls.push({
        id: typeof call.id === 'string' ? call.id : '',
        name: typeof call.function.name === 'string' ? call.function.name : '',
        arguments: argumentsOf(call.function.arguments),
      });
    }
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text,
      inputTokens: numberOrZero(usage.prompt_tokens),
      outputTokens: numberOrZero(usage.completion_tokens),
      stopFamily: stopFamilyOf(choice.finish_reason),
      toolCalls,
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  const toolCalls: UpstreamAnswer['toolCalls'] = [];
  const toolArgs = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(bodyText)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    if (choices.length === 0) continue;
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : 0;
      const slot = toolArgs.get(index) ?? { id: '', name: '', args: '' };
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof call.id === 'string' && call.id !== '') slot.id = call.id;
      if (typeof fn.name === 'string' && fn.name !== '') slot.name = fn.name;
      if (typeof fn.arguments === 'string') slot.args += fn.arguments;
      toolArgs.set(index, slot);
    }
    if (typeof choice.finish_reason === 'string') stopFamily = stopFamilyOf(choice.finish_reason);
  }
  for (const slot of toolArgs.values()) {
    toolCalls.push({ id: slot.id, name: slot.name, arguments: argumentsOf(slot.args) });
  }
  return { text, inputTokens, outputTokens, stopFamily, toolCalls };
}

/** An Anthropic Messages answer (JSON or SSE) as a normalised answer. */
export function parseAnthropicAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value)) return null;
    const text = textOf(value.content);
    const toolCalls: UpstreamAnswer['toolCalls'] = [];
    if (Array.isArray(value.content)) {
      for (const block of value.content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          arguments: isRecord(block.input) ? block.input : {},
        });
      }
    }
    const usage = isRecord(value.usage) ? value.usage : {};
    return {
      text,
      inputTokens: numberOrZero(usage.input_tokens),
      outputTokens: numberOrZero(usage.output_tokens),
      stopFamily: stopFamilyOf(value.stop_reason),
      toolCalls,
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  const toolCalls: UpstreamAnswer['toolCalls'] = [];
  const toolArgs = new Map<number, { id: string; name: string; args: string }>();
  for (const frame of parseSseFrames(bodyText)) {
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) {
          inputTokens = numberOrZero(data.message.usage.input_tokens);
        }
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string') text += delta.text;
        else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && index >= 0) {
          const slot = toolArgs.get(index) ?? { id: '', name: '', args: '' };
          slot.args += delta.partial_json;
          toolArgs.set(index, slot);
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0) {
          toolArgs.set(index, {
            id: typeof block.id === 'string' ? block.id : '',
            name: typeof block.name === 'string' ? block.name : '',
            args: '',
          });
        }
        break;
      }
      case 'message_delta': {
        const delta = isRecord(data.delta) ? data.delta : {};
        if (typeof delta.stop_reason === 'string') stopFamily = stopFamilyOf(delta.stop_reason);
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      default:
        break;
    }
  }
  for (const slot of toolArgs.values()) {
    toolCalls.push({ id: slot.id, name: slot.name, arguments: argumentsOf(slot.args) });
  }
  return { text, inputTokens, outputTokens, stopFamily, toolCalls };
}

/** A Gemini generateContent answer (JSON or SSE) as a normalised answer. */
export function parseGeminiAnswer(bodyText: string, stream: boolean): UpstreamAnswer | null {
  const geminiTextOf = (value: unknown): string => {
    if (!isRecord(value)) return '';
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    return parts
      .flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : []))
      .join('');
  };
  if (!stream) {
    const value = jsonOrNull(bodyText);
    if (!isRecord(value)) return null;
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
    const toolCalls: UpstreamAnswer['toolCalls'] = [];
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const part of parts) {
      if (!isRecord(part) || !isRecord(part.functionCall)) continue;
      const call = part.functionCall;
      toolCalls.push({
        id: '',
        name: typeof call.name === 'string' ? call.name : '',
        arguments: isRecord(call.args) ? call.args : {},
      });
    }
    return {
      text: geminiTextOf(value),
      inputTokens: numberOrZero(usage.promptTokenCount),
      outputTokens: numberOrZero(usage.candidatesTokenCount),
      stopFamily: stopFamilyOf(candidate.finishReason),
      toolCalls,
    };
  }
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  for (const frame of parseSseFrames(bodyText)) {
    const value = jsonOrNull(frame.data);
    if (!isRecord(value)) continue;
    text += geminiTextOf(value);
    const candidates = Array.isArray(value.candidates) ? value.candidates : [];
    const candidate = isRecord(candidates[0]) ? candidates[0] : {};
    if (typeof candidate.finishReason === 'string') stopFamily = stopFamilyOf(candidate.finishReason);
    if (isRecord(value.usageMetadata)) {
      inputTokens = numberOrZero(value.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  return { text, inputTokens, outputTokens, stopFamily, toolCalls: [] };
}

// ---------------------------------------------------------------------------
// Client answers: emitting
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

function splitText(text: string, size = 13): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/** A normalised answer as an OpenAI Chat Completions JSON object. */
export function answerToOpenAIJson(answer: UpstreamAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((call) => ({
      id: call.id !== '' ? call.id : 'call_unknown',
      type: 'function',
      function: { name: call.name, arguments: JSON.stringify(call.arguments) },
    }));
  }
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: 1767225600,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinishReason(answer.stopFamily), logprobs: null }],
    usage: {
      prompt_tokens: answer.inputTokens,
      completion_tokens: answer.outputTokens,
      total_tokens: answer.inputTokens + answer.outputTokens,
    },
  };
}

/** A normalised answer as an Anthropic Messages JSON object. */
export function answerToAnthropicJson(answer: UpstreamAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    content.push({ type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: call.arguments });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicStopReason(answer.stopFamily),
    stop_sequence: null,
    usage: { input_tokens: answer.inputTokens, output_tokens: answer.outputTokens },
  };
}

/** A normalised answer as an OpenAI SSE stream, ending with [DONE]. */
export function answerToOpenAIStream(answer: UpstreamAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: 1767225600,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent(
        null,
        chunk({ tool_calls: [{ index: 0, id: call.id !== '' ? call.id : 'call_unknown', type: 'function', function: { name: call.name, arguments: JSON.stringify(call.arguments) } }] }, null),
      ),
    );
  }
  frames.push(sseEvent(null, chunk({}, openAIFinishReason(answer.stopFamily))));
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: 1767225600,
      model,
      choices: [],
      usage: {
        prompt_tokens: answer.inputTokens,
        completion_tokens: answer.outputTokens,
        total_tokens: answer.inputTokens + answer.outputTokens,
      },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A normalised answer as an Anthropic SSE stream, in the required event order. */
export function answerToAnthropicStream(answer: UpstreamAnswer, model: string): string {
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: {
        id: 'msg_aggregate',
        type: 'message',
        role: 'assistant',
        model,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: answer.inputTokens, output_tokens: 0 },
      },
    }),
  ];
  let index = 0;
  if (answer.text !== '') {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', {
        type: 'content_block_start',
        index,
        content_block: { type: 'tool_use', id: call.id !== '' ? call.id : 'toolu_unknown', name: call.name, input: {} },
      }),
    );
    frames.push(
      sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index,
        delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.arguments) },
      }),
    );
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', {
      type: 'message_delta',
      delta: { stop_reason: anthropicStopReason(answer.stopFamily), stop_sequence: null },
      usage: { output_tokens: answer.outputTokens },
    }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** An Anthropic model list: the free models this entry serves. */
export function anthropicModelList(models: readonly { id: string; display_name: string }[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: '2026-01-01T00:00:00Z' })),
    has_more: false,
  };
}

/** An OpenAI model list: the free models this entry serves. */
export function openAIModelList(models: readonly { id: string; provider: string }[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: 1767225600, owned_by: m.provider })),
  };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}
