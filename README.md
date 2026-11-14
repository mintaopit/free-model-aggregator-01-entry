# free-model-aggregator-01-entry

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/mintaopit/free-model-aggregator-01-entry/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/mintaopit/free-model-aggregator-01-entry/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API — JSON and streaming — over ten simulated free-tier model providers. For each request it picks the best free model that meets the request's requirements, speaks that provider's own wire (OpenAI, Anthropic or Gemini), and fails over to the next best free model when a provider's budget is spent, it rate-limits, or it goes out. `node main.ts --list` prints the catalog of free models it found.

## Run it

```sh
node main.ts
```

## How it works

- `main.ts` — the server: the brief, the routing, the routes, the register and the poll.
- `lib/brief.ts` — the trial brief's documents, the catalog probe, the catalog.
- `lib/translate.ts` — the six client-wire × upstream-wire translations, in both directions.
- `lib/upstream.ts` — one inference call to a provider, in its own wire.

Routing ranks every free model that meets a request by the quality table's rating, then by how few calls its provider has taken so far, so the load spreads over same-rated providers instead of burning one provider's budget. A provider that refuses twice in a row is treated as spent and skipped for the rest of the trial. Every choice is a function of the brief, the requests and the providers' answers so far — no wall clock, no unseeded randomness — so a replay leaves the same state export.

---

<sub>Entered on [make.fun](https://make.fun/?ref=vs37sx56&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
