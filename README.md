# Free Model Aggregator

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/Jasonrichh/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/Jasonrichh/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

One server that serves the OpenAI Chat Completions API and the Anthropic Messages API over
ten simulated free-tier providers, routing each request to the best free model that can serve
it — an entry for the [free-model-aggregator-01 bounty](https://make.fun/bounty/free-model-aggregator-01)
on [make.fun](https://make.fun).

## What it does

- Reads the trial brief (provider directory, quality table, workload manifest) and probes
  each provider's catalog route in the provider's own wire (OpenAI, Anthropic, Gemini),
  building one catalog that `node main.ts --list` prints byte for byte as it registers it.
- Serves `POST /v1/chat/completions` and `POST /v1/messages`, in JSON and in server-sent
  events, plus `GET /v1/models` in the shape the auth header names (OpenAI for
  `Authorization: Bearer`, Anthropic for `x-api-key`).
- Routes every request (model `auto`) to the highest-rated free model meeting its
  requirements (tools, vision, JSON mode, long context), one candidate per provider, and
  fails over on 402/429: a refusal is counted against its provider, the request moves to
  the next candidate, a provider that fails twice in a row counts as spent, and a request
  whose candidates all failed once is tried again in the same order. A failure that carried
  a `retry-after` is never hammered with an immediate retry.
- Translates faithfully between wires: the system prompt, user text (images included),
  tools, stop sequences, `max_tokens` and JSON mode pass through untouched (a JSON-mode
  request from an OpenAI client never goes to an anthropic-wire provider, which cannot carry
  `response_format`), and the answer comes back with its text whole, its usage counts as
  that wire reports them, and its stop reason mapped.
- Deterministic: every choice derives from the brief, the requests and the providers'
  answers — no wall clock, no unseeded randomness — so a replay leaves the same state
  export byte for byte.

## Layout

- `main.ts` — the server: brief, catalog probe, routing, register, poll.
- `lib/providers.ts` — the brief's types and parser, auth headers, catalog probe.
- `lib/wire.ts` — Anthropic ↔ Chat Completions ↔ Gemini translation, JSON and SSE.
- `LICENSE` — MIT.

## Run

Node 22, standard library only, nothing to install:

```sh
node main.ts --list   # print the catalog and exit 0
node main.ts          # serve the trial (needs MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
```

## Local results (practice trials in the kit)

| trial | score | quality | served | wire | failover | catalog | calls |
|------:|------:|--------:|-------:|-----:|---------:|--------:|------:|
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 26 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 29 |

Quality 99 is the ceiling on these trials: the best model for 17 of the 20 requests is
one provider's, whose per-trial budget (10–12 calls) cannot cover them all, so some
requests must take the next-best free model. The state export is byte-identical across
repeated runs of the same trial.
