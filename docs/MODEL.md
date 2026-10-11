# ChronoScribe model and claim boundary

ChronoScribe is a system framework rather than a new monolithic forecasting model. It gives the language model responsibility for request interpretation, execution configuration, and response composition. A frozen time-series specialist remains the only source of numerical forecast values.

## Execution contract

The controller emits a schema-checked directive:

```json
{
  "use_chronos2": true,
  "prediction_length": 96,
  "context_length": 512
}
```

`use_chronos2` selects the reference specialist path, `prediction_length` records the requested horizon, and `context_length` selects the amount of past context. The paired benchmark supplies the executable target horizon to every system so that all models are scored on the same held-out window. The controller does not receive that window during execution.

The directive is an intermediate representation, not a forecast. The specialist receives history and horizon and returns point or quantile trajectories. The runner then constructs a deterministic evidence record containing values and summaries such as endpoints, range, direction, and interval information. The response model receives this evidence and may express it in natural language; it must not replace the trajectory with newly generated numbers.

```text
request → typed directive → specialist forecast → evidence record → response
```

## Directive fields

| Field | Meaning | Evaluation rule |
| --- | --- | --- |
| `use_chronos2` | Specialist or language-only route | Invalid forecastable directives are recorded and handled by the declared runner policy. |
| `prediction_length` | Controller-requested horizon | The benchmark horizon is authoritative for paired scoring. |
| `context_length` | Requested past-window length | The resolver builds a history-only context and records alignment. |

The reference field names are implementation names. An adapter for TimesFM, Sundial, Aurora, or another specialist must preserve the route, history, horizon, forecast, and evidence meanings even if its native API differs.

## Numerical provenance and guard

The specialist trajectory is the single numerical source. Evidence is computed deterministically from the returned trajectory. The numeric-claim guard checks response numbers against typed evidence and blocks unsupported numerical claims. A response can still contain an unsupported qualitative statement; the current guard audit does not prove that every nonnumeric semantic claim is grounded.

## Reference configuration

The reference controller is Qwen3.5-9B with a local LoRA adapter. The reference specialist is Chronos-2. TimesFM and Sundial are current specialist comparison systems, while Aurora is an additional strong boundary case. The adapter layer and controlled revision baseline live under `agent_baselines/`; they must not be confused with an official reproduction of an external agent.

## Adaptation data and reproducibility

The public repository contains a sanitized configuration and a CSV-to-SFT builder. It does not contain the private training corpus or the original serialized trainer state. The builder creates paired directive and evidence-conditioned response records and keeps future values out of controller inputs. End-to-end training reproduction requires provenance, licenses, record counts, split rules, prompt templates, LoRA target modules, and seeds to be released separately.

## Defensible claims

With the specialist fixed, the defensible mechanism statement is that language-generated context configuration changes forecasting behavior. Current strict paired results support an advantage over fixed-512 and only a limited aggregate advantage over fixed-1024. They do not establish universal superiority over full history or over every specialist. The retrospective context oracle is target-dependent and is an upper bound, not an online policy.
