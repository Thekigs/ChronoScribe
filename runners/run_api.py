#!/usr/bin/env python3
"""OpenAI-compatible API baseline runner for the paired evaluation manifest.

The runner records canonical sample IDs, generation outputs, failures, and
timing for any compatible provider. It never exposes API baselines to the
ChronoScribe forecasting specialist.
"""

import argparse, json, os, re, sys, time
from collections import defaultdict

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(BASE, "benchmark")
OUT_BASE = os.path.join(BASE, "results", "api")
sys.path.insert(0, BASE)


def read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
    return rows


def load_samples(benchmark_dir, manifest_path, dims=None):
    """Load the fixed, paired evaluation set used by every benchmark runner."""
    from utils.evaluation_manifest import load_manifest_samples

    return load_manifest_samples(benchmark_dir, manifest_path, dims)


def load_full_suite(benchmark_dir, dims=None):
    """Load every benchmark example for suite inspection, not reported comparisons."""
    dim_samples = defaultdict(list)
    for dim in sorted(os.listdir(benchmark_dir)):
        if dims and dim not in dims:
            continue
        dim_dir = os.path.join(benchmark_dir, dim)
        if not os.path.isdir(dim_dir):
            continue
        for fn in sorted(os.listdir(dim_dir)):
            if fn.endswith(".jsonl"):
                dim_samples[dim].extend(read_jsonl(os.path.join(dim_dir, fn)))
    return dim_samples


def call_api(messages, api_key, model="deepseek-v4-flash", max_tokens=None, temperature=0.0, base_url="https://api.deepseek.com/v1"):
    """
    Call OpenAI-compatible Chat Completions API.
    Supports: DeepSeek, OpenAI, Anthropic, Groq, etc. — any provider with /v1/chat/completions endpoint.

    For reasoning models (deepseek-v4-pro, deepseek-reasoner, o1, etc):
      - Uses larger max_tokens to accommodate thinking tokens
      - Falls back to reasoning_content if content is empty

    Returns: (content_text, input_tokens, output_tokens, latency_s, error)
    """
    import urllib.request

    # All models: use generous max_tokens (8192) to avoid truncation
    if max_tokens is None:
        max_tokens = 8192

    url = f"{base_url.rstrip('/')}/chat/completions"

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ChronoScribe-Evaluation-Suite",
        },
    )

    t0 = time.time()
    data = None; last_err = ""
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            break
        except Exception as e:
            last_err = str(e)
            try:
                if hasattr(e, 'read'):
                    last_err += " | " + e.read().decode("utf-8", errors="replace")[:200]
            except Exception:
                pass
            if attempt < 2:
                time.sleep(1.5 ** attempt)  # 1s, 1.5s backoff
    if data is None:
        return None, 0, 0, time.time() - t0, last_err, "", ""

    latency = round(time.time() - t0, 3)

    content = ""
    reasoning_content = ""
    input_tokens = 0
    output_tokens = 0
    finish_reason = ""

    if "choices" in data and len(data["choices"]) > 0:
        choice = data["choices"][0]
        msg = choice.get("message", {})
        content = msg.get("content", "") or ""
        reasoning_content = msg.get("reasoning_content", "") or ""
        finish_reason = choice.get("finish_reason", "")

        # For any model: if content is empty, fall back to reasoning_content
        if not content and reasoning_content:
            # Try to extract JSON from the reasoning content
            content = reasoning_content

        if "usage" in data:
            usage = data["usage"]
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

    return content, input_tokens, output_tokens, latency, None, finish_reason, reasoning_content


def extract_json(text):
    """Extract JSON from text, with multiple fallbacks."""
    if not text:
        return None
    if isinstance(text, dict):
        return text
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    for m in re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", str(text), re.DOTALL):
        try:
            return json.loads(m)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    m = re.search(r"\{.*\}", str(text), re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def enhance_prompt_for_dim(sample, dim_name, model):
    """
    Enhance prompts for specific dimensions.
    For reasoning models: force direct JSON output, prohibit thinking.
    """
    messages = [dict(m) for m in sample.get("prompt", [])]

    # All models: force strict JSON-only output
    is_pro = True

    for msg in messages:
        if msg["role"] == "system":
            if is_pro:
                # Force Pro to skip reasoning entirely
                msg["content"] = (
                    "【严格要求】你是numerical forecastingsystem，不是聊天助手。"
                    "直接outputJSON结果。禁止output任何inference过程、分析、解释或description。"
                    "回复必须以 { 开头，以 } 结尾，中间不能有任何其他Text。"
                    "不要output thinking、不要分析原因、不要解释步骤——只要JSON。\n\n"
                    + msg["content"]
                )
            elif "先outputJSON" not in msg["content"]:
                msg["content"] = "【Important】先output完整的JSON结果，再进行分析。\n\n" + msg["content"]

    if dim_name == "D10_uncertainty":
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += (
                    "\n\noutputJSON需package含四个Array: predicted_values(P50中位数预测), "
                    "predicted_p10(10%分位数,偏保守), predicted_p50(50%分位数), "
                    "predicted_p90(90%分位数,偏乐观)。每个Arraylength与forecast horizon相同。"
                )

    if dim_name == "D12_causal_reasoning":
        for msg in messages:
            if msg["role"] == "system":
                msg["content"] += (
                    "\n\nJSON需package含: primary_cause(主要归因), secondary_cause(次要归因)。"
                    "Optional原因: holiday, weather, weekend, policy, fault, seasonal, trend_change, external_shock。"
                )

    return messages


def run_benchmark(api_key, model="deepseek-v4-flash", dims=None,
                  manifest_path=None, base_url="https://api.deepseek.com/v1"):
    """Run the full benchmark."""
    print("=" * 60)
    print(f"ChronoScribe paired evaluation: {model}")
    print("=" * 60)

    # Load samples
    print("Loading fixed evaluation manifest...")
    dim_samples = load_samples(BENCH, manifest_path, dims=dims)
    total = sum(len(v) for v in dim_samples.values())
    print(f"  {len(dim_samples)} dimensions, {total} samples")

    # Output directory
    model_slug = model.replace("/", "_").replace(":", "_")
    out_dir = os.path.join(OUT_BASE, model_slug)
    os.makedirs(out_dir, exist_ok=True)

    all_results = []
    t0 = time.time()

    for dim_name, samples in sorted(dim_samples.items()):
        dim_dir = os.path.join(out_dir, dim_name)
        os.makedirs(dim_dir, exist_ok=True)
        dim_results = []
        dt0 = time.time()

        print(f"\n[{dim_name}] {len(samples)} samples")

        for i, sample in enumerate(samples):
            gt = sample.get("ground_truth", {})
            r = {
                "id": sample["id"],
                "dimension": dim_name,
                "category": sample.get("category", ""),
                "dataset": sample.get("dataset", "?"),
                "domain": sample.get("domain", "?"),
            }

            try:
                # Enhance prompts for specific dimensions
                messages = enhance_prompt_for_dim(sample, dim_name, model)

                # Call API
                content, input_tokens, output_tokens, latency, error, finish_reason, reasoning_content = call_api(
                    messages, api_key, model=model, max_tokens=None, base_url=base_url
                )

                r["raw_output"] = content
                r["parsed_output"] = extract_json(content) if content else None
                r["input_tokens"] = input_tokens
                r["output_tokens"] = output_tokens
                r["total_tokens"] = input_tokens + output_tokens
                r["latency_s"] = latency
                r["gpu_memory_mb"] = None  # API call, no local GPU
                r["error"] = error

                # Log reasoning model diagnostics
                if finish_reason == "length":
                    r["_finish_reason"] = "length"
                if reasoning_content:
                    r["_reasoning_tokens"] = len(reasoning_content)
                    if not content or content == reasoning_content:
                        r["_used_reasoning_fallback"] = True

                # Extract predicted values from parsed JSON
                parsed = r["parsed_output"]
                if isinstance(parsed, dict):
                    pred = parsed.get("predicted_values", parsed.get("forecast", parsed.get("predictions")))
                    if isinstance(pred, list):
                        r["predicted_values"] = [float(v) for v in pred if isinstance(v, (int, float))]
                    analysis = parsed.get("analysis", parsed.get("explanation", ""))
                    if analysis:
                        r["analysis"] = str(analysis)

                    # Capture P10/P50/P90 when a response provides them.
                    if dim_name == "D10_uncertainty":
                        for qkey in ["predicted_p10", "predicted_p50", "predicted_p90"]:
                            qval = parsed.get(qkey)
                            if isinstance(qval, list):
                                r[qkey] = [float(v) for v in qval if isinstance(v, (int, float))]

                    # Capture available labels for the causal reasoning task.
                    if dim_name == "D12_causal_reasoning":
                        for ckey in ["primary_cause", "secondary_cause", "cause_analysis"]:
                            cval = parsed.get(ckey)
                            if cval:
                                r[ckey] = cval

                # Compute forecast metrics if we have predictions AND ground truth
                gt_values = gt.get("values", [])
                pred_values = r.get("predicted_values", [])
                if gt_values and pred_values and not error:
                    try:
                        from scoring.detailed_scoring import compute_all_forecast_metrics
                        r["forecast_metrics"] = compute_all_forecast_metrics(
                            np.array(gt_values), np.array(pred_values),
                            naive_forecast=gt.get("naive_forecast")
                        )
                    except Exception as e:
                        r["_metrics_error"] = str(e)[:100]

                status = "✅" if not error else "❌"
                tps = f"{output_tokens/latency:.0f}tok/s" if latency > 0 and output_tokens else ""
                smape = r.get("forecast_metrics", {}).get("sMAPE")
                sm_str = f" sMAPE={smape:.0f}%" if smape is not None else ""
                print(f"  [{i+1}/{len(samples)}] {status} {sample['id'][:40]}... {latency:.1f}s {tps}{sm_str}")

            except Exception as e:
                r["error"] = str(e)[:200]
                r["latency_s"] = round(time.time() - t0, 3)
                print(f"  [{i+1}/{len(samples)}] ❌ {str(e)[:80]}")

            dim_results.append(r)

        # Save per-dimension results
        with open(os.path.join(dim_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(dim_results, f, ensure_ascii=False, indent=2)

        all_results.extend(dim_results)
        ok = sum(1 for r in dim_results if not r.get("error"))
        print(f"  ✅ {dim_name}: {ok}/{len(dim_results)} | {time.time() - dt0:.0f}s")

    # Save all results
    elapsed = time.time() - t0
    ok = [r for r in all_results if not r.get("error")]
    fail = [r for r in all_results if r.get("error")]

    with open(os.path.join(out_dir, "all_results.jsonl"), "w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total_cost_est = sum(r.get("input_tokens", 0) * 0.27 / 1_000_000 for r in all_results)
    total_cost_est += sum(r.get("output_tokens", 0) * 1.10 / 1_000_000 for r in all_results)

    print(f"\n{'='*60}")
    print(f"ChronoScribe paired evaluation complete: {model}")
    print(f"{'='*60}")
    print(f"  Samples: {len(all_results)} | OK: {len(ok)} | Failed: {len(fail)}")
    print(f"  Total tokens: {sum(r.get('total_tokens', 0) for r in all_results):,}")
    print(f"  Total time: {elapsed/60:.1f}min")
    print(f"  Est. cost: ${total_cost_est:.4f}")
    print(f"  Output: {out_dir}")
    print(
        "\n  Score with: python3 scoring/score_results.py "
        f"--results-dir {out_dir} --benchmark-dir {BENCH} "
        f"--output-dir scores/{model_slug} --model-name {model_slug}"
    )


def main():
    parser = argparse.ArgumentParser(description="ChronoScribe API evaluation runner (OpenAI-compatible)")
    parser.add_argument("--model", default="deepseek-v4-flash", help="Model name")
    parser.add_argument("--dims", type=str, default="")
    parser.add_argument(
        "--manifest",
        default=os.path.join(BENCH, "evaluation_manifest.json"),
        help="Fixed evaluation manifest shared by every compared system.",
    )
    parser.add_argument("--api-key", type=str, default="", help="API key (or API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY env)")
    parser.add_argument("--api-base", type=str, default="https://api.deepseek.com/v1", help="API base URL")
    args = parser.parse_args()

    # Forecast metrics are only needed after the CLI is parsed, so `--help`
    # remains useful on a source-only checkout.
    global np
    import numpy as np

    api_key = args.api_key or os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("API key not set. Use --api-key or API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY env")
        print("Examples:")
        print("  DeepSeek:  DEEPSEEK_API_KEY=sk-... python run_api.py --model deepseek-v4-flash")
        print("  OpenAI:    OPENAI_API_KEY=sk-... python run_api.py --model gpt-4o --api-base https://api.openai.com/v1")
        sys.exit(1)

    dims = [d.strip() for d in args.dims.split(",") if d.strip()] if args.dims else None

    run_benchmark(
        api_key=api_key,
        model=args.model,
        dims=dims,
        manifest_path=args.manifest,
        base_url=args.api_base,
    )


if __name__ == "__main__":
    main()
