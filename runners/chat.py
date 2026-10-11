#!/usr/bin/env python3
"""ChronoScribe interactive session with language-guided specialist use."""
import os, sys, re, json
import argparse

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

parser = argparse.ArgumentParser(description='ChronoScribe interactive session')
parser.add_argument('--device', type=str, default='auto', choices=['auto','cuda','mps','cpu'],
                    help='device选择 (Default: auto)')
parser.add_argument('--temperature', type=float, default=0.7, help='generation温度 (Default: 0.7)')
parser.add_argument('--top-p', type=float, default=0.9, help='nucleus sampling (Default: 0.9)')
parser.add_argument('--max-tokens', type=int, default=512, help='最大generationtoken数 (Default: 512)')
parser.add_argument('--model-dir', type=str, default=os.path.join(BASE,'models','Qwen3.5-9B'),
                    help='Qwen3.5模型path')
parser.add_argument('--lora-dir', type=str, default=os.path.join(BASE,'models','checkpoint-7500'),
                    help='LoRAadapterpath (设空String跳过LoRA)')
parser.add_argument('--c2-dir', type=str, default=os.path.join(BASE,'models','chronos-2'),
                    help='Chronos2模型path (设空String跳过Chronos2)')
parser.add_argument('--no-c2', action='store_true', help='禁用Chronos2，纯LLM模式')
args = parser.parse_args()

# Import model runtime packages only after parsing so `--help` works on a
# source-only checkout.
import numpy as np
import torch
from utils.numeric_claim_guard import enforce_numeric_claims

MODEL_DIR = args.model_dir
ADAPTER_DIR = args.lora_dir if args.lora_dir else None
CHRONOS2_DIR = None if args.no_c2 else (args.c2_dir if args.c2_dir else None)
TEMP = args.temperature
TOP_P = args.top_p
MAX_TOKENS = args.max_tokens

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Device selection
if args.device == 'auto':
    if torch.cuda.is_available(): DEVICE = 'cuda'
    elif torch.backends.mps.is_available(): DEVICE = 'mps'
    else: DEVICE = 'cpu'
else:
    DEVICE = args.device

# Check environment
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    if 'qwen3_5' not in CONFIG_MAPPING and 'qwen3_5' not in str(MODEL_DIR):
        pass  # allow custom model paths
except Exception: pass

use_lora = ADAPTER_DIR and os.path.isdir(ADAPTER_DIR)
label = "Qwen3.5+LoRA" if use_lora else "Qwen3.5 Base"
print(f"loading {label}...")
qb = AutoModelForCausalLM.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, device_map='cpu')
print(f"device: {DEVICE}")
if use_lora:
    model = PeftModel.from_pretrained(qb, ADAPTER_DIR).to(DEVICE); model.eval()
else:
    model = qb.to(DEVICE); model.eval()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
if tok.pad_token is None: tok.pad_token = tok.eos_token

# Load Chronos2
c2 = None
if CHRONOS2_DIR and os.path.isdir(CHRONOS2_DIR):
    from chronos import Chronos2Pipeline
    c2 = Chronos2Pipeline.from_pretrained(CHRONOS2_DIR)
    print(f"Chronos2 已loading")
elif not args.no_c2:
    print("⚠️ Chronos2 path无效，将usage纯LLM模式")

print(f"ChronoScribe ready (temperature={TEMP}, top_p={TOP_P}, max_tokens={MAX_TOKENS})")
print("input 'quit' 退出, 'clear' 清空conversation history\n")

# The controller emits an executable directive. The numerical specialist is
# invoked only for requests that both contain usable history and select it.
ROUTER_SYS = "只outputJSON。判断是否需要time series forecasting：\n有historical data+预测需求 → {\"use_chronos2\":true,\"prediction_length\":int,\"context_length\":int}\n普通对话/无数据 → {\"use_chronos2\":false,\"direct_reply\":\"简短直接回复\"}\n禁止outputJSON之外的任何Text。"

def qgen(msgs, mt=128):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=mt, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

def extract_json(text):
    if not text: return None
    try: return json.loads(text)
    except Exception: pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group())
        except Exception: pass
    return None

def extract_numbers(text):
    """Extract numerical sequence from user input. Prioritizes bracket-enclosed data."""
    # Try bracket-enclosed data first: [1.2, 3.4, ...]
    m = re.search(r'\[([\d.,\s]+)\]', text)
    if m:
        nums = re.findall(r'\d+\.?\d*|\.\d+', m.group(1))
        if len(nums) > 10:
            return [float(n) for n in nums]
    # Fallback: extract all numbers (excluding bare dots and trailing periods)
    nums = re.findall(r'\d+\.?\d*|\.\d+', text)
    if len(nums) > 10:
        return [float(n) for n in nums]
    return None

def build_c2_report(preds, pl):
    """Derive deterministic evidence for grounded language generation."""
    arr = np.array(preds)
    trend = '上升' if arr[-1] > arr[0] * 1.02 else ('下降' if arr[-1] < arr[0] * 0.98 else '平稳')
    if len(arr) >= 4:
        volatility = float(np.std(np.diff(arr)) / (np.mean(np.abs(arr)) + 1e-8))
        first_half = arr[:len(arr)//2]
        second_half = arr[len(arr)//2:]
        half_trend = '加速' if abs(second_half[-1]-second_half[0]) > abs(first_half[-1]-first_half[0]) else '减速'
    else:
        volatility = 0.0
        half_trend = '平稳'
    if volatility > 0.3: risk = '高'
    elif volatility > 0.1: risk = '中'
    else: risk = '低'
    peak_pos = int(np.argmax(arr))
    valley_pos = int(np.argmin(arr))
    preview = ', '.join(f'{v:.2f}' for v in arr[:min(8, len(arr))])
    return (f"Chronos2预测{pl}步:\n"
            f"  numerical sequence: [{preview}...]\n"
            f"  起点={arr[0]:.2f} 终点={arr[-1]:.2f} 最低={np.min(arr):.2f}(第{valley_pos+1}步) 最高={np.max(arr):.2f}(第{peak_pos+1}步)\n"
            f"  趋势={trend}({half_trend}中) 波动率={volatility:.2f} risk level={risk}")

history = [{"role": "system", "content": "你是一个专业的time series forecasting分析助手。回答简洁专业。"}]
while True:
    try:
        user_input = input("👤 你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break
    except UnicodeDecodeError:
        print("⚠️ inputEncodingError，请重试（terminal需支持UTF-8）")
        continue

    if not user_input: continue
    if user_input.lower() == 'quit': break
    if user_input.lower() == 'clear':
        history = [history[0]]; print("对话已清空\n"); continue

    # Extract embedded numerical history before invoking the controller. This
    # interactive convenience parser is separate from benchmark data loading.
    nums = extract_numbers(user_input)
    has_data = nums and len(nums) > 10

    route_raw = qgen([{"role":"system","content":ROUTER_SYS},
                       {"role":"user","content":user_input}], 256)
    s1p = extract_json(route_raw) or {}
    use_c2 = s1p.get("use_chronos2", has_data)

    c2_report = None
    if use_c2 and has_data:
        pl = min(s1p.get("prediction_length", 24) or 24, 96)
        cl = min(s1p.get("context_length", len(nums)) or len(nums), len(nums))
        ctx = np.array(nums[-cl:], dtype=np.float32).reshape(1, 1, -1)
        ct = torch.tensor(ctx)
        qlist, medlist = c2.predict_quantiles(ct, prediction_length=pl)
        q = qlist[0]; preds = q[0,:,q.shape[2]//2].tolist()
        c2_report = build_c2_report(preds, pl)

    # The final generation sees forecast-derived evidence, not a fresh request
    # to invent numerical values independently of the specialist.
    if c2_report:
        # Prediction: full history + C2 data (one-time, not saved)
        gen_messages = list(history)
        gen_messages.append({"role": "user", "content": user_input + f"\n\n[Chronos2forecast results]\n{c2_report}"})
    elif any(w in user_input for w in ['这个','它','那','这','这些','那些','刚才','上次']):
        # Follow-up question: keep full history
        gen_messages = list(history)
        gen_messages.append({"role": "user", "content": user_input})
    else:
        # Chat mode: clean slate — system + current message only, no old prediction bias
        gen_messages = history[:1] + [{"role": "user", "content": user_input}]
    history.append({"role": "user", "content": user_input})

    print("ChronoScribe: ", end="", flush=True)

    text = tok.apply_chat_template(gen_messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=True,
                            temperature=TEMP, top_p=TOP_P, pad_token_id=tok.eos_token_id)
    reply = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean thinking process
    for marker in ['**Output:**', '**Response**:', '</think>']:
        if marker in reply:
            reply = reply.split(marker)[-1].strip()
    if 'Thinking Process:' in reply:
        reply = reply.split('Thinking Process:')[-1].strip()
        # Try to extract the actual response after the thinking
        for marker in ['**Response**:', '**Output:**', '\n\n']:
            if marker in reply:
                after = reply.split(marker)[-1].strip()
                if after and len(after) > 5:
                    reply = after
                    break

    # Keep numerical narration grounded in the deterministic Chronos-2
    # evidence; unsupported values are masked before reaching the user.
    if c2_report:
        reply, claim_audit = enforce_numeric_claims(reply, c2_report)
        if claim_audit.blocked_count:
            print(f"[numeric-claim-guard] blocked {claim_audit.blocked_count} unsupported values")

    print(reply)
    history.append({"role": "assistant", "content": reply})
    print()
