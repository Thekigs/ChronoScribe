#!/usr/bin/env python3
"""ChronoScribe FastAPI Server — OpenAI-Compatible API

Usage:
  python runners/server.py --port 8000
  curl http://localhost:8000/v1/models
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"chronoscribe","messages":[{"role":"user","content":"hello"}]}'
"""

import os, sys, time, uuid, re, json, logging
from contextlib import asynccontextmanager

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

import argparse
parser = argparse.ArgumentParser(description='ChronoScribe OpenAI-Compatible API Server')
parser.add_argument('--host', type=str, default='127.0.0.1', help='监听地址；如需局域网访问请显式传入 --host，并自行配置认证和反向代理')
parser.add_argument('--port', type=int, default=8000, help='监听端口')
parser.add_argument('--device', type=str, default='auto', choices=['auto','cuda','mps','cpu'])
parser.add_argument('--model-dir', type=str, default=os.path.join(BASE,'models','Qwen3.5-9B'))
parser.add_argument('--lora-dir', type=str, default=os.path.join(BASE,'models','checkpoint-7500'))
parser.add_argument('--c2-dir', type=str, default=os.path.join(BASE,'models','chronos-2'))
parser.add_argument('--no-c2', action='store_true', help='禁用Chronos2')
parser.add_argument('--max-tokens', type=int, default=512, help='Defaultmax_tokens')
parser.add_argument('--allow-origin', action='append', default=[], help='显式允许的浏览器 Origin；可重复传入，默认不启用跨域访问')
args = parser.parse_args()

# Keep command-line help independent of the local inference stack.
import numpy as np
import torch
from utils.numeric_claim_guard import enforce_numeric_claims

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any
import uvicorn

# Models load once during application startup; request latency excludes this
# one-time initialization cost.
model = None
tok = None
c2 = None
DEVICE = None

def load_models():
    global model, tok, c2, DEVICE
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if args.device == 'auto':
        if torch.cuda.is_available(): DEVICE = 'cuda'
        elif torch.backends.mps.is_available(): DEVICE = 'mps'
        else: DEVICE = 'cpu'
    else:
        DEVICE = args.device

    logging.info(f"Device: {DEVICE}")
    qb = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map='cpu')

    if os.path.isdir(args.lora_dir):
        model = PeftModel.from_pretrained(qb, args.lora_dir).to(DEVICE); model.eval()
        logging.info("LoRA adapter loaded")
    else:
        model = qb.to(DEVICE); model.eval()
        logging.info("Base model loaded (no LoRA)")

    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    if not args.no_c2 and os.path.isdir(args.c2_dir):
        from chronos import Chronos2Pipeline
        c2 = Chronos2Pipeline.from_pretrained(args.c2_dir)
        logging.info("Chronos2 loaded")

    logging.info("ChronoScribe API server ready")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield

app = FastAPI(title="ChronoScribe API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=args.allow_origin,
    allow_credentials=bool(args.allow_origin),
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI-compatible request and response schemas.

class Message(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]

class ChatCompletionRequest(BaseModel):
    model: str = "chronoscribe"
    messages: List[Message]
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    stream: bool = False

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Usage

class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "chronoscribe"

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# The service follows the same directive -> forecast -> evidence -> response
# order as the interactive path, while keeping the API response schema stable.

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
    m = re.search(r'\[([\d.,\s]+)\]', str(text))
    if m:
        nums = re.findall(r'[\d.]+', m.group(1))
        if len(nums) > 10: return [float(n) for n in nums]
    nums = re.findall(r'[\d.]+', str(text))
    if len(nums) > 10: return [float(n) for n in nums]
    return None

def build_c2_report(preds, pl):
    """Compute language-ready evidence directly from the specialist forecast."""
    arr = np.array(preds)
    trend = '上升' if arr[-1] > arr[0]*1.02 else ('下降' if arr[-1] < arr[0]*0.98 else '平稳')
    if len(arr) >= 4:
        vol = float(np.std(np.diff(arr))/(np.mean(np.abs(arr))+1e-8))
    else:
        vol = 0.0
    risk = '高' if vol > 0.3 else ('中' if vol > 0.1 else '低')
    preview = ', '.join(f'{v:.2f}' for v in arr[:min(8, len(arr))])
    return (f"Chronos2预测{pl}步:\n"
            f"  numerical sequence: [{preview}...]\n"
            f"  起点={arr[0]:.2f} 终点={arr[-1]:.2f} "
            f"最低={np.min(arr):.2f} 最高={np.max(arr):.2f}\n"
            f"  趋势={trend} risk level={risk}")

def build_c2_typed_evidence(preds, pl):
    """Machine-readable fields used by the semantic numeric guard."""
    arr = np.asarray(preds, dtype=float)
    if arr.size == 0:
        return {}
    return {"horizon": float(pl), "start": float(arr[0]), "end": float(arr[-1]),
            "min": float(np.min(arr)), "max": float(np.max(arr))}

def process_chat(messages, temperature=0.7, top_p=0.9, max_tokens=None):
    """Process a request through the ChronoScribe execution workflow."""
    if max_tokens is None: max_tokens = args.max_tokens

    # Extract last user message for routing
    user_text = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_text = str(msg.get("content", ""))
            break

    nums = extract_numbers(user_text)
    has_data = nums is not None

    # Route only after a numerical history has been recovered from the request.
    # Invalid directives fall back to the data-available path for robustness.
    c2_report = None
    c2_typed_evidence = {}
    if c2 and has_data:
        route_raw = qgen([{"role":"system","content":ROUTER_SYS},
                           {"role":"user","content":user_text}], 256)
        s1p = extract_json(route_raw) or {}
        use_c2 = s1p.get("use_chronos2", has_data)
        if use_c2:
            pl = min(s1p.get("prediction_length", 24) or 24, 96)
            cl = min(s1p.get("context_length", len(nums)) or len(nums), len(nums))
            ctx = np.array(nums[-cl:], dtype=np.float32).reshape(1, 1, -1)
            ct = torch.tensor(ctx)
            qlist, _ = c2.predict_quantiles(ct, prediction_length=pl)
            q = qlist[0]
            preds = q[0,:,q.shape[2]//2].tolist()
            c2_report = build_c2_report(preds, pl)
            c2_typed_evidence = build_c2_typed_evidence(preds, pl)

    # Add derived evidence to the final user turn before language generation.
    gen_msgs = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in ("system", "user", "assistant"):
            gen_msgs.append({"role": role, "content": content})

    if c2_report:
        # Inject C2 data into the last user message
        gen_msgs[-1]["content"] = user_text + f"\n\n[Chronos2forecast results]\n{c2_report}"

    # Ensure system prompt
    if not gen_msgs or gen_msgs[0]["role"] != "system":
        gen_msgs.insert(0, {"role": "system", "content": "你是一个专业的time series forecasting分析助手。回答简洁专业。"})

    # Generate
    text = tok.apply_chat_template(gen_msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors='pt').to(DEVICE)
    t0 = time.time()
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tok.eos_token_id,
            "do_sample": temperature > 0.0,
        }
        if temperature > 0.0:
            generation_kwargs.update({"temperature": temperature, "top_p": top_p})
        out = model.generate(**inputs, **generation_kwargs)
    reply = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean thinking artifacts
    for marker in ['**Output:**', '**Response**:', '</think>']:
        if marker in reply:
            reply = reply.split(marker)[-1].strip()
    if 'Thinking Process:' in reply:
        after = reply.split('Thinking Process:')[-1].strip()
        for marker in ['**Response**:', '**Output:**', '\n\n']:
            if marker in after:
                parts = after.split(marker)[-1].strip()
                if parts and len(parts) > 5:
                    after = parts
                    break
        reply = after if after else reply

    # Numerical claims are allowed only when they can be aligned to the
    # deterministic specialist evidence injected above.
    if c2_report:
        reply, claim_audit = enforce_numeric_claims(reply, c2_report, typed_evidence=c2_typed_evidence)
        if claim_audit.blocked_count:
            logging.warning("Blocked %d unsupported numerical claims", claim_audit.blocked_count)

    latency = round(time.time() - t0, 3)
    prompt_tokens = inputs['input_ids'].shape[1]
    completion_tokens = out.shape[1] - inputs['input_ids'].shape[1]

    return reply, prompt_tokens, completion_tokens, latency, c2_report is not None

# ═══ Endpoints ═══

@app.get("/health")
@app.get("/")
async def health():
    return {
        "status": "ok",
        "model": "chronoscribe",
        "device": DEVICE,
        "chronos2": c2 is not None,
    }

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[
        ModelInfo(id="chronoscribe", created=1728000000, owned_by="chronoscribe")
    ])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true is not implemented; send stream=false")
    try:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        reply, prompt_tok, comp_tok, latency, used_c2 = process_chat(
            msgs, req.temperature, req.top_p, req.max_tokens)
    except Exception as e:
        logging.error(f"Generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
        created=int(time.time()),
        model=req.model,
        choices=[ChatCompletionChoice(
            index=0,
            message=Message(role="assistant", content=reply),
            finish_reason="stop"
        )],
        usage=Usage(
            prompt_tokens=prompt_tok,
            completion_tokens=comp_tok,
            total_tokens=prompt_tok + comp_tok
        )
    )

# ═══ Main ═══

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    print(f"\n{'='*60}")
    print("ChronoScribe API Server")
    print(f"{'='*60}")
    print(f"Endpoint: http://{args.host}:{args.port}")
    print(f"Swagger:  http://{args.host}:{args.port}/docs")
    print(f"Device:   {args.device}")
    print(f"Chronos2: {'enabled' if not args.no_c2 else 'disabled'}")
    print(f"{'='*60}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
