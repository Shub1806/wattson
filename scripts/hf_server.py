"""Minimal OpenAI-compatible LLM server using Hugging Face transformers.

A deliberately simple baseline for Wattson: it runs one request at a time on
the GPU (no batching), in arrival order. Later we compare it against vLLM,
which batches many requests together.

Only standard library + torch + transformers, which Kaggle already has.

    MODEL=Qwen/Qwen2.5-0.5B-Instruct PORT=8000 python scripts/hf_server.py

Endpoints:
    GET  /health
    GET  /metrics               (queue length, vLLM metric names)
    POST /v1/chat/completions   (stream, max_tokens, stream_options.include_usage,
                                 ignore_eos for fixed-length benchmarks)
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PORT = int(os.environ.get("PORT", "8000"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# T4 has no bfloat16 support, so use float16 on GPU.
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"loading {MODEL} on {DEVICE} ({DTYPE}) ...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=DTYPE).to(DEVICE).eval()

# One request on the GPU at a time. Others wait here, which shows up as a
# longer time-to-first-token for them.
gpu_lock = threading.Lock()

# Queue counters, published at /metrics with the same names vLLM uses, so
# Wattson can tell waiting requests from running ones.
counts_lock = threading.Lock()
counts = {"waiting": 0, "running": 0}


class gpu_slot:
    """`with gpu_slot():` waits for the GPU and keeps the queue counters right."""

    def __enter__(self):
        with counts_lock:
            counts["waiting"] += 1
        gpu_lock.acquire()
        with counts_lock:
            counts["waiting"] -= 1
            counts["running"] += 1

    def __exit__(self, *exc):
        with counts_lock:
            counts["running"] -= 1
        gpu_lock.release()


def build_inputs(messages):
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = "\n".join(str(m.get("content", "")) for m in messages)
    return tokenizer(text, return_tensors="pt").to(DEVICE)


def generation_kwargs(inputs, body):
    max_new = int(body.get("max_tokens") or 128)
    kwargs = dict(**inputs, max_new_tokens=max_new, do_sample=False,
                  pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    if body.get("ignore_eos"):
        kwargs["min_new_tokens"] = max_new  # always generate exactly max_tokens
    return kwargs


def chunk(model_name, delta, finish=None, usage=None):
    event = {
        "id": "chatcmpl-hf",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [] if usage else [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage:
        event["usage"] = usage
    return event


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the notebook output quiet
        pass

    def do_GET(self):
        if self.path == "/health":
            self._send(200, b"ok", "text/plain")
        elif self.path == "/metrics":
            with counts_lock:
                text = (f"vllm:num_requests_waiting {counts['waiting']}\n"
                        f"vllm:num_requests_running {counts['running']}\n")
            self._send(200, text.encode(), "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            return self._send(404, b"not found", "text/plain")
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        if body.get("stream"):
            self._stream(body)
        else:
            self._complete(body)

    def _complete(self, body):
        inputs = build_inputs(body.get("messages", []))
        prompt_len = inputs["input_ids"].shape[1]
        with gpu_slot(), torch.no_grad():
            output = model.generate(**generation_kwargs(inputs, body))
        new_tokens = output[0][prompt_len:]
        response = {
            "id": "chatcmpl-hf",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": tokenizer.decode(new_tokens, skip_special_tokens=True)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": prompt_len, "completion_tokens": len(new_tokens),
                      "total_tokens": prompt_len + len(new_tokens)},
        }
        self._send(200, json.dumps(response).encode(), "application/json")

    def _stream(self, body):
        include_usage = (body.get("stream_options") or {}).get("include_usage", False)
        inputs = build_inputs(body.get("messages", []))
        prompt_len = inputs["input_ids"].shape[1]
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        result = {}

        def run():
            with torch.no_grad():
                result["output"] = model.generate(**generation_kwargs(inputs, body), streamer=streamer)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        try:
            with gpu_slot():
                worker = threading.Thread(target=run)
                worker.start()
                for text in streamer:
                    if text:
                        self._event(chunk(MODEL, {"content": text}))
                worker.join()
            completion = result["output"].shape[1] - prompt_len
            self._event(chunk(MODEL, {}, finish="stop"))
            if include_usage:
                self._event(chunk(MODEL, {}, usage={
                    "prompt_tokens": prompt_len, "completion_tokens": completion,
                    "total_tokens": prompt_len + completion}))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            worker.join()  # client left; let generation finish, then move on

    def _event(self, payload):
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
        self.wfile.flush()

    def _send(self, status, data, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"hf_server listening on http://127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
