import requests
import json
from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
import random
import sys
from dataclasses import dataclass


@dataclass
class Backend:
    url: str
    format: str

    def to_raw(self, qwen: str):
        if self.format in ["qwen", "raw"]:
            return qwen
        if self.format == "graphite":
            return qwen_to_graphite(qwen)
        raise ValueError(f"invalid format {self.format=}")

    def to_qwen(self, raw: str):
        if self.format in ["qwen", "raw"]:
            return raw
        if self.format == "graphite":
            return graphite_to_qwen(raw)
        raise ValueError(f"invalid format {self.format=}")


BACKENDS = [
    Backend("http://localhost:10000", "raw"),
    Backend("http://localhost:10001", "raw"),
    Backend("http://localhost:10002", "raw"),
]


def graphite_to_qwen(prompt):
    prompt = (
        prompt.replace(
            "<|start_of_role|>system<|end_of_role|>" f"<|im_start|>system\n",
        )
        .replace(
            "<|start_of_role|>user<|end_of_role|>" f"<|im_start|>user\n",
        )
        .replace(
            "<|start_of_role|>assistant<|end_of_role|>",
            f"<|im_start|>assistant\n",
        )
        .replace(
            f"<|end_of_text|>\n",
            f"<|im_end|>\n",
        )
    )

    return prompt


def qwen_to_graphite(prompt):
    for sfx in ["\n", ""]:
        prompt = (
            prompt.replace(
                f"<|im_start|>system{sfx}", "<|start_of_role|>system<|end_of_role|>"
            )
            .replace(f"<|im_start|>user{sfx}", "<|start_of_role|>user<|end_of_role|>")
            .replace(
                f"<|im_start|>assistant{sfx}",
                "<|start_of_role|>assistant<|end_of_role|>",
            )
            .replace(f"<|im_end|>{sfx}", f"<|end_of_text|>{sfx}")
        )

    return prompt


# --- Configuration ---
# ## NEW: Define multiple backend servers for blending ##
# The proxy will cycle through these servers for each generation chunk.

PROXY_HOST = "127.0.0.1"
PROXY_PORT = 11111

# --- Configuration for Chained Completion ---
DEFAULT_N_PREDICT = 100
GENERATION_CHUNK_SIZE = 15
END_TOKENS = ["</s>", "<|endoftext|>", "<|im_end|>"]


# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)


def generate_chained_completion(original_request_data):
    """
    This generator function handles the chained completion logic, now blending
    responses by routing requests to different backend servers in a round-robin fashion.
    """
    prompt = original_request_data.get("prompt", "")
    n_predict_req = original_request_data.get("n_predict", DEFAULT_N_PREDICT)

    if n_predict_req == -1 or "n_predict" not in original_request_data:
        print(
            f"Client requested n_predict=-1 or was missing. Proxy is overriding with default: {DEFAULT_N_PREDICT}"
        )
        n_predict_total = DEFAULT_N_PREDICT
    else:
        n_predict_total = n_predict_req

    print(f"--- Blended Completion Started ---")
    print(f"Original Prompt: '{prompt[:50]}...'")
    print(f"Total tokens to generate (proxy limit): {n_predict_total}")
    print(f"Chunk size per call: {GENERATION_CHUNK_SIZE}")

    full_prompt = prompt
    total_tokens_generated = 0
    model_stopped_naturally = False
    # ## NEW: Keep track of which request batch we are on to select a server ##
    request_batch_index = 0

    while total_tokens_generated < n_predict_total and not model_stopped_naturally:
        # Pick the next server from the list, cycling back to the start if needed.
        backend = random.choice(BACKENDS)
        current_backend_url = backend.url
        target_url = f"{current_backend_url}/completion"
        # --------------------------------------------------------

        remaining_tokens = n_predict_total - total_tokens_generated
        tokens_to_request = min(
            GENERATION_CHUNK_SIZE + random.randint(0, GENERATION_CHUNK_SIZE),
            remaining_tokens,
        )

        payload = original_request_data.copy()
        payload["prompt"] = backend.to_raw(full_prompt)
        payload["n_predict"] = tokens_to_request
        payload["stream"] = True

        print(
            f"\n> [Batch {request_batch_index}] Routing to {current_backend_url} for {tokens_to_request} tokens..."
        )

        try:
            response = requests.post(
                target_url,  # Use the dynamically selected URL
                headers={"Content-Type": "application/json"},
                json=payload,
                stream=True,
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to backend server {current_backend_url}: {e}")
            yield f"data: {json.dumps({'error': f'Failed to connect to backend: {current_backend_url}'})}\n\n"
            # Optional: You could implement logic here to skip the failing server and try the next one.
            # For now, we'll just stop the generation on failure.
            break

        chunk_generated_text = ""
        last_chunk_data = None
        content_chunks_in_batch = 0
        data_str = ""

        for line in response.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    yield line_str + "\n\n"
                    try:
                        data_str = line_str[len("data: ") :]
                        chunk_data = json.loads(data_str)
                        last_chunk_data = chunk_data
                        if "content" in chunk_data and chunk_data["content"]:
                            chunk_generated_text += chunk_data["content"]
                            content_chunks_in_batch += 1
                    except json.JSONDecodeError:
                        print(f"Warning: Could not decode JSON chunk: {data_str}")
                        continue

        if chunk_generated_text:
            print(f"< Received from {current_backend_url}: '{chunk_generated_text}'")
            full_prompt += chunk_generated_text
            total_tokens_generated += tokens_to_request

        # Smart Stop Detection Logic (Unchanged)
        if last_chunk_data and last_chunk_data.get("stop"):
            if content_chunks_in_batch < tokens_to_request:
                print("Backend stopped early. Assuming natural stop.")
                model_stopped_naturally = True
            else:
                clean_text = chunk_generated_text.strip()
                if any(clean_text.endswith(token) for token in END_TOKENS):
                    print(f"Backend stopped on a known end token: '{clean_text[-5:]}'")
                    model_stopped_naturally = True
                else:
                    print(
                        "Artificial stop detected (limit reached). Continuing generation..."
                    )

        # ## NEW: Increment the batch index for the next loop ##
        request_batch_index += 1

        if model_stopped_naturally:
            print("--- Chained Completion Finished (Model Stopped Naturally) ---")
            break

    if not model_stopped_naturally:
        print("--- Chained Completion Finished (n_predict reached) ---")


# --- The Core Proxy Route (Unchanged) ---
@app.route("/<path:subpath>", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
def proxy(subpath):
    # This function remains exactly the same as v4
    if subpath == "completion" and request.method == "POST":
        try:
            request_data = request.get_json()
            if not request_data:
                return Response("Invalid JSON", status=400)
        except Exception as e:
            return Response(f"Invalid JSON", status=400)
        if request_data.get("stream", False):
            return Response(
                stream_with_context(generate_chained_completion(request_data)),
                mimetype="text/event-stream",
            )
    print(f"--- Generic Proxy Passthrough for /{subpath} ---")
    try:
        resp = requests.request(
            method=request.method,
            url=f"{BACKENDS[0].url}/{subpath}",
            headers={k: v for k, v in request.headers if k.lower() != "host"},
            data=request.get_data(),
            stream=True,
            timeout=300,
        )
        headers = [
            (n, v)
            for n, v in resp.raw.headers.items()
            if n.lower()
            not in [
                "content-encoding",
                "content-length",
                "transfer-encoding",
                "connection",
            ]
        ]
        return Response(resp.iter_content(chunk_size=8192), resp.status_code, headers)
    except requests.exceptions.RequestException as e:
        return Response(f"Error connecting to llama.cpp server: {e}", status=502)


# --- Main Entry Point ---
if __name__ == "__main__":
    if not BACKENDS:
        print(
            "🚨 Error: The BACKEND_SERVERS list is empty. Please configure at least one server."
        )
    else:
        print(f"🦙 Llama in the Middle Proxy v5 (Blending Mode) starting...")
        print(f"Listening on http://{PROXY_HOST}:{PROXY_PORT}")
        print(f"Blending requests across: {', '.join([x.url for x in BACKENDS])}")
        app.run(host=PROXY_HOST, port=PROXY_PORT)
