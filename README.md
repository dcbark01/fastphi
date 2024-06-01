# Phi-3 LLM Inference Server

Simple REST API interface for the Microsoft Phi-3 large language models (LLMs).
Served using FastAPI, with inputs/outputs mirroring same interface
as used by the Huggingface Text Generation Inference Server.

## Install

Recommend Python 3.10 or above. Run:

```bash
pip install -r requirements.txt
```

## Start the Inference Server

To start the server, run:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Usage

To send an inference request:

```bash
curl -X POST "http://0.0.0.0:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "<|user|>\nHow old is the universe? <|end|>\n<|assistant|>",
        "parameters": {
            "best_of": 1,
            "decoder_input_details": false,
            "details": true,
            "do_sample": true,
            "frequency_penalty": 0.0,
            "grammar": null,
            "max_new_tokens": 20,
            "repetition_penalty": 1.0,
            "return_full_text": false,
            "seed": null,
            "stop": [
                "\n\n"
            ],
            "temperature": 0.2,
            "top_k": 10,
            "top_n_tokens": 5,
            "top_p": 0.95,
            "truncate": null,
            "typical_p": 0.95,
            "watermark": true
        }
    }'
```

Which should give you a response back like: 

```json
{"generated_text": "The universe is approximately 13.8 billion years old."}
```

You can also check the server status by sending a GET request to `/health`:

```bash
curl "http://0.0.0.0:8000/health"
```