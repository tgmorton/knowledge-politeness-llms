#!/usr/bin/env python3
"""
Local vLLM Mock Server for M1 Mac Testing

Mimics vLLM's OpenAI-compatible API using transformers library.
Slower than vLLM but works on M1 Mac for testing scripts before K8s deployment.

Usage:
    python3 tests/local_vllm_mock.py --model google/gemma-2-2b-it --port 8000
"""

import argparse
import json
import logging
from typing import Dict, List, Optional
from flask import Flask, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model and tokenizer (loaded once at startup)
model = None
tokenizer = None
model_name = None


def load_model(model_path: str):
    """Load model and tokenizer"""
    global model, tokenizer, model_name

    logger.info(f"Loading model: {model_path}")
    model_name = model_path

    # Use MPS (Metal Performance Shaders) on M1 Mac if available
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
        low_cpu_mem_usage=True,
    )

    logger.info("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "model": model_name}), 200


@app.route('/v1/completions', methods=['POST'])
def completions():
    """
    OpenAI-compatible completions endpoint

    Mimics vLLM's /v1/completions endpoint for testing
    """
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 1.0)
        logprobs = data.get('logprobs', None)

        logger.info(f"Received request: prompt_len={len(prompt)}, max_tokens={max_tokens}, temp={temperature}, logprobs={logprobs}")

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")

        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                return_dict_in_generate=True,
                output_scores=logprobs is not None,
            )

        # Decode
        generated_ids = outputs.sequences[0][inputs['input_ids'].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Prepare response
        response = {
            "id": "local-completion",
            "object": "text_completion",
            "created": 1234567890,
            "model": model_name,
            "choices": [
                {
                    "text": generated_text,
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": inputs['input_ids'].shape[1],
                "completion_tokens": len(generated_ids),
                "total_tokens": inputs['input_ids'].shape[1] + len(generated_ids),
            }
        }

        # Add logprobs if requested
        if logprobs is not None and hasattr(outputs, 'scores'):
            # Get logprobs for first generated token
            scores = outputs.scores[0][0]  # First token, first batch item
            probs = torch.nn.functional.softmax(scores, dim=-1)

            # Get top-k tokens
            top_probs, top_indices = torch.topk(probs, k=min(logprobs, len(probs)))

            # Convert to dictionary format
            top_logprobs_dict = {
                tokenizer.decode([idx.item()]): np.log(prob.item())
                for prob, idx in zip(top_probs, top_indices)
            }

            response['choices'][0]['logprobs'] = {
                'top_logprobs': [top_logprobs_dict],
            }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error processing request: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500


def main():
    parser = argparse.ArgumentParser(description='Local vLLM Mock Server for M1 Mac')
    parser.add_argument(
        '--model',
        type=str,
        default='google/gemma-2-2b-it',
        help='Model to load (default: google/gemma-2-2b-it)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to run server on (default: 8000)'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to (default: 127.0.0.1)'
    )

    args = parser.parse_args()

    # Load model
    load_model(args.model)

    # Run server
    logger.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == '__main__':
    main()
