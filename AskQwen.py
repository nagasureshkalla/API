import os
import time

import torch
from flask import Flask, jsonify, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer


app = Flask(__name__)


MODEL_NAME = os.getenv("ASK_QWEN_MODEL", "Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = int(os.getenv("ASK_QWEN_MAX_NEW_TOKENS", "2048"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class QwenChatbot:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading Qwen chatbot model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32,
        ).to(DEVICE)
        self.model.eval()
        self.history = []
        print("Qwen chatbot model loaded successfully!")

    def generate_response(self, user_input):
        messages = self.history + [{"role": "user", "content": user_input}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(DEVICE)
        input_length = inputs.input_ids.shape[-1]

        with torch.inference_mode():
            response_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
            )[0][input_length:]

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()

        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": response})

        return response

    def clear_history(self):
        self.history.clear()


chatbot = QwenChatbot()


def get_prompt_from_request():
    request_data = request.get_json(silent=True) if request.is_json else {}
    return (request.form.get("prompt") or request_data.get("prompt") or "").strip()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
@app.route("/api/ask", methods=["POST"])
def ask_qwen():
    try:
        print("Received Qwen prompt request")
        prompt_text = get_prompt_from_request()
        if not prompt_text:
            return jsonify({"message": "Error: prompt is required"}), 400

        start_time = time.time()
        response = chatbot.generate_response(prompt_text)
        total_time = time.time() - start_time

        return jsonify(
            {
                "message": "Generated response successfully!",
                "prompt": prompt_text,
                "response": response,
                "time_taken": f"{total_time:.2f} seconds",
            }
        )

    except Exception as e:
        return jsonify({"message": f"Error: {str(e)}"}), 500


@app.route("/api/reset", methods=["POST"])
def reset_chat():
    chatbot.clear_history()
    return jsonify({"message": "Chat history cleared successfully!"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8092)
