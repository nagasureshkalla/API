import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime

class PromptGenerator:
    def __init__(self, model_name="Qwen/Qwen3-0.6B", device="cuda:0", max_new_tokens=1024):
        ##############################################
        # ENVIRONMENT (MEMORY STABILITY)
        ##############################################
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        ##############################################
        # CONFIG
        ##############################################
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.start_time = datetime.now()

        ##############################################
        # TORCH OPTIMIZATIONS
        ##############################################
        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ##############################################
        # LOAD TOKENIZER & MODEL
        ##############################################
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            low_cpu_mem_usage=True,
            use_cache=True,
        )
        self.model.eval()

    def build_prompt(self, user_input):
        """
        Create a prompt string for the model based on user input.
        """
        prompt = (
            "Generate a prompt to edit the image provided "
            "with the user provided details. User provided details are: " + user_input +",Make sure Tokens must be limited to 60"
        )
        return prompt

    def generate_text(self, user_input, temperature=0.7, top_p=0.9):
        """
        Generate text from the model based on user input.
        """
        prompt = self.build_prompt(user_input)
        messages = [{"role": "user", "content": prompt}]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        model_inputs = self.tokenizer(text, return_tensors="pt", padding=False)
        model_inputs = {k: v.to(self.device) for k, v in model_inputs.items()}

        torch.cuda.reset_peak_memory_stats()

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][model_inputs["input_ids"].shape[-1]:].tolist()

        # Parse thinking content (</think> token id = 151668)
        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip()
        return content

    def print_stats(self):
        """
        Print GPU memory usage and elapsed time.
        """
        print("Max GPU memory used on {} (GB):".format(self.device),
              round(torch.cuda.max_memory_allocated(device=int(self.device[-1])) / (1024 ** 3), 2))
        print("Total time:", datetime.now() - self.start_time)


if __name__ == "__main__":
    generator = PromptGenerator()
    user_input = input("Enter prompt: ")
    output = generator.generate_text(user_input)
    print("Generated Prompt:", output)
    generator.print_stats()
