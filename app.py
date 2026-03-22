import os
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --- Configuration ---
# 1. Your specific repository details from the YAML
LOCAL_MODEL_PATH = "./summarizer_model" 
# Use your username and a model repo name (e.g., same as space or a separate model repo)
HF_MODEL_REPO = "vinodshukla1608/text-summarizer-research" 

# 2. Detect Environment
IS_SPACES = "SPACE_ID" in os.environ

if IS_SPACES:
    # On Hugging Face, it will pull from the Hub
    model_source = HF_MODEL_REPO
else:
    # Locally, it prioritizes your local folder if it exists
    if os.path.exists(LOCAL_MODEL_PATH):
        model_source = LOCAL_MODEL_PATH
    else:
        model_source = HF_MODEL_REPO

# --- Load Model & Tokenizer ---
tokenizer = T5Tokenizer.from_pretrained(model_source, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_source)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def summarize(text, max_len, min_len, beam_size):
    inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=int(max_len),
        min_length=int(min_len),
        num_beams=int(beam_size),
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Interface ---
demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(label="Article", placeholder="Paste text..."),
        gr.Slider(20, 200, value=80, step=5, label="Max Length"),
        gr.Slider(10, 100, value=20, step=5, label="Min Length"),
        gr.Slider(1, 10, value=4, step=1, label="Beam Size")
    ],
    outputs="text",
    title="Summarizer Demo"
)

def launch_app():
    demo.launch(inline=True, server_port=7860)

if __name__ == "__main__":
    demo.launch()