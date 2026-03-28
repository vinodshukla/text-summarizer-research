import os
import gradio as gr
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# Optional tracking
import mlflow
import dagshub

REPO_OWNER = "vinodshukla"
REPO_NAME = "AI-Lab"

def init_tracking():
    try:
        dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME)
        mlflow.set_experiment("AI-Lab")
        print("✅ MLflow Tracking Active")
    except Exception as e:
        print(f"⚠️ Tracking skipped: {e}")

init_tracking()

MODEL_NAME = "./summarizer_model" if os.path.exists("./summarizer_model") else "t5-small"
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model.to("cpu")

def summarize(text, max_len, min_len, beam_size):
    try:
        with mlflow.start_run(run_name="Gradio-Inference", nested=True):
            mlflow.log_params({"max_len": max_len, "min_len": min_len, "beams": beam_size})
            inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
            outputs = model.generate(
                inputs["input_ids"],
                max_length=int(max_len),
                min_length=int(min_len),
                num_beams=int(beam_size)
            )
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            mlflow.log_metric("summary_len", len(summary))
            return summary
    except Exception as e:
        # Fallback if MLflow logging fails
        print(f"⚠️ MLflow logging skipped: {e}")
        inputs = tokenizer("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(
            inputs["input_ids"],
            max_length=int(max_len),
            min_length=int(min_len),
            num_beams=int(beam_size)
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

demo = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=5, label="Input Text"),
        gr.Slider(20, 200, value=80, label="Max Length"),
        gr.Slider(10, 100, value=20, label="Min Length"),
        gr.Slider(1, 10, value=4, label="Beam Size")
    ],
    outputs="text",
    title="AI-Lab Summarizer"
)

def launch_app():
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch_app()