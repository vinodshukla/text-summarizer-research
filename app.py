import gradio as gr
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

demo = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Enter text"),
    outputs=gr.Textbox(label="Summary"),
    title="Text Summarizer Research"
)

if __name__ == "__main__":
    demo.launch()
