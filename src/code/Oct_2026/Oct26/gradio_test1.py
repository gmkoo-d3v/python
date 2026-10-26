# from model import generate_image

import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("#Your Fancy Image Generator")
    with gr.Row(equal_height=True):
        textbox = gr.Textbox(lines=1, show_label=False)
        button = gr.Button("Generate", variant="primary")

demo.launch()