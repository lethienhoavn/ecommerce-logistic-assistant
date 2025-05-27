import gradio as gr
from chatbot.handler import Handler

handler = Handler()

with gr.Blocks() as app:
    gr.HTML("<h1 style='text-align:center; color:#2c3e50;'>🤖 Ecom-Logistic Assistant</h1>")

    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, visible=False)
    with gr.Row():
        entry = gr.Textbox(label="Chat with our AI Assistant:")
    with gr.Row():
        clear = gr.Button("Clear", visible=False)

    def do_entry(message, history):
        history += [{"role":"user", "content":message}]
        return "", history

    entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
        handler.streaming, inputs=chatbot, outputs=[chatbot, image_output]
    )
    clear.click(lambda: (None, gr.update(visible=False)), inputs=None, outputs=chatbot, queue=False)

app.launch()