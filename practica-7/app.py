import time
import gradio as gr
from transformers import pipeline

clasificador = pipeline(
    "image-classification",
    model="google/mobilenet_v2_1.0_224",
)

def clasificar(imagen):
    inicio = time.time()
    resultados = clasificador(imagen, top_k=3)
    segundos = time.time() - inicio
    etiquetas = {r["label"]: r["score"] for r in resultados}
    return etiquetas, f"{segundos:.2f} s"

demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Tu foto"),
    outputs=[
        gr.Label(num_top_classes=3, label="Lo que ve el modelo"),
        gr.Textbox(label="Tiempo en CPU"),
    ],
    title="¿Qué hay en la foto?",
    description="MobileNetV2 (14 MB) corriendo en mi computadora. Conoce 1,000 categorías de ImageNet, con nombres en inglés.",
    examples=["ejemplos/carro.jpg", "ejemplos/mi-foto.jpg"],
)

demo.launch()