import time
import os
from pathlib import Path
import gradio as gr
import spaces
from transformers import pipeline


# En un Space con ZeroGPU, Hugging Face define SPACES_ZERO_GPU: el modelo va a la GPU.
# En tu computadora esa variable no existe: el modelo se queda en la CPU.
DISPOSITIVO = "cuda" if os.getenv("SPACES_ZERO_GPU") else "cpu"

clasificador = pipeline("image-classification", model="google/mobilenet_v2_1.0_224", 
device=DISPOSITIVO)

@spaces.GPU
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
    cache_examples=False,
)

demo.launch()