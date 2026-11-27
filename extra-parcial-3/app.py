import os
import time
import gradio as gr
import spaces
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

# En un Space con ZeroGPU, Hugging Face define SPACES_ZERO_GPU: el modelo va a la GPU.
# En tu computadora esa variable no existe: el modelo se queda en la CPU.
DISPOSITIVO = "cuda" if os.getenv("SPACES_ZERO_GPU") else "cpu"

CLASIFICADOR_ID = "google/mobilenet_v2_1.0_224"
TRADUCTOR_ID = "Helsinki-NLP/opus-mt-en-es"

clasificador = pipeline(
    "image-classification",
    model=CLASIFICADOR_ID,
    device=DISPOSITIVO,
)

tokenizador = AutoTokenizer.from_pretrained(TRADUCTOR_ID)
traductor = AutoModelForSeq2SeqLM.from_pretrained(TRADUCTOR_ID).to(DISPOSITIVO)


def traducir_en_es(texto: str) -> str:
    """Traduce una etiqueta corta del inglés al español usando Opus-MT."""
    entradas = tokenizador(texto, return_tensors="pt").to(DISPOSITIVO)
    with torch.no_grad():
        salida = traductor.generate(**entradas, max_new_tokens=60)
    return tokenizador.decode(salida[0], skip_special_tokens=True)


@spaces.GPU
def clasificar(imagen):
    """Clasifica la imagen con MobileNetV2 y traduce la etiqueta principal al español."""
    if imagen is None:
        return {}, "", "", "0.00 s"

    inicio = time.time()
    resultados = clasificador(imagen, top_k=3)
    etiquetas = {r["label"]: r["score"] for r in resultados}

    etiqueta_en = resultados[0]["label"]
    etiqueta_es = traducir_en_es(etiqueta_en)
    segundos = time.time() - inicio

    return etiquetas, etiqueta_en, etiqueta_es, f"{segundos:.2f} s"


demo = gr.Interface(
    fn=clasificar,
    inputs=gr.Image(type="pil", label="Tu foto"),
    outputs=[
        gr.Label(num_top_classes=3, label="Lo que ve el modelo (Inglés)"),
        gr.Textbox(label="Etiqueta más probable (Inglés)"),
        gr.Textbox(label="Etiqueta más probable (Español)"),
        gr.Textbox(label="Tiempo total en CPU"),
    ],
    title="¿Qué hay en la foto? (Inglés + Español)",
    description=(
        "Clasificación con MobileNetV2 (`google/mobilenet_v2_1.0_224`) y traducción automática "
        "al español con `Helsinki-NLP/opus-mt-en-es`."
    ),
    examples=["ejemplos/carro.jpg", "ejemplos/deportivo.jpg"],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
