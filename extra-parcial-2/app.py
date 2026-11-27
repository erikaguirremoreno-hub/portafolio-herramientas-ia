import json
import gradio as gr
import pandas as pd


MODELOS_INFO = {
    "google/mobilenet_v2_1.0_224": {
        "tarea": "Clasificación de imágenes (1,000 clases)",
        "tamano": "14 MB",
        "velocidad": "Muy rápida en CPU (~0.05 s)",
        "uso": "Apps móviles o demos rápidas de visión",
    },
    "Helsinki-NLP/opus-mt-en-es": {
        "tarea": "Traducción automática Inglés → Español",
        "tamano": "~300 MB",
        "velocidad": "Rápida en CPU (< 0.5 s)",
        "uso": "Traducir etiquetas o frases cortas sin API externa",
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "tarea": "Generación de texto y chat",
        "tamano": "~6 GB",
        "velocidad": "Recomendado en GPU",
        "uso": "Asistentes conversacionales y resumen de documentos",
    },
}

MODOS_EJECUCION = {
    "CPU local": "Sin costo de GPU; ideal para modelos ligeros como MobileNetV2 o traductores pequeños.",
    "GPU en la nube (ZeroGPU)": "Aceleración por segundos bajo demanda; ideal para modelos medianos en Hugging Face Spaces.",
    "Modelo cuantizado (INT8 / 4-bit)": "Reduce el uso de memoria RAM/VRAM sacrificando muy poca precisión.",
}

RESPUESTAS_API = {
    "Clasificación de imagen": {
        "modelo": "google/mobilenet_v2_1.0_224",
        "tiempo_s": 0.04,
        "predicciones": [
            {"etiqueta": "sports car", "confianza": 0.84},
            {"etiqueta": "racer", "confianza": 0.04},
            {"etiqueta": "convertible", "confianza": 0.01},
        ],
    },
    "Traducción EN → ES": {
        "modelo": "Helsinki-NLP/opus-mt-en-es",
        "entrada": "sports car",
        "salida": "coche deportivo",
        "tiempo_s": 0.19,
    },
    "Extracción de entidades (NER)": {
        "modelo": "dslim/bert-base-NER",
        "texto": "MobileNetV2 corre en Gradio desde León.",
        "entidades": [
            {"texto": "MobileNetV2", "tipo": "MODELO"},
            {"texto": "Gradio", "tipo": "HERRAMIENTA"},
            {"texto": "León", "tipo": "LUGAR"},
        ],
    },
}

EJEMPLOS_NER = {
    "Proyecto de visión": [
        ("El modelo ", None),
        ("MobileNetV2", "MODELO"),
        (" clasifica fotos en ", None),
        ("Gradio 6", "FRAMEWORK"),
        (" usando ", None),
        ("PyTorch", "LIBRERIA"),
        (" en ", None),
        ("CPU", "HARDWARE"),
        (".", None),
    ],
    "Proyecto de traducción": [
        ("La red ", None),
        ("opus-mt-en-es", "MODELO"),
        (" de ", None),
        ("Helsinki-NLP", "AUTOR"),
        (" traduce etiquetas de ", None),
        ("ImageNet", "DATASET"),
        (" al ", None),
        ("español", "IDIOMA"),
        (".", None),
    ],
    "Evaluación de riesgo": [
        ("Advertencia: ", "ALERTA"),
        ("el término ", None),
        ("tiger cat", "ENTRADA"),
        (" se tradujo literalmente como ", None),
        ("tigre gato", "ERROR"),
        (".", None),
    ],
}

COLORES_NER = {
    "MODELO": "#2563eb",
    "FRAMEWORK": "#16a34a",
    "LIBRERIA": "#9333ea",
    "HARDWARE": "#ea580c",
    "AUTOR": "#0d9488",
    "DATASET": "#ca8a04",
    "IDIOMA": "#0284c7",
    "ALERTA": "#dc2626",
    "ENTRADA": "#4f46e5",
    "ERROR": "#e11d48",
}

TEXTOS_MARKDOWN = {
    "Resumen de arquitectura": (
        "### Pipeline de dos modelos\n"
        "- **Entrada:** Imagen RGB redimensionada a `224x224`.\n"
        "- **Paso 1:** `MobileNetV2` calcula las probabilidades sobre 1,000 clases.\n"
        "- **Paso 2:** `opus-mt-en-es` traduce la clase ganadora al español.\n\n"
        "> Consejo: usa `gr.Markdown` para mostrar respuestas limpias con listas, citas y código."
    ),
    "Tabla comparativa": (
        "### Comparativa rápida\n\n"
        "| Modelo | Parámetros | Latencia CPU |\n"
        "|---|---|---|\n"
        "| MobileNetV2 | 3.5 M | ~0.04 s |\n"
        "| Opus-MT EN-ES | 77 M | ~0.25 s |\n"
    ),
    "Buenas prácticas": (
        "### Checklist de entrega\n"
        "1. Verificar que `requirements.txt` incluya todas las dependencias.\n"
        "2. Confirmar que `.venv/` esté en `.gitignore`.\n"
        "3. Probar la app localmente antes de hacer `git push`."
    ),
}

DATOS_GRAFICA = {
    "Precisión Top-1 (%)": pd.DataFrame(
        {
            "Modelo": ["MobileNetV2", "ResNet-50", "EfficientNet-B0", "ViT-Base"],
            "Valor": [71.8, 76.1, 77.7, 81.1],
        }
    ),
    "Velocidad en CPU (imgs/s)": pd.DataFrame(
        {
            "Modelo": ["MobileNetV2", "ResNet-50", "EfficientNet-B0", "ViT-Base"],
            "Valor": [24.0, 6.5, 14.2, 2.8],
        }
    ),
}

TABLA_INICIAL = pd.DataFrame(
    {
        "Muestra": ["Foto 1", "Foto 2", "Foto 3"],
        "Etiqueta": ["sports car", "beetle", "espresso"],
        "Confianza (%)": [84.0, 91.5, 78.2],
    }
)

CODIGO_INICIAL = '''from transformers import pipeline

clasificador = pipeline("image-classification", model="google/mobilenet_v2_1.0_224")
resultado = clasificador("carro.jpg", top_k=3)
print(resultado)'''


def demo_slider(top_k: int, temperatura: float) -> str:
    estilo = "determinista y preciso" if temperatura < 0.4 else "equilibrado" if temperatura < 0.8 else "creativo y variado"
    return (
        f"### Configuración aplicada\n"
        f"- **Top k:** `{top_k}` candidatos mostrados.\n"
        f"- **Temperatura:** `{temperatura:.2f}` (comportamiento **{estilo}**)."
    )


def demo_dropdown(modelo: str) -> str:
    info = MODELOS_INFO[modelo]
    return (
        f"### Ficha del modelo `{modelo}`\n"
        f"- **Tarea:** {info['tarea']}\n"
        f"- **Tamaño aproximado:** {info['tamano']}\n"
        f"- **Velocidad:** {info['velocidad']}\n"
        f"- **Cuándo usarlo:** {info['uso']}"
    )


def demo_radio(modo: str) -> str:
    return f"### Modo elegido: {modo}\n{MODOS_EJECUCION[modo]}"


def demo_checkbox_group(pasos: list[str]) -> str:
    if not pasos:
        return "Selecciona al menos un paso de preprocesamiento para construir el pipeline."
    lista = "\n".join(f"{i}. `{paso}`" for i, paso in enumerate(pasos, start=1))
    return f"### Pipeline configurado ({len(pasos)} pasos activos)\n{lista}"


def demo_markdown(plantilla: str) -> str:
    return TEXTOS_MARKDOWN[plantilla]


def demo_json(ejemplo: str) -> dict:
    return RESPUESTAS_API[ejemplo]


def demo_highlighted(caso: str) -> list[tuple[str, str | None]]:
    return EJEMPLOS_NER[caso]


def demo_barplot(metrica: str) -> tuple[pd.DataFrame, str]:
    df = DATOS_GRAFICA[metrica]
    return df, f"Mostrando métrica: **{metrica}**"


def demo_dataframe(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "La tabla está vacía. Agrega al menos una fila."
    promedio = pd.to_numeric(df["Confianza (%)"], errors="coerce").dropna().mean()
    total = len(df)
    if pd.isna(promedio):
        return f"Filas registradas: **{total}**. Ingresa valores numéricos en `Confianza (%)`."
    return f"Filas registradas: **{total}** · Confianza promedio: **{promedio:.2f}%**"


def demo_code(codigo: str, lenguaje: str) -> str:
    lineas = len(codigo.splitlines()) if codigo else 0
    caracteres = len(codigo or "")
    return (
        f"### Análisis rápido del fragmento ({lenguaje})\n"
        f"- **Líneas:** `{lineas}`\n"
        f"- **Caracteres:** `{caracteres}`\n"
        f"- **Estado:** Listo para copiar o ejecutar."
    )


with gr.Blocks(title="Catálogo de 10 Componentes de Gradio 6") as demo:
    gr.Markdown(
        "# Catálogo Interactivo: 10 Componentes de Gradio 6\n"
        "Cada pestaña demuestra un componente distinto (sin usar `gr.Image`, `gr.Label` ni `gr.Textbox`), "
        "explicando si es de **Entrada**, **Salida** o **Entrada y salida** y en qué proyecto de IA se usa."
    )

    with gr.Tabs():
        with gr.Tab("1. gr.Slider"):
            gr.Markdown("**Tipo:** Entrada · **Uso en IA:** Ajustar hiperparámetros numéricos como `top_k` o `temperature`.")
            with gr.Row():
                slider_k = gr.Slider(1, 10, value=3, step=1, label="Top k")
                slider_temp = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperatura")
            out_slider = gr.Markdown(demo_slider(3, 0.7))
            slider_k.change(demo_slider, [slider_k, slider_temp], out_slider)
            slider_temp.change(demo_slider, [slider_k, slider_temp], out_slider)

        with gr.Tab("2. gr.Dropdown"):
            gr.Markdown("**Tipo:** Entrada · **Uso en IA:** Elegir un modelo preentrenado, idioma o dataset de una lista.")
            dropdown_modelo = gr.Dropdown(
                choices=list(MODELOS_INFO.keys()),
                value="google/mobilenet_v2_1.0_224",
                label="Selecciona un modelo",
            )
            out_dropdown = gr.Markdown(demo_dropdown("google/mobilenet_v2_1.0_224"))
            dropdown_modelo.change(demo_dropdown, dropdown_modelo, out_dropdown)

        with gr.Tab("3. gr.Radio"):
            gr.Markdown("**Tipo:** Entrada · **Uso en IA:** Seleccionar una única estrategia o entorno de ejecución.")
            radio_modo = gr.Radio(
                choices=list(MODOS_EJECUCION.keys()),
                value="CPU local",
                label="Entorno de inferencia",
            )
            out_radio = gr.Markdown(demo_radio("CPU local"))
            radio_modo.change(demo_radio, radio_modo, out_radio)

        with gr.Tab("4. gr.CheckboxGroup"):
            gr.Markdown("**Tipo:** Entrada · **Uso en IA:** Activar múltiples pasos de preprocesamiento o métricas.")
            checks_pasos = gr.CheckboxGroup(
                choices=["Redimensionar 224x224", "Normalizar píxeles", "Aumento de datos", "Evaluar F1-Score"],
                value=["Redimensionar 224x224", "Normalizar píxeles"],
                label="Pasos del pipeline",
            )
            out_checks = gr.Markdown(demo_checkbox_group(["Redimensionar 224x224", "Normalizar píxeles"]))
            checks_pasos.change(demo_checkbox_group, checks_pasos, out_checks)

        with gr.Tab("5. gr.Markdown"):
            gr.Markdown("**Tipo:** Salida · **Uso en IA:** Mostrar respuestas formateadas de un LLM, tablas o reportes.")
            selector_md = gr.Radio(
                choices=list(TEXTOS_MARKDOWN.keys()),
                value="Resumen de arquitectura",
                label="Vista de ejemplo",
            )
            out_md = gr.Markdown(TEXTOS_MARKDOWN["Resumen de arquitectura"])
            selector_md.change(demo_markdown, selector_md, out_md)

        with gr.Tab("6. gr.JSON"):
            gr.Markdown("**Tipo:** Salida · **Uso en IA:** Inspeccionar la respuesta estructurada de una API o modelo.")
            selector_json = gr.Dropdown(
                choices=list(RESPUESTAS_API.keys()),
                value="Clasificación de imagen",
                label="Payload de ejemplo",
            )
            out_json = gr.JSON(value=RESPUESTAS_API["Clasificación de imagen"], label="Respuesta JSON")
            selector_json.change(demo_json, selector_json, out_json)

        with gr.Tab("7. gr.HighlightedText"):
            gr.Markdown("**Tipo:** Salida · **Uso en IA:** Resaltar entidades nombradas (NER), tokens o errores en texto.")
            selector_ner = gr.Radio(
                choices=list(EJEMPLOS_NER.keys()),
                value="Proyecto de visión",
                label="Oración analizada",
            )
            out_ner = gr.HighlightedText(
                value=EJEMPLOS_NER["Proyecto de visión"],
                color_map=COLORES_NER,
                show_legend=True,
                label="Entidades detectadas",
            )
            selector_ner.change(demo_highlighted, selector_ner, out_ner)

        with gr.Tab("8. gr.BarPlot"):
            gr.Markdown("**Tipo:** Salida · **Uso en IA:** Comparar precisión, latencia o probabilidades por clase.")
            selector_metrica = gr.Radio(
                choices=list(DATOS_GRAFICA.keys()),
                value="Precisión Top-1 (%)",
                label="Métrica a graficar",
            )
            out_plot = gr.BarPlot(
                value=DATOS_GRAFICA["Precisión Top-1 (%)"],
                x="Modelo",
                y="Valor",
                title="Comparativa de modelos de visión",
                y_title="Valor",
                height=320,
            )
            out_plot_info = gr.Markdown("Mostrando métrica: **Precisión Top-1 (%)**")
            selector_metrica.change(demo_barplot, selector_metrica, [out_plot, out_plot_info])

        with gr.Tab("9. gr.Dataframe"):
            gr.Markdown("**Tipo:** Entrada y salida · **Uso en IA:** Editar lotes de prueba o visualizar métricas tabulares.")
            df_componente = gr.Dataframe(
                value=TABLA_INICIAL,
                headers=["Muestra", "Etiqueta", "Confianza (%)"],
                datatype=["str", "str", "number"],
                interactive=True,
                label="Resultados editables",
            )
            btn_df = gr.Button("Calcular resumen de la tabla")
            out_df = gr.Markdown(demo_dataframe(TABLA_INICIAL))
            btn_df.click(demo_dataframe, df_componente, out_df)

        with gr.Tab("10. gr.Code"):
            gr.Markdown("**Tipo:** Entrada y salida · **Uso en IA:** Mostrar o editar código generado por IA o snippets de API.")
            lenguaje_code = gr.Dropdown(
                choices=["python", "json", "markdown"],
                value="python",
                label="Lenguaje de resaltado",
            )
            code_componente = gr.Code(
                value=CODIGO_INICIAL,
                language="python",
                lines=6,
                interactive=True,
                label="Editor / Visor de código",
            )
            btn_code = gr.Button("Analizar fragmento")
            out_code = gr.Markdown(demo_code(CODIGO_INICIAL, "python"))
            lenguaje_code.change(lambda lang: gr.Code(language=lang), lenguaje_code, code_componente)
            btn_code.click(demo_code, [code_componente, lenguaje_code], out_code)


if __name__ == "__main__":
    demo.launch()
