import os
import time

import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model.h5"
IMG_SIZE = (224, 224)
LOW_CONFIDENCE = 0.70

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(
        "model.h5 was not found. Upload model.h5 to the root of this Hugging Face Space."
    )

# compile=False avoids loading training-only optimizer/loss state.
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CATEGORIES = {
    "recyclable": {
        "title": "Recyclable Waste",
        "emoji": "♻️",
        "bin": "Blue bin",
        "color": "#3b82f6",
        "tips": [
            "Rinse off food or liquid residue.",
            "Keep recyclable items dry and uncontaminated.",
            "Flatten boxes and bottles to save space.",
        ],
    },
    "organic": {
        "title": "Organic Waste",
        "emoji": "🍎",
        "bin": "Green bin",
        "color": "#10b981",
        "tips": [
            "Remove plastic wrapping or stickers.",
            "Suitable for composting where local facilities accept it.",
            "Do not mix with recyclable waste.",
        ],
    },
}


def preprocess(image: Image.Image):
    # Match notebook preprocessing: RGB, 224x224, pixel values scaled to [0, 1].
    image = image.convert("RGB").resize(IMG_SIZE, resample=Image.Resampling.NEAREST)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)


def classify(image):
    if image is None:
        return "Upload an image to classify it.", {"Recyclable": 0.0, "Organic": 0.0}

    start = time.time()
    batch = preprocess(image)
    score = float(model.predict(batch, verbose=0).reshape(-1)[0])
    elapsed = time.time() - start

    # flow_from_directory assigns labels alphabetically:
    # O (Organic) = 0 and R (Recyclable) = 1.
    recyclable_probability = score
    organic_probability = 1.0 - score

    if recyclable_probability > 0.5:
        key = "recyclable"
        confidence = recyclable_probability
    else:
        key = "organic"
        confidence = organic_probability

    category = CATEGORIES[key]
    warning = (
        "\n\n⚠️ Low confidence: try a clearer photo with one item, good lighting "
        "and a plain background."
        if confidence < LOW_CONFIDENCE
        else ""
    )
    tips = "\n".join(f"• {tip}" for tip in category["tips"])

    result = (
        f"{category['emoji']} **{category['title']}**\n\n"
        f"**Suggested destination:** {category['bin']}\n\n"
        f"**Confidence:** {confidence:.1%}"
        f"{warning}\n\n"
        f"**Disposal tips**\n{tips}\n\n"
        f"_Inference time: {elapsed * 1000:.0f} ms_"
    )

    probabilities = {
        "Recyclable": recyclable_probability,
        "Organic": organic_probability,
    }
    return result, probabilities


with gr.Blocks(title="BinGENIUS – Waste Classifier", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🗑️ BinGENIUS Smart Classifier
        ### Upload or capture a photo to classify waste as recyclable or organic.
        """
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Select or capture a waste image",
                sources=["upload", "webcam"],
                type="pil",
            )
            classify_button = gr.Button("Identify Waste", variant="primary")
        with gr.Column():
            result = gr.Markdown("Your prediction will appear here.")
            probabilities = gr.Label(label="Class probabilities", num_top_classes=2)

    classify_button.click(
        fn=classify,
        inputs=input_image,
        outputs=[result, probabilities],
    )
    input_image.change(
        fn=classify,
        inputs=input_image,
        outputs=[result, probabilities],
    )

    gr.Markdown(
        "_This binary CNN model distinguishes recyclable and organic waste. "
        "Predictions are guidance only; follow your local waste-disposal rules._"
    )

if __name__ == "__main__":
    demo.launch()
