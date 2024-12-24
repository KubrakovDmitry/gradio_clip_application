"""Программа проверки соответствия описания изображению."""
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import torch


def predict(image, texts, threshold=0.6):
    """Функция классификации изображений."""
    inputs = processor(text=texts,
                       images=image,
                       return_tensors='pt',
                       padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    prob = probs[0, 0].item()
    if prob > threshold:
        result = f"Положительный ответ (соответствие: {prob:.2f})"
    else:
        result = f"Отрицательный ответ (соответствие: {prob:.2f})"
    return result


if __name__ == '__main__':
    model_name = 'openai/clip-vit-base-patch32'
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    with gr.Blocks() as image2text:
        gr.Markdown(('### Проверте соответствие изображения'
                     ' текстовому описанию'))
        with gr.Row():
            image_input = gr.Image(label='Загрузите изображение',
                                   type='pil')
            text_input = gr.Textbox(label=('Введите описание изображения'))
        output = gr.Textbox(label='Результат')
        button = gr.Button('Проверить')

        def process(image, text):
            """Функция непосредственного сопоставления изображения и текста."""
            texts = [text, 'Ничего общего']
            return predict(image, texts)

        button.click(process, inputs=[image_input, text_input], outputs=output)
    image2text.launch(server_name="0.0.0.0", server_port=7860)
