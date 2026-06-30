"""Scene captioning with BLIP (pretrained, runs locally, no API)."""
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import config


class Captioner:
    def __init__(self, model_name=config.CAPTION_MODEL):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        self.model.to(self.device).eval()

    def caption(self, pil_image):
        """pil_image: PIL RGB image. Returns an English caption string."""
        inputs = self.processor(pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=40, num_beams=3)
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    import sys
    from PIL import Image
    print(Captioner().caption(Image.open(sys.argv[1]).convert("RGB")))
