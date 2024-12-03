from transformers import FastSpeech2ConformerTokenizer, FastSpeech2ConformerModel


class FastSpeechWrapper():
    def __init__(self):
        self.tokenizer = FastSpeech2ConformerTokenizer.from_pretrained("espnet/fastspeech2_conformer")
        self.model = FastSpeech2ConformerModel.from_pretrained("espnet/fastspeech2_conformer")

    def __call__(self, text):
        inputs = self.tokenizer(text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        output_dict = self.model(input_ids, return_dict=True)
        spectrogram = output_dict["spectrogram"].clamp_(min=1e-5).log_()
        return spectrogram.permute(0, 2, 1)  # -> (1, c, t_spec)