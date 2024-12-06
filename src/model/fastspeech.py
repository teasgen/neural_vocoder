import torch


class FastSpeechWrapper():
    def __init__(self):
        tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp32')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
        tacotron2 = tacotron2.to('cuda')
        tacotron2.eval()
        self.model = tacotron2
        self.max_segement_len = 150

    def __call__(self, text):
        spectrograms = []
        print(text.split(", "))
        for cur_text in text.split(", "):
            sequence, seq_length = self.utils.prepare_input_sequence([cur_text + " "])
            total_length = seq_length.item()
            last_index = 0
            sequences = []
            lengths = []

            while total_length - last_index >= self.max_segement_len:
                cur_max_seq = sequence[..., last_index: last_index + self.max_segement_len]
                r_index = cur_max_seq.shape[-1] - 1 - cur_max_seq.squeeze().tolist()[-2::-1].index(11)

                sequences.append(cur_max_seq[..., :r_index])
                lengths.append(torch.tensor([sequences[-1].shape[-1]]).to(seq_length.device))

                last_index = r_index

            sequences.append(sequence[..., last_index:])
            lengths.append(torch.tensor([total_length - last_index]).to(seq_length.device))

            for seq, cur_len in zip(sequences, lengths):
                print(seq, cur_len)
                with torch.no_grad():
                    spectrogram, _, _ = self.model.infer(seq, cur_len)
                spectrograms.append(spectrogram)
            # print(sequences, lengths)
            # with torch.no_grad():
            #     spectrogram, _, _ = self.model.infer(sequences, lengths)
            # spectrograms.append(spectrogram)

        spectrograms = torch.cat(spectrograms, -1)
        return spectrograms