import base64
import io

from ray import serve


@serve.deployment(num_replicas=1)
class Wav2Vec2Transcription:
    def __init__(self):
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    async def __call__(self, request):
        import torch
        import torchaudio

        content_type = request.headers.get("content-type", "")
        if "audio/wav" in content_type:
            audio_bytes = await request.body()
        else:
            data = await request.json()
            audio_bytes = base64.b64decode(data.get("audio", data.get("data")))

        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes), backend="ffmpeg")

        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        inputs = self.processor(
            waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return {"transcription": transcription.strip()}


app = Wav2Vec2Transcription.bind()
