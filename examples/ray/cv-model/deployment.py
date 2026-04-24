import io

from PIL import Image
from ray import serve


@serve.deployment(
    autoscaling_config={"min_replicas": 1, "max_replicas": 2},
)
class DenseNetClassifier:
    def __init__(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms

        self.model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.class_names = models.DenseNet161_Weights.IMAGENET1K_V1.meta["categories"]

    async def __call__(self, request):
        import torch

        image_bytes = await request.body()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)

        top5_prob, top5_idx = torch.topk(probabilities, 5)
        predictions = [
            {
                "class_id": int(top5_idx[i]),
                "class_name": self.class_names[int(top5_idx[i])],
                "probability": float(top5_prob[i]),
            }
            for i in range(5)
        ]
        return {"predictions": predictions}


app = DenseNetClassifier.bind()
