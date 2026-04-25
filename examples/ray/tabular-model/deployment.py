import json
import os

from ray import serve


@serve.deployment(num_replicas=1)
class IrisClassifier:
    def __init__(self):
        import torch
        import torch.nn as nn

        class IrisModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 16)
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 3)
                self.relu = nn.ReLU()

            def forward(self, x):
                return self.fc3(self.relu(self.fc2(self.relu(self.fc1(x)))))

        model_dir = "/opt/ml/model"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = IrisModel()
        self.model.load_state_dict(
            torch.load(os.path.join(model_dir, "iris_model.pth"), map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()

        with open(os.path.join(model_dir, "norm_params.json")) as f:
            norm = json.load(f)
        self.mean = torch.tensor(norm["mean"]).to(self.device)
        self.std = torch.tensor(norm["std"]).to(self.device)
        self.classes = norm["class_names"]

    async def __call__(self, request):
        import torch

        data = await request.json()
        features = data.get("features", data.get("data"))
        x = torch.tensor([features], dtype=torch.float32).to(self.device)
        x_norm = (x - self.mean) / self.std

        with torch.no_grad():
            probs = torch.softmax(self.model(x_norm), dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()

        return {
            "prediction": self.classes[pred_idx],
            "confidence": float(probs[0][pred_idx]),
            "probabilities": {cls: float(probs[0][i]) for i, cls in enumerate(self.classes)},
        }


app = IrisClassifier.bind()
