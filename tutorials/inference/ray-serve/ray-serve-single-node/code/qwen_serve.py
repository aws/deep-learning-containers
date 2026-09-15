from ray import serve 
from transformers import AutoModelForImageTextToText, AutoProcessor 
import torch 
 
 
@serve.deployment(ray_actor_options={"num_gpus": 1}) 
class QwenVLService: 
    def __init__(self): 
        model_name = "Qwen/Qwen3-VL-2B-Instruct" 
        self.processor = AutoProcessor.from_pretrained(model_name) 
        self.model = AutoModelForImageTextToText.from_pretrained( 
        model_name, torch_dtype=torch.float16 
        ).to("cuda") 
 
    async def __call__(self, request): 
        body = await request.json() 
        image_url = body.get("image_url") 
        prompt = body.get("prompt") 
 
        messages = [ 
            { 
                "role": "user", 
                "content": [ 
                    {"type": "image", "image": image_url}, 
                    {"type": "text", "text": prompt}, 
                ], 
            } 
        ] 
 
        inputs = self.processor.apply_chat_template( 
            messages, tokenize=True, add_generation_prompt=True, 
            return_dict=True, return_tensors="pt" 
        ).to("cuda") 
 
        generated_ids = self.model.generate(**inputs, max_new_tokens=200) 
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)] 
        output = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0] 
        return {"response": output}  
 
app = QwenVLService.bind() 