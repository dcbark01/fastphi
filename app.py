from typing import List, Optional

import torch
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Response, status
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Model setup
model_id = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map=device,
    torch_dtype="auto", 
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# Pydantic models for the API inputs
class GenerateParameters(BaseModel):
    best_of: Optional[int] = None
    decoder_input_details: bool = False
    details: bool = True
    do_sample: bool = True
    frequency_penalty: Optional[float] = None
    grammar: Optional[str] = None
    max_new_tokens: int
    repetition_penalty: Optional[float] = None
    return_full_text: bool = False
    seed: Optional[int] = None
    stop: List[str]
    temperature: float
    top_k: Optional[int] = None
    top_n_tokens: Optional[int] = None
    top_p: Optional[float] = None
    truncate: Optional[int] = None
    typical_p: Optional[float] = None
    watermark: bool

class GenerateRequest(BaseModel):
    inputs: str
    parameters: GenerateParameters

class GenerateResponse(BaseModel):
    generated_text: str

app = FastAPI()

@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    # Prepare arguments based on request
    generation_args = {
        "max_new_tokens": request.parameters.max_new_tokens,
        "return_full_text": request.parameters.return_full_text,
        "temperature": request.parameters.temperature,
    }
    
    if request.parameters.do_sample:
        generation_args.update({
            "do_sample": request.parameters.do_sample,
            "top_k": request.parameters.top_k,
        })

    if request.parameters.seed:
        torch.manual_seed(request.parameters.seed)
    
    try:
        # Generate text from the model
        output = pipe(request.inputs, **generation_args)
        generated_text = output[0]['generated_text']
        return GenerateResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

class ModelInfo(BaseModel):
    model_id: str
    model_device_type: str


@app.get("/info", response_model=ModelInfo)
def get_info():
    return ModelInfo(
        model_id=model_id,
        model_device_type=str(device.type),
    )
    

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    # Simple health check to confirm model is operational
    try:
        test_output = pipe("Hello", max_length=5, return_full_text=False)
        return Response(status_code=status.HTTP_200_OK, content="Everything is working fine")
    except Exception as e:
        return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content="Text generation inference is down")
    

if __name__ == "__main__":
    """ Run FastAPI inference server that mirrors HF TGI interface.

    Example inference request:
    
    curl -X 'POST' "http://127.0.0.1:8000/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "inputs": "<|user|>\nHow old is the universe? <|end|>\n<|assistant|>",
        "parameters": {
            "best_of": 1,
            "decoder_input_details": false,
            "details": true,
            "do_sample": true,
            "frequency_penalty": 0.0,
            "grammar": null,
            "max_new_tokens": 20,
            "repetition_penalty": 1.0,
            "return_full_text": false,
            "seed": null,
            "stop": [
                "\n\n"
            ],
            "temperature": 0.5,
            "top_k": 10,
            "top_n_tokens": 5,
            "top_p": 0.95,
            "truncate": null,
            "typical_p": 0.95,
            "watermark": true
        }
    }'

    """
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
