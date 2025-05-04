from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize FastAPI
app = FastAPI()

# Serve static files (like the frontend HTML)
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS (useful if embedding in Wix or calling from browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Phi-3-mini (ensure you have internet and enough memory)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini-4k-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini-4k-instruct", trust_remote_code=True, torch_dtype=torch.float16)
model.eval()

# Serve the main page
@app.get("/", response_class=HTMLResponse)
def read_index():
    with open("static/index.html", "r") as f:
        return f.read()

# Analyze endpoint
@app.post("/analyze")
async def analyze_folder(folder_path: str = Form(...)):
    if not os.path.isdir(folder_path):
        return JSONResponse(content={"error": "Invalid path"}, status_code=400)

    files = list(Path(folder_path).rglob("*"))
    valid_files = [f for f in files if f.suffix.lower() in ['.csv', '.xlsx']]
    summaries = []

    for file in valid_files:
        try:
            if file.suffix.lower() == ".csv":
                df = pd.read_csv(file)
            elif file.suffix.lower() == ".xlsx":
                df = pd.read_excel(file)
            else:
                continue

            preview = df.head(3).to_string()
            prompt = f"Analyze these files {file.name}:\n{preview}\n and come up with business ideas for implementing data science, amachine learning, or artificial intelligence features using LLMs"
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=400)
            summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

            summaries.append({"file": str(file), "summary": summary})
        except Exception as e:
            summaries.append({"file": str(file), "summary": f"Error: {str(e)}"})

    return {"summaries": summaries}
