import whisper
from fastapi import FastAPI, UploadFile, File
import uvicorn
import tempfile

# Load Whisper model (tiny/base/small/medium/large)
model = whisper.load_model("base")

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        result = model.transcribe(tmp.name)
    return {"text": result["text"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
