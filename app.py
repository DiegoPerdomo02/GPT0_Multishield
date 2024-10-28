from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from GPT0_Model import main_app

# Crear la aplicación FastAPI
app = FastAPI(title="GPT0 Model API", version="1.0")

# Definir el esquema de entrada
class InputData(BaseModel):
    texto: str

# Ruta principal de la API para procesar texto
@app.post("/procesar/")
def procesar_texto(data: InputData):
    try:
        resultado = main_app(data.texto)
        return {"resultado": resultado}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Ruta de bienvenida
@app.get("/")
def home():
    return {"mensaje": "Bienvenido a la API de GPT0 Model"}
