from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import convertTextTocsv
import subprocess

#creating object of FastAPI class

app=FastAPI()

@app.get("/sendText")
def send_text(t1: str):
    output_filename='cipher.csv'
    convertTextTocsv.text_to_csv_file(t1,output_filename)
    result = subprocess.run(["python", "BEST.py"], capture_output=True, text=True)
    
    response_data = {
        "message": result
    }
    return JSONResponse(content=response_data)


# uvicorn main:app --reload
