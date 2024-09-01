from fastapi import FastAPI,Request
from fastapi.responses import JSONResponse
import convertTextTocsv

#creating object of FastAPI class

app=FastAPI()

@app.get("/sendText")
def send_text(t1: str):
    output_filename='cipher.csv'
    convertTextTocsv.text_to_csv_file(t1,output_filename)
    
    response_data = {
        "message": "CSV file has been written"
    }
    return JSONResponse(content=response_data)


# uvicorn main:app --reload
