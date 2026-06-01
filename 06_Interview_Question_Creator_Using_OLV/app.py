from fastapi import FastAPI, Form, Request, Response, File, Depends, HTTPException, status
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.encoders import jsonable_encoder
import uvicorn
import os
import aiofiles
import json
import csv
from src.helper import llm_pipeline                                        # imports the updated llm_pipeline that returns (rag_chain, filtered_ques_list)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")      # serves static files (CSS, JS, uploaded PDFs, output CSVs)

templates = Jinja2Templates(directory="templates")                         # sets up Jinja2 template rendering for HTML pages


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

   

@app.post("/upload")
async def chat(request: Request, pdf_file: bytes = File(), filename: str = Form(...)):
    base_folder = "static/docs/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)                                              # creates the docs folder if it doesn't exist

    pdf_filename = os.path.join(base_folder, filename)                     # constructs full path for the uploaded PDF

    async with aiofiles.open(pdf_filename, "wb") as f:
        await f.write(pdf_file)                                            # saves the uploaded PDF bytes to disk asynchronously

    response_data = jsonable_encoder(json.dumps({"msg": "success", "pdf_filename": pdf_filename}))
    return Response(response_data)                                         # returns the saved PDF path so frontend can display it


def get_csv(file_path):
    """
    Runs the full Q&A pipeline on the uploaded PDF and saves results to a CSV file.
    Uses the updated llm_pipeline which returns (rag_chain, filtered_ques_list).
    """
    rag_chain, ques_list = llm_pipeline(file_path)                        # ✅ updated variable name: rag_chain (replaces answer_generation_chain)

    base_folder = "static/output/"
    if not os.path.isdir(base_folder):
        os.mkdir(base_folder)                                              # creates the output folder if it doesn't exist

    output_file = base_folder + "QA.csv"

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])                        # writes the CSV header row

        for question in ques_list:
            print("Question: ", question)
            answer = rag_chain.invoke(question)                            # ✅ replaces deprecated answer_generation_chain.run(question)
            print("Answer: ", answer)
            print("--------------------------------------------------\n\n")

            csv_writer.writerow([question, answer])                        # saves each Q&A pair as a row in the CSV

    return output_file                                                     # returns the path to the generated CSV file


@app.post("/analyze")
async def analyze(request: Request, pdf_filename: str = Form(...)):
    output_file = get_csv(pdf_filename)                                    # runs the Q&A pipeline and gets the output CSV path
    response_data = jsonable_encoder(json.dumps({"output_file": output_file}))
    return Response(response_data)                                         # returns the CSV path so frontend can show the download button


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)        # starts the FastAPI server on port 8080 with auto-reload