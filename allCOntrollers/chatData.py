from fastapi import FastAPI, HTTPException
import pandas as pd
import requests
import pdfplumber
import pandas as pd
import io
from io import BytesIO
from pymongo import MongoClient
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from fastapi import File, UploadFile
import PyPDF2
from bson import ObjectId
from transformers import pipeline 

client=MongoClient("mongodb+srv://vikas05:VikasKushwaha123@cluster0.w4oeelf.mongodb.net/troven?retryWrites=true&w=majority")
ChatDB=client['ChatSocketDB']


app = FastAPI()


class urlSchema(BaseModel):
    url:str

class chatSchema(BaseModel):
    chat_id:str
    question:str


@app.post("/process-url/")
def extract_text_from_pdf(pdfUrl: urlSchema):
    try:
        # Download the PDF file from the URL
        response = requests.get(pdfUrl.url)
        response.raise_for_status()
        
        # Open the PDF file using pdfplumber
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            text_data = []
            
            # Extract text from each page
            for page in pdf.pages:
                text = page.extract_text()
                text_data.append(text)
            
            # Combine all page text into a single string
            full_text = "\n".join(text_data)
            chatCOll=ChatDB['wellChat']
            insertData=chatCOll.insert_one({ "message": full_text})
        
        return {
            "chat_id":str( insertData.inserted_id),
            "message": "URL content processsed and stored successfully",
            "processed_method":'url'
        }
    
    except Exception as e:
        return {"error": str(e)}
    



@app.post("/process-pdf")
async def process_pdf_file(file: UploadFile = File(...)):
    # Read the uploaded file into a bytes stream
    pdf_bytes = await file.read()
    pdf_io = io.BytesIO(pdf_bytes)

    # Open the PDF file using PyPDF2
    reader = PyPDF2.PdfReader(pdf_io)
    text = ""

    # Extract text from each page
    for page in reader.pages:
        text += page.extract_text()

    chatCOll=ChatDB['wellChat']
    insertData=chatCOll.insert_one({ "message": text})

    # Return the extracted text as a response
    return {
            "chat_id":str( insertData.inserted_id),
            "message": "PDF content processsed and stored successfully",
            "processed_method":'pdf'
        }


# Load pre-trained embedding model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


summarizer = pipeline("summarization")

# Endpoint to process user query
@app.post("/chat")
def chat_endpoint(query: chatSchema):
    # Convert the query to embedding
    query_embedding = model.encode(query.question, convert_to_tensor=True)
    chatColl=ChatDB['wellChat']
    # Retrieve all stored content (assumed to be stored in MongoDB)
    stored_content = list(chatColl.find({'_id':ObjectId(query.chat_id)}, {"_id": 0, "message": 1}))
    # print(stored_content)
    # # Compute embeddings for the stored content
    stored_embeddings = []
    for content in stored_content:
        content_embedding = model.encode(content['message'], convert_to_tensor=True)
        stored_embeddings.append((content['message'], content_embedding))

    # Compare the query embedding with the stored embeddings using cosine similarity
    max_similarity = -1
    most_relevant_message = ""
    
    for message, embedding in stored_embeddings:
        similarity = util.pytorch_cos_sim(query_embedding, embedding).item()
        if similarity > max_similarity:
            max_similarity = similarity
            most_relevant_message = message

    if len(most_relevant_message.split()) > 50:  # Example threshold for summarization
        summary = summarizer(most_relevant_message, max_length=50, min_length=25, do_sample=False)
        response = summary[0]['summary_text']
    else:
        response = most_relevant_message

    # Return the most relevant message
    return {"response": response, "similarity_score": max_similarity}


