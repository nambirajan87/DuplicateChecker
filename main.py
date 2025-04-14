from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List
from fastapi.responses import StreamingResponse
import pandas as pd
import io

app = FastAPI()

# Load the pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

class SentenceInput(BaseModel):
    sentences: List[str]

@app.post("/meaningful_match_json")
async def meaningful_match(request: SentenceInput):
    sentences = request.sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Create readable response
    results = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):  # avoid repeating and self-comparisons
            results.append({
                "sentence_1": sentences[i],
                "sentence_2": sentences[j],
                "similarity_score": round(cosine_scores[i][j].item(), 3)
            })

    return {"comparisons": results}

@app.post("/meaningful_match_csv")
async def meaningful_match_csv(request: SentenceInput):
    sentences = request.sentences
    embeddings = model.encode(sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)

    # Prepare pairwise similarity results
    results = []
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            results.append({
                "sentence_1": sentences[i],
                "sentence_2": sentences[j],
                "similarity_score": round(cosine_scores[i][j].item(), 3)
            })

    # Convert to CSV in memory
    df = pd.DataFrame(results)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)

    # Return CSV as downloadable file
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=meaningful_similarity.csv"}
    )
