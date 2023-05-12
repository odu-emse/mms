from fastapi import FastAPI
import json

from keys import Keys
from recommend import Recommender, Recs

app = FastAPI()


@app.get("/recommend/")
async def read_item(userID: str):
    rec = Recs()
    res = await rec.run()

    cleaned_data = json.loads(res)

    return {"user": userID, "data": cleaned_data[0:50]}


@app.get("/keys/")
async def read_item(query: str):
    rec = Keys(target=query)

    cleaned_data = json.loads(rec.run())

    return {"query": query, "data": cleaned_data}
