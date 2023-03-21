from fastapi import FastAPI
import json

from keys import Keys
from models.meta import MetaRecommender
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


@app.get("/meta/")
async def read_item(query: str):
    recs = MetaRecommender(path="input/modules.json", target=query)

    res = recs.run()

    print(res)

    return {"query": query, "data": res}
