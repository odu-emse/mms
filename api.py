from fastapi import FastAPI
import json

from keys import Keys
from models.meta import MetaRecommender
from recommend import Recommender, Recs
from utils.fetch import Fetcher

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
async def read_item(query: str, size: int = 10):
    fetcher = Fetcher("http://emse.dev.joeldesante.com:4000/graphql")

    module_data = fetcher.getModuleData()

    conv_data = json.dumps(module_data["data"]["module"], indent=4)

    recs = MetaRecommender(path="input/modules.json", target=query, size=size)

    res = recs.run()

    return {
        "query": query,
        "data": res,
        "DB_module_data": json.loads(conv_data),
    }
