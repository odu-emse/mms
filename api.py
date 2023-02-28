from fastapi import FastAPI
import json
from recommend import Recommender, Recs

app = FastAPI()


@app.get("/recommend/")
async def read_item(userID: str):
    rec = Recs()
    res = await rec.run()

    cleaned_data = json.loads(res)

    return {"user": userID, "data": cleaned_data[0:50]}
