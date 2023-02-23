from fastapi import FastAPI

from app import getUserProfile
from recommend import Recommender

app = FastAPI()


@app.get("/recommend/")
async def read_item(userID: str):
    rec = Recommender(target=userID)
    await rec.recommend()
    # account = await getUserProfile(userID)
    return {"user": userID}
