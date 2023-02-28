from fastapi import FastAPI

from app import getUserProfile
from recommend import Recommender, Recs

app = FastAPI()


@app.get("/recommend/")
async def read_item(userID: str):
    # rec = Recommender(target=userID)
    # await rec.recommend()
    # account = await getUserProfile(userID)

    rec = Recs()
    res = rec.run()

    print(res.json())

    return {"user": userID, "data": res}
