from fastapi import FastAPI

from app import getUserProfile

app = FastAPI()


@app.get("/recommend/")
async def read_item(userID: str):
    account = await getUserProfile(userID)
    print(account)
    return {"user": account}
