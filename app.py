import asyncio
from prisma import Prisma

from utils.db import testConnection
from utils.seed import Seeder, Skipper

async def main() -> None:
    print('Starting application...')
    testConnection()

    db = Seeder(
        skip=[Skipper.all],
        cleanup=[Skipper.all],
        iterations=100,
    )

    await db.seedAll()
    await db.cleanupAll()
    # targetID = await db.createTargetUser()
    await getUserProfile(ID='63f7a3068b546b91eadb20a6')

    exit(0)


async def getUserProfile(ID):
    print('Fetching user data...')
    # get user from ID
    #  find enrolled modules
    #  remove modules that already enrolled in
    #  get feedback for each module
    #  get similarity matrix
    #  get recommendations
    #  return recommendations
    prisma = Prisma()
    await prisma.connect()
    account = await prisma.user.find_unique(
        where={
            'id': ID
        }
    )
    print(account)


if __name__ == '__main__':
    asyncio.run(main())
