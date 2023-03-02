import asyncio
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

    exit(0)


if __name__ == '__main__':
    asyncio.run(main())
