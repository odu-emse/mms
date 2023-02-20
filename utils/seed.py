from enum import Enum
import os
from essential_generators import DocumentGenerator
from numpy import random
from prisma import Prisma
from requests import post
from utils.sdl import createMutationString, createModuleMutationString


async def seedUserDB():
    print('Seeding user model...')
    prisma = Prisma()
    await prisma.connect()
    gen = DocumentGenerator()
    iterations = 25
    accounts = []

    for i in range(iterations):
        accounts.append({
            'firstName': gen.name(),
            'lastName': gen.name(),
            'email': gen.email(),
            'openID': str(gen.integer()),
        })

    users = await prisma.user.create_many(data=accounts)

    print('User model seeded successfully with %d documents!' % users)
    await prisma.disconnect()


async def seedPlanOfStudyDB():
    print('Seeding plan of study model...')
    gen = DocumentGenerator()
    prisma = Prisma()
    await prisma.connect()

    modules = await prisma.module.find_many()
    accounts = await prisma.user.find_many()

    for account in accounts:
        for module in modules:
            # create plan of study for each module 25 times
            await prisma.planofstudy.create(
                data={
                    "modules": {
                        "connect": {
                            "id": module.get('id')
                        }
                    },
                    "student": {
                        "connect": {
                            "id": account['id']
                        }
                    }
                }
            )

    print('Plan of study model seeded successfully!')


# async def seedEnrollmentDB():
#     print('Seeding enrollment model...')
#     gen = DocumentGenerator()
#     prisma = Prisma()
#     await prisma.connect()
#     iterations = 25
#
#     modules = await prisma.module.find_many()
#
#     for account in accounts:
#         # create enrollment for each module 25 times
#         for module in modules:
#             # create enrollment for each module 25 times
#             await prisma.moduleenrollment.create(
#                 data={
#                     "role": gen.word(),
#                 }
#             )
#
#     print('Enrollment model seeded successfully!')


def seedDbFeedback():
    print('Seeding feedback model...')
    gen = DocumentGenerator()
    iterations = 25
    modules = getModules()['module']
    for module in modules:
        # create feedback for each module 25 times
        for i in range(iterations):
            res = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"),
                       {},
                       {'query': createMutationString(
                           comment=gen.sentence(),
                           rating=random.randint(1, 6),
                           moduleID=module['id'],
                       )
                       })
            print(res.json())

    print('Feedback model seeded successfully!')


def seedModuleModel():
    print('Seeding module model...')
    gen = DocumentGenerator()
    iterations = 25
    for i in range(iterations):
        key_length = random.randint(1, 11)
        res = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"),
                   {},
                   {'query': createModuleMutationString(
                       moduleName=gen.word(),
                       moduleNumber=random.randint(1, 1000),
                       description=gen.sentence(),
                       duration=random.randint(1, 100),
                       intro=gen.sentence(),
                       numSlides=random.randint(1, 100),
                       keywords=[gen.word() for i in range(key_length)]
                   )
                   })
        print(res.json())

    print('Module model seeded successfully!')


def getModules():
    mods = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"), {}, {
        'query': """query{
          module(input:{}){
            id
            moduleName
            moduleNumber
          }
        }"""
    })
    return mods.json()['data']


def getModuleFeedback():
    mods = post('http://%s/graphql' % os.environ.get("API_URL", "localhost:4000"), {}, {
        'query': """query{
          module(input:{}){
            id
            moduleName
            moduleNumber
            feedback{
              rating
              feedback
            }
            members{
              id
            }
          }
        }"""
    })
    return mods.json()['data']


class Skipper(Enum):
    user = 1
    module = 2
    feedback = 3
    plan = 4
    enrollment = 5
    none = 6


class Seeder:
    def __init__(self, skip=Skipper.none):
        self.gen = DocumentGenerator()
        self.prisma = Prisma()
        self.iterations = 25
        self.accounts = []
        self.modules = []
        self.enrollments = []
        self.feedbacks = []
        self.plans = []
        self.skip = skip

    async def seedUserDB(self):
        print('Seeding user model...')
        accounts = []
        await self.prisma.connect()

        for i in range(self.iterations):
            accounts.append({
                'firstName': self.gen.name(),
                'lastName': self.gen.name(),
                'email': self.gen.email(),
                'openID': str(self.gen.integer()),
            })

        users = await self.prisma.user.create_many(data=accounts)
        await self.prisma.disconnect()
        self.accounts = accounts
        print('User model seeded successfully with %d documents!' % users)

    def createKeywordList(self):
        key_length = random.randint(1, 11)
        return [self.gen.word() for i in range(key_length)]

    async def seedModuleDB(self):
        print('Seeding module model...')
        modules = []
        await self.prisma.connect()

        template = {
            'moduleName': 'word',
            'moduleNumber': {
                'typemap': 'small_int',
                'unique': True,
                'tries': 10
            },
            'description': 'sentence',
            'duration': 'small_int',
            'intro': 'sentence',
            'numSlides': 'small_int',
            'keywords': self.createKeywordList()
        }

        self.gen.set_template(template)
        docs = self.gen.documents(self.iterations)

        modules = await self.prisma.module.create_many(data=docs)
        await self.prisma.disconnect()
        self.modules = modules
        print('Module model seeded successfully with %d documents!' % modules)

    async def seedPlanOfStudyDB(self):
        print('Seeding plan of study model...')

    async def seedEnrollmentDB(self):
        print('Seeding enrollment model...')

    async def seedFeedbackDB(self):
        print('Seeding feedback model...')

    async def seedAll(self):
        if self.skip != self.skip.user:
            await self.seedUserDB()
        else:
            print('Skipping user model seeding...')

        if self.skip != self.skip.module:
            await self.seedModuleDB()
        else:
            print('Skipping module model seeding...')
        if self.skip != self.skip.feedback:
            await self.seedFeedbackDB()
        else:
            print('Skipping feedback model seeding...')
        if self.skip != self.skip.plan:
            await self.seedPlanOfStudyDB()
        else:
            print('Skipping plan of study model seeding...')
        if self.skip != self.skip.enrollment:
            await self.seedEnrollmentDB()
        else:
            print('Skipping enrollment model seeding...')

        print('All models seeded successfully!')
