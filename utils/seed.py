from enum import Enum
import os
from essential_generators import DocumentGenerator
from numpy import random
from prisma import Prisma
from prisma.models import User, PlanOfStudy
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
    all = 6


class Seeder:
    """
    Seeder class for seeding the database with dummy data. Makes use of the prisma python client to connection to the
    database and the essential_generators library to create document templates.
    """

    def __init__(self, skip: [Skipper] = None, cleanup: [Skipper] = None):
        self.gen = DocumentGenerator()
        self.prisma = Prisma()
        self.iterations = 25
        self.accounts = []
        self.modules = []
        self.enrollments = []
        self.feedbacks = []
        self.plans = []
        self.skip = skip
        self.cleanup = cleanup

    async def connect(self):
        await self.prisma.connect()

    async def disconnect(self):
        await self.prisma.disconnect()

    async def _getUserAccounts(self):
        """
        Gets all user accounts from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<User>
        """
        accounts = await self.prisma.user.find_many()

        acc = []

        for i in range(len(accounts)):
            acc.append(accounts[i].dict())

        return acc

    async def _getModules(self):
        """
        Gets all modules from the database and returns them as a list of dictionaries.
        :return: class<list> of class<dict> of type<Module>
        """
        self.modules = await self.prisma.module.find_many()

        mods = []

        for i in range(len(self.modules)):
            mods.append(self.modules[i].dict())

        return mods

    async def _getEnrollments(self):
        self.enrollments = await self.prisma.moduleenrollment.find_many()

        enrollments = []

        for i in range(len(self.enrollments)):
            enrollments.append(self.enrollments[i].dict())

        return enrollments

    async def _getFeedbacks(self):
        self.feedbacks = await self.prisma.modulefeedback.find_many()

        feedbacks = []

        for i in range(len(self.feedbacks)):
            feedbacks.append(self.feedbacks[i].dict())

        return feedbacks

    async def _getPlans(self) -> list[dict]:
        self.plans = await self.prisma.planofstudy.find_many()

        plans = []

        for i in range(len(self.plans)):
            plans.append(self.plans[i].dict())

        return plans

    async def _seedUserDB(self):
        print('Seeding user model...')
        accounts = []

        for i in range(self.iterations):
            accounts.append({
                'firstName': self.gen.name(),
                'lastName': self.gen.name(),
                'email': self.gen.email(),
                'openID': str(self.gen.integer()),
            })

        users = await self.prisma.user.create_many(data=accounts)

        self.accounts = accounts
        print('User model seeded successfully with %d documents!' % users)

    async def _seedModuleDB(self):
        print('Seeding module model...')
        modules = []

        template = {
            'moduleName': {
                'typemap': 'word',
                'unique': True,
                'tries': 100
            },
            'moduleNumber': {
                'typemap': 'small_int',
                'unique': True,
                'tries': 100
            },
            'description': 'sentence',
            'duration': 'small_int',
            'intro': 'sentence',
            'numSlides': 'small_int',
            'keywords': {
                'set': ['engineering', 'mathematics', 'physics', 'chemistry', 'biology', 'computer science',
                        'economics']
            }
        }

        self.gen.set_template(template)
        docs = self.gen.documents(self.iterations)

        module_res = await self.prisma.module.create_many(data=docs)

        self.modules = modules
        print('Module model seeded successfully with %d documents!' % module_res)

    async def _seedPlanOfStudyDB(self):
        print('Seeding plan of study model...')
        plans = []

        for account in self.accounts:
            plans.append({
                'studentID': account['id'],
            })

        plan_res = await self.prisma.planofstudy.create_many(data=plans)

        self.plans = plans
        print('Plan of study model seeded successfully with %d documents!' % plan_res)

    async def _seedEnrollmentDB(self):
        print('Seeding enrollment model...')
        enrollments = []

        class EnrollmentRole(Enum):
            STUDENT = 1
            TEACHER = 2
            GRADER = 3

        for pos in self.plans:
            for module in self.modules:
                enrollments.append({
                    'planID': pos['id'],
                    'moduleId': module['id'],
                    'role': random.choice(list(EnrollmentRole)).name
                })

        enrollment_res = await self.prisma.moduleenrollment.create_many(data=enrollments)

        self.enrollments = enrollments
        print('Enrollment model seeded successfully with %d documents!' % enrollment_res)

    async def _seedFeedbackDB(self):
        print('Seeding feedback model...')
        feedbacks = []

        # find the user's ID given a plan ID
        plans = await self._getPlans()

        for enrollment in self.enrollments:
            feedbacks.append({
                'moduleId': enrollment['moduleId'],
                'rating': random.randint(1, 6),
                'feedback': self.gen.sentence(),
                'studentId': list(map(lambda x: x['studentID'], filter(lambda x: x['id'] == enrollment['planID'], plans))).pop()
            })

        feedback_res = await self.prisma.modulefeedback.create_many(data=feedbacks)
        self.feedbacks = feedbacks
        print('Feedback model seeded successfully with %d documents!' % feedback_res)

    async def seedAll(self):
        await self.connect()
        print('Seeding operation started...')

        if Skipper.all in self.skip:
            print('Skipping all model seeding...')
        else:
            if Skipper.user in self.skip:
                print('Skipping user model seeding...')
                self.accounts = await self._getUserAccounts()
            else:
                await self._seedUserDB()

            if Skipper.module in self.skip:
                print('Skipping module model seeding...')
                self.modules = await self._getModules()
            else:
                await self._seedModuleDB()

            if Skipper.plan in self.skip:
                print('Skipping plan of study model seeding...')
                self.plans = await self._getPlans()
            else:
                await self._seedPlanOfStudyDB()

            if Skipper.enrollment in self.skip:
                print('Skipping enrollment model seeding...')
                self.enrollments = await self._getEnrollments()
            else:
                await self._seedEnrollmentDB()

            if Skipper.feedback in self.skip:
                print('Skipping feedback model seeding...')
                self.feedbacks = await self._getFeedbacks()
            else:
                await self._seedFeedbackDB()

            print('All models seeded successfully!')

        await self.disconnect()

    async def cleanupAll(self):
        await self.connect()
        print('Cleanup operation started...')

        if Skipper.all in self.cleanup:
            print('Skipping all model cleanup...')
            return
        else:
            if Skipper.user in self.cleanup:
                print('Skipping user model cleanup...')
            else:
                await self._cleanupUserDB()

            if Skipper.module in self.cleanup:
                print('Skipping module model cleanup...')
            else:
                await self._cleanupModuleDB()

            if Skipper.feedback in self.cleanup:
                print('Skipping feedback model cleanup...')
            else:
                await self._cleanupFeedbackDB()

            if Skipper.plan in self.cleanup:
                print('Skipping plan of study model cleanup...')
            else:
                await self._cleanupPlanOfStudyDB()

            if Skipper.enrollment in self.cleanup:
                print('Skipping enrollment model cleanup...')
            else:
                await self._cleanupEnrollmentDB()

            print('All models cleaned up successfully!')
        await self.disconnect()

    async def _cleanupUserDB(self):
        count = await self.prisma.user.delete_many()
        print('User model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupModuleDB(self):
        count = await self.prisma.module.delete_many()
        print('Module model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupFeedbackDB(self):
        count = await self.prisma.modulefeedback.delete_many()
        print('Feedback model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupPlanOfStudyDB(self):
        count = await self.prisma.planofstudy.delete_many()
        print('Plan of study model cleaned up successfully! (%d documents deleted)' % count)

    async def _cleanupEnrollmentDB(self):
        count = await self.prisma.moduleenrollment.delete_many()
        print('Enrollment model cleaned up successfully! (%d documents deleted)' % count)
