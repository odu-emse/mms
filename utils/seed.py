from essential_generators import DocumentGenerator
from numpy import random
from requests import post
from utils.sdl import createMutationString, createModuleMutationString


def seedDbFeedback():
    print('Seeding feedback model...')
    gen = DocumentGenerator()
    iterations = 25
    modules = getModules()['module']
    for module in modules:
        # create feedback for each module 25 times
        for i in range(iterations):
            res = post('http://localhost:4000/graphql',
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
        res = post('http://localhost:4000/graphql',
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
    mods = post('http://localhost:4000/graphql', {}, {
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
    mods = post('http://localhost:4000/graphql', {}, {
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
