from requests import post


def testConnection():
    print('Testing connection...')
    # test connection to API
    response = post('http://localhost:4000/graphql', {}, {
        'query': """query{
              module(input:{
              }){
                id
              }
        }"""
    })

    if response.status_code == 200:
        print('API is running')
    else:
        print('API is not running')
        exit(1)
