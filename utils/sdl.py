def createMutationString(
        comment="Dummy feedback",
        rating=3,
        moduleID="63e12ee8a30457c24d67bd1a",
        userID="63da9e40020a625cc55f64c5"
):
    mutation = """
    mutation {
        addModuleFeedback(
            moduleId:"%s"
            userId: "%s"
            input:{
                feedback: "%s",
                rating: %d
            }
        ){
            feedback{
                rating
            }
        }
    }   
    """ % (moduleID, userID, comment, rating)
    return mutation


def createModuleMutationString(
        moduleName="Dummy Module",
        moduleNumber=1,
        description="Dummy description",
        duration=1,
        intro="Dummy intro",
        numSlides=1,
        keywords=["Dummy", "Keywords"],
):
    keys = ""
    for words in keywords:
        keys += '"%s",' % words
    mutation = """
    mutation {
        addModule(
            input:{
                moduleName: "%s"
                moduleNumber: %d
                description: "%s"
                duration: %d
                intro: "%s"
                numSlides: %d
                keywords: []
            }
        ){
            id
            moduleName
            moduleNumber
        }
    }   
    """ % (moduleName, moduleNumber, description, duration, intro, numSlides)
    return mutation
