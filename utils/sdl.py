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
