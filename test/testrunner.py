import os

#from src.github import GitHubHandler

if __name__ == "__main__":
    print('*************************')
    print(os.getenv('TEST_TRIGGER'))
    context = "dlc-sanity-test-{}".format(os.getenv("TEST_TRIGGER"))
    print(context)
    #handler = GitHubHandler()
    #handler.set_status(state="success", context=context, description="test")
    print('*************************')
