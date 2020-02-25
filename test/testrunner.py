import os

#from src.github import GitHubHandler

if __name__ == "__main__":
    print('*************************')
    print(os.getenv('CODEBUILD_LOG_PATH'))
    #handler = GitHubHandler()
    #handler.set_status(state="success", context=context, description="test")
    print('*************************')
