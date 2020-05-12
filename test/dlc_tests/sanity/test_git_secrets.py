import os

from invoke.context import Context


# def test_git_secrets():
#     ctx = Context()
#     pwd = ctx.run("pwd").stdout.strip("\n")
#     repository_path = pwd
#     while os.path.basename(repository_path) != "deep-learning-containers":
#         repository_path = os.path.dirname(repository_path)
#     with ctx.cd(repository_path):
#         ctx.run("git clone https://github.com/awslabs/git-secrets.git")
#         with ctx.cd("git-secrets"):
#             ctx.run("make install")
#         ctx.run("git secrets --install")
#         ctx.run("git secrets --register-aws")
#         ctx.run("git secrets --list", echo=True)
#         scan_results = ctx.run("git secrets --scan", echo=True)
#     assert scan_results.ok, scan_results.stderr
