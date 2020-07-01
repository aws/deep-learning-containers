from job_requester.requester import JobRequester


def main():
    test = JobRequester()
    response = test.send_request(
        "754106851545.dkr.ecr.us-west-2.amazonaws.com/pr-tensorflow-training:2.2.0-gpu-py37-cu101-ubuntu18.04-example-pr-269-2020-06-11-22-13-27",
        "PR",
        1,
    )
    print(response)


if __name__ == "__main__":
    main()
