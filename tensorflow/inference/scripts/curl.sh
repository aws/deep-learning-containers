#!/bin/bash
#
# Some example curl requests to try on local docker containers.

curl -X POST --data-binary @test/resources/inputs/test.json -H 'Content-Type: application/json' -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' http://localhost:8080/invocations
curl -X POST --data-binary @test/resources/inputs/test-gcloud.jsons -H 'Content-Type: application/json' -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' http://localhost:8080/invocations
curl -X POST --data-binary @test/resources/inputs/test-generic.json -H 'Content-Type: application/json' -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' http://localhost:8080/invocations
curl -X POST --data-binary @test/resources/inputs/test.csv -H 'Content-Type: text/csv' -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=half_plus_three' http://localhost:8080/invocations
curl -X POST --data-binary @test/resources/inputs/test-cifar.json -H 'Content-Type: application/json' -H 'X-Amzn-SageMaker-Custom-Attributes: tfs-model-name=cifar' http://localhost:8080/invocations