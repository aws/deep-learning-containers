# Changelog

## v1.1.5 (2019-10-22)

### Bug fixes and other changes

 * update instance type region availability

## v1.1.4 (2019-09-26)

### Bug fixes and other changes

 * build context in release build
 * make consistent build context for py versions
 * Revert "Update build context on 1.4.1 EI dockerfiles (#64)"
 * a typo in build_all script
 * Update build context on 1.4.1 EI dockerfiles
 * correct typo in buildspec
 * enable PR build
 * separate pip upgrade from other installs
 * New py2 dlc dockerfiles
 * New py3 dlc dockerfiles

## v1.1.3.post0 (2019-08-29)

### Documentation changes

 * Update the build instructions to match reality.

## v1.1.3 (2019-08-28)

### Bug fixes and other changes

 * retry mms until it's ready

## v1.1.2 (2019-08-17)

### Bug fixes and other changes

 * split cpu and gpu tests for deployments in buildspec-release.yml
 * add missing placeholder in test for non-eia test command.

## v1.1.1 (2019-08-15)

### Bug fixes and other changes

 * fix flake8
 * update no-p2 regions and no-p3 regions.
 * Skipping EIA test if accelerator type is None.

## v1.1.0 (2019-07-18)

### Features

 * add MXNet 1.4.1 Dockerfiles

### Bug fixes and other changes

 * use Python 2 build logic for EI images during release
 * add missing files needed for building MXNet 1.4.1 Python 2 images
 * configure flake8 to ignore the docker/ directory

## v1.0.6 (2019-07-03)

### Bug fixes and other changes

 * add retries to all integ tests commands in buildspec-release.yml

## v1.0.5 (2019-07-03)

### Bug fixes and other changes

 * fix account number for EI deployment test in buildspec-release.yml

## v1.0.4 (2019-07-01)

### Bug fixes and other changes

 * remove unnecessary pytest marks
 * add retries to remote integ tests in buildspec-release.yml
 * update tests to except pytest's ExceptionInfo object
 * parametrize Python version and processor type in integ tests

## v1.0.3 (2019-06-28)

### Bug fixes and other changes

 * fix account number in deployment test command

## v1.0.2 (2019-06-28)

### Bug fixes and other changes

 * remove nonexistent EI GPU images from buildspec-release.yml

## v1.0.1 (2019-06-27)

### Bug fixes and other changes

 * add release buildspec
 * Fix SageMaker integration tests
 * skip GPU test in regions with limited p2s or p3s
 * Add link to SageMaker integ test setup requirements to README
 * Add SageMaker Elastic Inference test

## v1.0.0

Initial commit
