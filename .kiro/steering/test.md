______________________________________________________________________

## inclusion: always description: Test strategy and guidelines for the deep-learning-containers repository.

# Test Strategy & Guidelines

## Philosophy

### Tests are the product

A product is only as good as its tests. This is not a metaphor — it is literally true. The tests are the only proof that the product works. Without them, you are shipping hope.

Bad tests produce a bad product because they fail to catch defects before customers do. Outdated tests produce an outdated product because they validate behavior that no longer matters while ignoring behavior that does. A test suite carrying tech debt is a product silently slipping into a bad state — every skipped test, every ignored failure, every "we'll fix that later" is a gap in your proof that the product works. If the tests don't prove it, it isn't proven.

### Write the test first

Test-driven development means writing the test before writing the code that makes it pass. This matters because it changes how you think. When you write code first and tests second, the test becomes an afterthought — a formality that confirms what you already built. When you write the test first, you are forced to define what correct behavior looks like before you decide how to achieve it. The test becomes the specification. The implementation becomes the proof.

Code written without a failing test to motivate it tends to be undertested, overbuilt, or both. You end up with features that "work" but have no clear definition of what working means, and tests that were written to pass rather than to catch failures.

### Tests are documentation

A new team member should be able to read the test names alone and understand what the product guarantees. Each test name is a statement of fact about the image: "CUDA is available," "training loss decreases," "SSH is configured on port 2222." Together, they form a living specification that stays in sync with the code because it *is* the code.

If a capability isn't tested, it's effectively undocumented. It may work today, but nothing prevents it from breaking tomorrow. Undocumented behavior will eventually break without anyone noticing — and without a test, no one will notice until a customer reports it.

### Test the contract, not the implementation

Tests should assert on what customers depend on — packages import, CUDA works, training converges, checkpoints round-trip correctly. They should not assert on internal details like specific file sizes, exact log output, or internal directory structures that aren't part of the public contract.

The distinction matters because implementations change. A refactor that reorganizes internal files, changes a log format, or restructures a build step should not break tests — as long as the customer-facing behavior is preserved. Implementation-coupled tests create a perverse incentive: they punish improvement by making every internal change a test-fixing exercise, which slows the team down and teaches them to avoid refactoring.

### Delete tests that don't earn their keep

A test that has been skipped for months is not a safety net — it is clutter that gives the illusion of coverage. A test that passes regardless of whether the feature actually works is actively harmful because it provides false confidence. Both should be deleted.

Regularly audit the suite. For each test, ask: has this test ever caught a real bug? Has it ever prevented a real regression? If the answer to both is no, and the test doesn't validate a customer-facing contract, question whether it belongs. A smaller suite of meaningful tests is more valuable than a large suite padded with tests that exist only to inflate a coverage number.

### Tests must be deterministic

Flaky tests are the single most corrosive force in a test suite. A test that fails intermittently — due to unseeded randomness, timing dependencies, or race conditions — trains the team to dismiss failures. The pattern is predictable:

1. A test fails. Someone investigates. No real bug — just flakiness. They re-run. It passes.
1. This happens a few more times. The team learns: "that test is flaky, just re-run it."
1. The reflex becomes automatic. See failure → re-run → green → move on.
1. One day the test fails because of a real regression. But the reflex doesn't turn off. Someone re-runs it. It happens to pass (the bug only manifests under certain conditions). The regression ships.

At scale, even a handful of flaky tests degrades confidence in the entire suite. Developers stop treating CI as a trusted gate and start treating it as an obstacle to retry until it's green. Once that happens, the suite has lost its primary value: being a reliable signal of correctness.

The fix is aggressive. If a test involves randomness (e.g. training with random initialization), either seed it or assert on properties that hold regardless of seed — loss decreased, gradients are finite, all-reduce sum is mathematically correct. If a test can't be made deterministic, delete it. A test that sometimes lies is worse than no test at all, because no test doesn't give you false confidence.

## Separate Test Logic from Infrastructure

Tests contain only assertions and validation logic. All infrastructure — starting containers, attaching GPUs, creating networks — is handled by the CI workflow or replicated manually by the developer.

When test code is coupled with infrastructure code (e.g. a test that calls `docker run` to spawn a container, runs a command inside it, then parses the output), several problems emerge:

- **Environment lock-in**: Tests that shell out to `docker run` only work on hosts with a Docker daemon. They can't run inside a container, on a different container runtime, or in a CI system that manages containers differently. Moving to a new CI platform means rewriting test infrastructure, not just updating a workflow file.
- **Fragile abstractions**: Test fixtures that wrap Docker commands become a maintenance burden — they accumulate flags, error handling for container failures, and timeout logic that has nothing to do with what the test is actually validating. A bug in the fixture breaks every test, even though the image itself is fine.
- **Unclear failure signals**: When a test fails, you want to know whether the image is broken or the test harness is broken. Coupling the two makes this harder to distinguish. A pure Python test that fails inside the container points directly at an image problem.
- **Difficult to run locally**: A developer who wants to quickly validate a single test shouldn't need to understand the fixture's Docker orchestration. With separated concerns, they just exec into a running container and run the test.

In practice: a test file is plain Python that imports packages, calls functions, and asserts results — it assumes it is already running inside the correct environment. The workflow (or developer) is responsible for building the image, starting the container with the right configuration, and invoking the test runner inside it. If the infrastructure setup needs to change, only the workflow changes — not the tests.

## Writing Test Cases

### Unit tests: maximize coverage without expensive compute

Unit tests run on CPU only and require no GPUs or specialized hardware. Because they are cheap to run, they should be comprehensive — covering as much of the image's correctness surface as possible. This includes verifying that expected packages are installed and importable, versions match the declared pins, required files and directories exist, environment variables and PATH entries are set correctly, binaries are present and executable, and configuration files contain the right values.

The goal is to catch the majority of image defects — missing dependencies, broken installs, misconfigured paths — before any GPU time is spent. A well-maintained unit test suite acts as a fast, reliable gate that prevents obviously broken images from reaching more expensive validation stages.

### Functional tests: validate real use cases in depth

Functional tests consume expensive compute resources (GPUs, multi-node clusters). Because of this cost, each functional test should justify its resource usage by testing an actual end-to-end use case with meaningful validation — not just checking that an API call doesn't crash.

A shallow test that only verifies a function returns without error provides little signal. A strong functional test exercises the full code path a customer would use and asserts on the outcome: that training loss actually decreases over steps, that gradients are finite and synchronized across ranks, that a checkpoint can be saved and restored with identical weights, that mixed-precision produces numerically correct results.

When writing functional tests, ask: "If this test passes, what can I confidently tell a customer?" If the answer is only "the function didn't throw an exception," the test is too shallow. If the answer is "distributed training with this framework produces correct, converging results on this image," the test is earning its GPU time.

## Running Tests Outside CI

When testing outside CI (e.g. on a remote EC2 instance), replicate the workflow setup yourself: build the image, start a container with the appropriate configuration, then run the tests inside it. See the framework-specific workflow YAML for the exact setup steps to replicate.
