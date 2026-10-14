# Rules for Writing Good Tests

These rules synthesize the best practices from Kent Beck (Test Desiderata, programmer tests), Vladimir Khorikov (four pillars of a good unit test), Google Testing Blog, Microsoft guidelines, and modern TDD/AI-agent workflows.

Goal: tests that protect against regressions, resist unnecessary changes, give fast feedback, and stay cheap to maintain. Prefer behavior over implementation. Tests are living documentation.

## Four Pillars of a Valuable Test (Khorikov)

Every test should score high on:

1. **Protection against regressions** — Detects real bugs in important behavior.
2. **Resistance to refactoring** — Does not fail when internal structure changes but observable behavior stays the same. Couples to the public contract, not implementation details.
3. **Fast feedback** — Runs in milliseconds. Prefer in-memory over I/O, network, or real clocks/random.
4. **Maintainability** — Short, readable, easy to understand and change. Low setup cost.

Multiply the four scores: a test that fails any pillar has low value.

## Kent Beck / Test Desiderata Priorities

Prefer tests that are:

- **Isolated** and **composable**: independent of order, other tests, or shared mutable state.
- **Deterministic**: same inputs → same result. No flakiness from time, random, network, or external systems.
- **Fast**.
- **Behavioral** (sensitive to behavior changes) and **structure-insensitive**.
- **Specific**: failure points clearly at the cause.
- **Readable** and **writable**.
- **Automated**.

Trade-offs exist. Do not sacrifice determinism or resistance to refactoring for coverage numbers.

## Structure

- Follow **Arrange – Act – Assert** (AAA). Make the three sections obvious (blank lines or comments).
- Prefer one logical behavior per test. Multiple related assertions are acceptable if they verify the same outcome.
- Keep the Act section to a single call or action.
- Use the minimal data needed to exercise the behavior under test.
- Prefer Given–When–Then wording when it improves clarity (especially for higher-level tests).

## Naming

Names must read as documentation. A reader should understand the scenario and expected result without opening the test body.

Preferred patterns:

- `methodOrBehavior_condition_expectedResult`
- `shouldDoX_whenY`
- `doesNotDoX_whenY`

Avoid: `test1`, `testAdd`, `works`, `success`.

## What and How to Test

- Test the public contract / observable behavior, not private methods or internal steps.
- Cover: happy path, important edge cases, boundary values, and error/invalid input paths.
- Use non-default, non-zero, non-empty values so accidental defaults do not hide bugs.
- Prefer black-box style (inputs → outputs/side effects the caller cares about) over white-box verification of internal calls.
- For complex objects, prefer asserting the final state or return value over verifying every intermediate interaction.
- Do not assert on log messages, exact exception text, or private fields unless they are part of the contract.
- Avoid snapshot tests for logic that should be asserted precisely; use them sparingly for large structured outputs when appropriate.

## Isolation and Dependencies

- Tests must not depend on each other or on execution order.
- Reset or isolate all state (files, statics, databases, caches) between tests.
- Mock or stub only what is slow, non-deterministic, or external (network, real DB, system clock, random, file system when unnecessary).
- Prefer real, fast, deterministic collaborators over mocks when possible. Over-mocking tests the mocks, not the code.
- Do not assert interactions with stubs. Assert on the result or the state that matters.
- Inject dependencies so tests control them.

## Types of Tests and the Pyramid

- Prefer many fast unit tests that exercise domain logic in isolation.
- Add integration tests for collaborations between components (databases, message queues, external APIs behind interfaces).
- Keep end-to-end / UI tests few, focused on critical user journeys, and stable.
- A failure in a high-level test should usually be reproducible by a lower-level test; otherwise the lower-level suite is incomplete.
- Match the project's existing test style, frameworks, and conventions (describe/it vs. test, assertion library, fixture style, etc.).

## Anti-Patterns to Avoid

- Tests that fail on refactoring without behavior change (over-specified, white-box, brittle).
- Shared mutable state or order-dependent tests.
- Slow tests that discourage running the suite.
- Tests that only verify that a mock was called, without checking outcomes.
- Testing trivial getters/setters or third-party library behavior.
- Copy-paste tests with only data changed — extract helpers or use parameterized tests when the behavior is the same.
- Weakening, deleting, or skipping assertions merely to make a suite green.
- Claiming “tests pass” without actually running them and observing the result.

## Guidelines

- Read existing tests in the same module/project first and match their style, naming, and helpers.
- Prefer writing the failing test first (TDD) when the behavior is new or changed, then implement until it passes.
- After writing or modifying production code, run the relevant tests and report the actual command and outcome. Never claim success without execution evidence.
- Do not modify, delete, or weaken existing tests or assertions unless the behavior contract itself is intentionally changing.
- When generating tests, include the important edge and error cases the user or requirements imply; do not stop at the happy path.
- Keep tests self-contained so a reader can understand the scenario from the test alone.
- If a test is hard to write, that is a design smell — prefer improving the design for testability over complicated test setup.

## Quick Checklist Before Finishing

- [ ] Test name describes the behavior and condition.
- [ ] AAA structure is clear.
- [ ] Tests only the public observable behavior.
- [ ] Independent and deterministic.
- [ ] Fast (no unnecessary I/O or real external systems).
- [ ] Covers happy path + relevant edges/errors.
- [ ] Does not couple to internal implementation details.
- [ ] Existing suite still passes; new tests were actually executed.
- [ ] Matches project conventions.

Apply these rules consistently. Value comes from tests that stay green when the code is correct and turn red only when behavior regresses.
