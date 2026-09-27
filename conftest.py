"""
Root conftest: prevent pytest from collecting the demo project's own test suite.

``tests/demo_project/`` contains a minimal application and test file that
is used *as a fixture* for integration tests.  It is not part of the
incident-replay test suite itself and must not be collected directly
(its imports only work when the demo project is the cwd).
"""
collect_ignore_glob = ["tests/demo_project/*"]
