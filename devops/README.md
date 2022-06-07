# Testing Strategy

Currently, test pipelines are run when PRs are made to main and nightly. The nightly badge on the root README of the repo reflects the nightly unit tests.

The Windows test pipelines are run against PRs and nightly and should run successfully.
The Mac test pipeline also runs nightly, but currently has known failures. These failures are not expected to affect the user experience of Mac users.