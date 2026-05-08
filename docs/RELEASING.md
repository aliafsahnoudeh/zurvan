# Releasing zurvan

zurvan uses a tag-driven release flow. Pushing a tag named `vX.Y.Z` triggers
[.github/workflows/release.yml](../.github/workflows/release.yml), which:

1. Runs the full test matrix (Python 3.11 / 3.12 / 3.13).
2. Verifies the tag matches the `version` in [pyproject.toml](../pyproject.toml).
3. Builds the wheel and sdist.
4. Publishes to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/) (OIDC — no API tokens).
5. Creates a GitHub Release with auto-generated notes.

## Setting up trusted publishing

Trusted publishing is configured once on PyPI; after that every release
authenticates via the GitHub Actions OIDC handshake, so there is no API
token to store, rotate, or leak.

### On PyPI

1. Create an account at https://pypi.org if you don't have one. Verify your
   email and enable two-factor authentication — PyPI requires both before
   it will let you configure trusted publishing.
2. Open https://pypi.org/manage/account/publishing/.
3. Under **Add a new pending publisher**, select the **GitHub** tab and fill in:

   | Field             | Value            |
   | ----------------- | ---------------- |
   | PyPI Project Name | `zurvan`         |
   | Owner             | `aliafsahnoudeh` |
   | Repository name   | `zurvan`         |
   | Workflow name     | `release.yml`    |
   | Environment name  | `pypi`           |

4. Click **Add**. The entry should appear under **Pending publishers**.

The first successful release graduates the entry from "pending" to a
regular trusted publisher and creates the project on PyPI. Every release
afterwards reuses the same entry.

### On GitHub (optional)

GitHub auto-creates the `pypi` environment on first use. If you want an
approval gate before each publish, pre-create the environment at
*Settings → Environments → New environment* and add yourself (or another
maintainer) as a required reviewer.

## Cutting a release

zurvan follows [Semantic Versioning](https://semver.org): bump the patch
component for bug fixes, the minor component for backwards-compatible
features, the major component for breaking changes.

```bash
# 1. Bump project.version in pyproject.toml, e.g. "0.2.0"
git commit -am "release: v0.2.0"
git push

# 2. Tag and push
git tag v0.2.0
git push origin v0.2.0
```

Watch the workflow at https://github.com/aliafsahnoudeh/zurvan/actions.
On success the new version appears at https://pypi.org/project/zurvan/
within a minute or two.

## Troubleshooting

| Problem | Fix |
| --- | --- |
| Tag doesn't match `pyproject.toml` version | The `build` job fails fast. Delete the tag, fix the version, retag. |
| PyPI rejects with "version already exists" | PyPI versions are immutable. Bump to the next version and retag. |
| Tests fail on the release matrix | Fix on `main`, delete the tag, retag. |
| `invalid-publisher` from the OIDC handshake | The trusted-publisher entry on PyPI is missing or its fields don't match. Open https://pypi.org/manage/account/publishing/ and verify each field against the table above. |

If only the `publish-pypi` or `github-release` job failed, fix the
underlying issue and **re-run only that job** from the Actions UI — no
need to delete the tag and retag.

To delete a bad tag locally and remotely:

```bash
git tag -d v0.2.0
git push origin :refs/tags/v0.2.0
```
