# GitHub Publishing

The local staging repository is ready to push, but the local token available on
this machine did not have permission to create a new GitHub repository through
the API.

To publish:

1. Create an empty repository on GitHub.
2. Keep it private until the publication checklist is complete.
3. From this repository, run:

```bash
scripts/push_after_remote_created.sh https://github.com/OWNER/REPO.git
```

Suggested repository name:

```text
zazaki-stt-interlinear-workflows
```

Recommended initial visibility:

```text
private
```

Make it public after choosing a license and confirming that the example
interlinear/dictionary-derived material can be redistributed.
