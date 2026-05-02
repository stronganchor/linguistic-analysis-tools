# GitHub Publishing

This repository has been pushed to:

```text
https://github.com/stronganchor/linguistic-analysis-tools
```

The local checkout is:

```text
C:\Users\messy\OneDrive\Documents\GitHub\linguistic-analysis-tools
```

Historical note: the local token available on this machine could authenticate
and push, but it did not have permission to create a new GitHub repository
through the API. The repository therefore had to be created first on GitHub.

For a future freshly created empty repository:

1. Create an empty repository on GitHub.
2. Keep it private until the publication checklist is complete.
3. From this repository, run:

```bash
scripts/push_after_remote_created.sh https://github.com/OWNER/REPO.git
```
