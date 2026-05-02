# Publication Checklist

This repository is currently a staged curation of a working research directory.
Before making it public, check the following.

## Licensing

- Choose a source-code license.
- Confirm whether dictionary-derived files can be redistributed.
- Confirm whether interlinear seed/correction files may be public examples.
- Confirm whether any speaker-derived text examples require anonymization or
  permission notes.

## Data Boundaries

Do not commit:

- raw audio or video
- full LL Tools export bundles
- trained model checkpoints
- extracted bundle directories
- virtual environments
- cache directories
- large generated transcription outputs
- private site credentials or API tokens

Use GitHub Releases, Hugging Face, or a separate dataset repository for
artifacts that are safe and useful to publish.

## Path And Secret Review

Run:

```bash
rg -n -e 'C:\\Users\\' -e '/mnt/c/Users/' -e 'ghp_' -e 'sk-[A-Za-z0-9]' -e 'GITHUB_TOKEN' .
```

Expected result:

- no personal absolute paths
- no credentials
- only harmless code references to environment variable names such as
  `OPENAI_API_KEY`

## Public Usability

- Add minimal installation instructions.
- Add a tiny synthetic or permission-safe sample manifest.
- Add a small end-to-end command sequence.
- Mark scripts that are historical or project-specific.
- Split reusable library code from one-off batch scripts when there is time.

## Suggested Release Path

1. Keep the initial GitHub repo private.
2. Review data/licensing boundaries.
3. Add a real license.
4. Add minimal test/sample data.
5. Make the repo public once the examples are safe.
