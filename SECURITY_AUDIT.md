# Security Audit

Audit date: 2026-08-26

## Results

| Severity | Finding | Status |
| --- | --- | --- |
| Critical | Model and training checkpoints were loaded through unrestricted Python pickle deserialization. Several artifacts were downloaded automatically, including one over HTTP. A malicious checkpoint could execute code with the user's privileges. | Fixed. State-dict loads use `weights_only=True`, the HTTP artifact moved to its publisher's HTTPS URL, and legacy Lightning/OpenCLIP checkpoints require a trusted SHA-256 digest before deserialization. |
| High | The environment pinned Python 3.6, PyTorch 1.7.1, `future` 0.17.1, and setuptools 59.5.0. These versions are end-of-life or affected by published security advisories. | Fixed. The environment now uses Python 3.11, PyTorch 2.13.0, torchvision 0.28.0, and upgraded pinned top-level dependencies. Unused legacy packages were removed. The remaining Lightning advisory is documented below. |
| Medium | Demo and preprocessing scripts ran `pip install pydicom` at runtime without a version pin. | Fixed. Runtime installation was removed and pydicom 3.0.2 is declared in the environment. |
| Medium | DICOM and image inputs are parsed from user-selected local directories without explicit file-size, pixel-count, or aggregate-work limits. Crafted or very large files may cause resource exhaustion. | Open. These are local batch tools rather than remotely exposed services; callers should process trusted datasets or sandbox ingestion. |
| Low | Downloaded datasets and several model archives do not have publisher-provided signatures or repository-pinned checksums. | Open. Executable model deserialization is now restricted or hash-gated, but availability and corruption checks remain desirable. |

PyTorch Lightning 2.6.5 is listed in CVE-2026-58659, and no patched release was
available at audit time. The affected checkpoint API remains usable only after
the repository verifies a caller-supplied trusted digest, preventing an
untrusted checkpoint from reaching that code path.

## Requested Categories With No Finding

- No hardcoded API keys, access tokens, passwords, private keys, or cloud
  credentials were found. High-entropy notebook metadata findings were reviewed
  and were not credentials.
- No SQL client, query construction, or database access exists. Static SQL
  warnings were false positives around unrelated variables.
- No HTTP server, route, CORS configuration, authentication, authorization, or
  debug endpoint exists in this research codebase.
- User-controlled values are local file paths and datasets, not remote request
  parameters.

## Validation

- Bandit static security scan
- detect-secrets scan plus repository-history credential-pattern review
- OSV dependency advisory queries
- Python bytecode compilation
- SHA-256 verification unit tests
- Safe-mode loading of representative published robust and DINO checkpoints
- Dependency resolution check for the pinned pip environment
