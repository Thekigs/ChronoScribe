# Security

Do not commit API keys, tokens, private model adapters, local paths, or raw data with access restrictions. Use `.env` for local credentials and keep it ignored. If a credential was ever committed, revoke it at the provider before removing the file.

The local API server binds to `127.0.0.1` by default, disables browser cross-origin access by default, and does not provide authentication. Do not pass `--host 0.0.0.0` or expose the port publicly without putting an authenticated, access-controlled reverse proxy in front of it. If a trusted browser client is required, pass its exact origin with `--allow-origin`.

To report a security issue, open a private report through the repository owner's GitHub security contact rather than posting credentials publicly.
