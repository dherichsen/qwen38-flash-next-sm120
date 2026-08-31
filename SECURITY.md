# Security

This project starts a large-model server with Docker. The example binds only
to `127.0.0.1`, drops all container capabilities before adding `IPC_LOCK`, and
enables `no-new-privileges`. Do not expose the OpenAI-compatible endpoint to an
untrusted network without authentication and a separately reviewed proxy.

Report suspected vulnerabilities privately through GitHub's security advisory
feature. Do not include API keys, model-license credentials, private prompts,
or production logs in a public issue.
