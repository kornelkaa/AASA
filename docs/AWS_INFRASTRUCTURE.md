# AASA Infrastructure

This document gives an overview of how AASA is built and deployed. It does not include account details, addresses, or security settings.

- **Live site:** https://aasa.huszka.hu
- **Source code:** https://github.com/kornelkaa/AASA

## Services

| Service | Purpose |
| --- | --- |
| GitHub Actions | Runs tests and builds the Docker image on every push |
| Amazon ECR | Stores the Docker images |
| Amazon EC2 | Runs the app in a Docker container |
| Cloudflare | DNS and HTTPS for the live site |
| Groq | External LLM API used for safety decisions |
| Komodo | Manages the app container on the server |

## How a change gets deployed

```
Push to GitHub
   │
   ▼
GitHub Actions
   ├── Run tests (pytest)
   └── On main only:
         ├── Build the Docker image
         ├── Push the image to ECR
         └── Start the new image on the server
                  │
                  ▼
         EC2 server (Docker container)
           ├── Loads the nuScenes dataset from the server disk
           └── Loads its configuration from a .env file on the server
```

## What the app needs at runtime

The Docker image does not contain the dataset or the API key. The server must provide them when the container starts:

- The nuScenes v1.0-mini dataset, mounted into the container.
- A `.env` file with the Groq API key.

If either is missing, the app fails to start and keeps restarting.

## Notes

- Secrets are stored in GitHub and on the server, not in the repository.