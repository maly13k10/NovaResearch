# Online Deployment — Access NovaResearch AI from Anywhere

## Recommended: Streamlit Community Cloud
1. Create a GitHub repository for this folder.
2. Commit/push all project files except local `.venv`, `build` and `dist`.
3. Sign in to Streamlit Community Cloud using GitHub.
4. Create a new app from the repository.
5. Set the entry point to `Nova.py`.
6. Optional: add `OPENAI_API_KEY` to the app's Secrets for cloud Ask Nova. The app works without it.
7. Deploy.
8. Share the generated HTTPS URL with the professor.

The same public URL can be opened from a university PC, home computer, phone or tablet as long as the device has internet access and the network does not block the hosting domain.

## Docker deployment
A `Dockerfile` is included. Build/run:

```bash
docker build -t novaresearch-ai .
docker run -p 8501:8501 novaresearch-ai
```

The same image can be deployed to a Docker-compatible cloud host.

## Same local network only
Run `start_nova_lan.bat`. It binds Streamlit to `0.0.0.0` so another device on the same Wi-Fi/LAN can open `http://YOUR-PC-IP:8501`.

## Important
This source package cannot create a permanent public URL by itself. A hosting account (for example Streamlit Community Cloud or a Docker host) is required for the final public address.
