# tests/integration/test_rag_smoke.py
"""Manual end-to-end smoke test for the real Docker Compose stack, including a
real Claude API call. Not part of `pytest tests/` (no `test_*` function is
defined at module level — see test_docker_compose_smoke.py's own docstring
for why). Run directly, after the stack is up (see docker/docker-compose.yml
and Task 15 of the rag-pipeline plan) and ANTHROPIC_API_KEY is a real key.

    uv run python tests/integration/test_rag_smoke.py

Expected: register/login succeed, a small .txt document uploads and is
searchable, and /chat returns a real Claude-generated answer that's actually
grounded in the uploaded content.
"""
import asyncio

import httpx


async def _run() -> None:
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        await client.post("/auth/register", json={"email": "smoke-rag@example.com", "password": "hunter2hunter2"})
        login = await client.post("/auth/login", json={"email": "smoke-rag@example.com", "password": "hunter2hunter2"})
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        upload = await client.post(
            "/documents",
            headers=headers,
            files={"file": ("facts.txt", b"The unified RAG x CAG x MAG project targets a 7900 XTX GPU for local serving.", "text/plain")},
        )
        print("upload:", upload.status_code, upload.json())
        assert upload.status_code == 201

        search = await client.post(
            "/documents/search", headers=headers, json={"query": "what GPU does this project target", "top_k": 5}
        )
        print("search:", search.status_code, search.json())
        assert search.status_code == 200

        chat = await client.post("/chat", headers=headers, json={"question": "What GPU does this project target?"})
        print("chat:", chat.status_code, chat.json())
        assert chat.status_code == 200
        assert "answer" in chat.json()


if __name__ == "__main__":
    asyncio.run(_run())
