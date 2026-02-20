.PHONY: up-nvidia down logs test pull build install-mac up-mac down-mac logs-mac

VLLM_MODEL := jinaai/jina-embeddings-v5-text-nano-retrieval

build:
	docker compose build

up-nvidia: build
	docker compose --profile nvidia up -d

down:
	docker compose down

logs:
	docker compose logs -f

test:
	.venv/bin/python3 test_embeddings.py

pull:
	docker compose pull
