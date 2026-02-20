.PHONY: up-nvidia down logs test pull build setup generate

VLLM_MODEL := jinaai/jina-embeddings-v5-text-nano-retrieval

build:
	docker compose build

up: build
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

pull:
	docker compose pull

setup:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

generate:
	.venv/bin/python3 test_embeddings.py --generate

test:
	.venv/bin/python3 test_embeddings.py
