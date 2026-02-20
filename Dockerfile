FROM vllm/vllm-openai:latest

RUN pip install --no-deps "transformers>=5.1.0" "peft>=0.15.2" && \
    pip install --upgrade huggingface_hub
