# vLLM OpenAI server + audio extras (librosa, soundfile) needed by /v1/audio/transcriptions
FROM vllm/vllm-openai:v0.29.0
RUN pip install --no-cache-dir "vllm[audio]==0.29.0"
