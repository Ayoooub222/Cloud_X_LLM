FROM vllm/vllm-openai:v0.29.0

# Pinned versions for reproducibility.
# numpy is pinned to the version shipped in the base image so that
# the resolver can never upgrade it and break vLLM / torch.
ARG VLLM_VERSION=0.29.0
ARG LIBROSA_VERSION=1.0.0
ARG SOUNDFILE_VERSION=0.14.0
ARG NUMPY_VERSION=2.2.6

# vllm[audio] brings av, soxr, scipy, mistral_common[audio];
# librosa is not part of the extra, so it is added explicitly.
# Everything is installed in one command so pip resolves all constraints together.
RUN python3 -m pip install --no-cache-dir \
      "vllm[audio]==${VLLM_VERSION}" \
      "librosa==${LIBROSA_VERSION}" \
      "soundfile==${SOUNDFILE_VERSION}" \
      "numpy==${NUMPY_VERSION}"

# Smoke test: fail the build if any audio dependency cannot be imported
RUN python3 -c "import librosa, soundfile, soxr, av, numpy; print('librosa', librosa.__version__, 'soundfile', soundfile.__version__, 'av', av.__version__, 'numpy', numpy.__version__)"
