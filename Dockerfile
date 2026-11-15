# NOHI energy bench - image used by the Green Metrics Tool (and usable on any machine with Docker)
FROM python:3.13-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
# versions pinned to those used for the experiments of the paper
RUN pip install --no-cache-dir numpy==2.5.3 pandas==3.0.5 pyarrow==26.0.1 scikit-learn==1.9.1 scipy==1.18.1 \
    xgboost==3.4.1 lightgbm==4.7.0 matplotlib==3.11.2 huggingface_hub==1.33.0
ENV NOHI_BENCH_DIR=/bench MPLBACKEND=Agg
WORKDIR /bench
COPY . /bench/
# if sample.parquet is not in the repository, the data set is downloaded from its public source at build time
# and only a stratified sample is kept in the image
RUN test -f /bench/sample.parquet || (python3 bench_energy.py prepare --n 300000 && rm -rf /bench/raw /root/.cache)
CMD ["sleep", "infinity"]
