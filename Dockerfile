# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    LIBTORCH=/opt/libtorch \
    LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/cargo/bin:/root/.foundry/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build git curl unzip pkg-config \
    libgmp-dev libmpfr-dev liblmdb-dev libclang-dev clang z3 libz3-dev \
    && rm -rf /var/lib/apt/lists/*

# Download official C++ LibTorch 2.5.1+cu124 for tch-rs
RUN curl -L https://download.pytorch.org/libtorch/cu124/libtorch-cxx11-abi-shared-with-deps-2.5.1%2Bcu124.zip -o /tmp/libtorch.zip && \
    unzip -q /tmp/libtorch.zip -d /opt && rm /tmp/libtorch.zip

# Install Rust 1.82.0 toolchain
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.82.0

# Install Foundry 1.0.0-stable
RUN curl -L https://foundry.paradigm.xyz | bash && foundryup --version nightly-e3f9a1c8a9fb3b2e7f480e8f4b4b9c2f1a8d2e1b

WORKDIR /workspace
COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY tests/ ./tests/

# Compile Rust workspace binary with LTO and AVX2/SIMD
RUN RUSTFLAGS="-C target-cpu=haswell -C lto=thin" cargo build --release -p gfn-cli --bin gfn-forge

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgmp10 libmpfr6 liblmdb0 libz3-4 git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -g 10001 gfnuser && useradd -u 10001 -g gfnuser -m -s /bin/bash gfnuser

COPY --from=builder /opt/libtorch/lib /opt/libtorch/lib
COPY --from=builder /root/.foundry/bin/forge /usr/local/bin/forge
COPY --from=builder /root/.foundry/bin/cast /usr/local/bin/cast
COPY --from=builder /root/.foundry/bin/anvil /usr/local/bin/anvil
COPY --from=builder /workspace/target/release/gfn-forge /usr/local/bin/gfn-forge
COPY configs/ /app/configs/

ENV LD_LIBRARY_PATH="/opt/libtorch/lib:${LD_LIBRARY_PATH}" \
    PATH="/usr/local/bin:${PATH}" \
    RAYON_NUM_THREADS=8 \
    RUST_LOG="info,gfn_prover=debug"

USER gfnuser
WORKDIR /app
EXPOSE 8080 9090
ENTRYPOINT ["gfn-forge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
