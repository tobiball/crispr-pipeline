# Dockerfile
# Use a Debian 12 (Bookworm)-based Rust image
FROM rust:1.83-slim

# 1) Install build dependencies and OpenSSL dev libs
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    zlib1g-dev \
    libssl-dev \
    ca-certificates \
    libcurl4 \
    libbz2-dev \
    --no-install-recommends && \
    rm -rf /var/lib/apt/lists/*



# 2) Download and build Python 2.7.18 from source
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/2.7.18/Python-2.7.18.tgz && \
    tar xzf Python-2.7.18.tgz && \
    cd Python-2.7.18 && \
    ./configure \
        --prefix=/usr/local \
        --enable-shared \
        --enable-unicode=ucs4 \
        --with-ensurepip=install \
        --with-openssl && \
    make -j4 && make install && \
    cd .. && rm -rf Python-2.7.18*

# 3) Make sure the system knows about /usr/local/lib
RUN echo "/usr/local/lib" > /etc/ld.so.conf.d/local-python2.conf && ldconfig

# 4) Set environment so /usr/local/bin is first in PATH
ENV PATH="/usr/local/bin:${PATH}"
# Optionally set LD_LIBRARY_PATH if needed
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# (IMPORTANT) 5) Force project_root() to be "/app" so config.json is written under /app
ENV PROJECT_ROOT=/app

# 6) Verify python2.7 can import encodings/site
RUN python2.7 --version && \
    python2.7 -c "import encodings; print('encodings is at', encodings.__file__)" && \
    python2.7 -m ensurepip && \
    python2.7 -m pip install --upgrade pip && \
    python2.7 -m pip install virtualenv

# 7) Create /app dir, copy your project
WORKDIR /app
COPY . /app

# 8) Create virtualenv with python2.7 inside chopchop/ subdir
RUN rm -rf ./chopchop/chopchop_env && \
    virtualenv -p python2.7 ./chopchop/chopchop_env && \
    ./chopchop/chopchop_env/bin/pip install --upgrade pip && \
    ./chopchop/chopchop_env/bin/pip install -r requirements.txt

# 9) Build the Rust workspace
RUN cargo build --workspace
# (Or cargo build --release --workspace if you prefer)

# 10) Default entrypoint
ENTRYPOINT ["./target/debug/validator"]
