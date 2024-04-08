FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary packages
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libldap2-dev \
    libxslt1-dev \
    unixodbc-dev \
    libxmlsec1-dev \
    libsasl2-dev \
    libffi-dev \
    wget \
    xorg \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-wheel \
    ca-certificates \
    vim \
    gnupg

# Install wkhtmltopdf
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
    tar -xJf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
    cp -r wkhtmltox/* /usr/ && \
    rm -rf wkhtml*

# Set working directory
WORKDIR /rest/supermat

# Copy requirements and install dependencies
COPY ./requirements.txt ./
RUN pip3 install wheel && \
    pip3 install -r requirements.txt

# Copy project files
COPY ./ /rest/supermat