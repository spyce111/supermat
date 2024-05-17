# OS: Ubuntu 22.04 image
FROM ubuntu:22.04

# Non interactive
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
    gnupg \
    poppler-utils

# Install wkhtmltopdf
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
    tar -xJf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
    cp -r wkhtmltox/* /usr/ && \
    rm -rf wkhtml*

# Set working directory
WORKDIR /rest/supermat
RUN mkdir logs /rest/logs
# Copy requirements and install dependencies
COPY ./requirements.txt ./
RUN pip3 install wheel && \
    pip3 install -r requirements.txt

COPY ./requirements1.txt ./
RUN pip3 install -r requirements1.txt

# NLTK and spcay specific package install
RUN python3 -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger && \
    python3 -m spacy download en_core_web_sm
    
# Copy project files
COPY ./ /rest/supermat

# Django Model initial Migrate
RUN python3 manage.py migrate


# # OS: Ubuntu 22.04 image
# FROM ubuntu:22.04

# # Non interactive
# ENV DEBIAN_FRONTEND=noninteractive

# # Update package lists and install necessary packages
# RUN apt-get update -y && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     libxml2-dev \
#     libldap2-dev \
#     libxslt1-dev \
#     unixodbc-dev \
#     libxmlsec1-dev \
#     libsasl2-dev \
#     libffi-dev \
#     wget \
#     xorg \
#     python3-dev \
#     python3-pip \
#     python3-venv \
#     python3-wheel \
#     ca-certificates \
#     vim \
#     gnupg \
#     poppler-utils \
#     chromium-browser \
#     npm

# # Install wkhtmltopdf
# RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
#     tar -xJf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
#     cp -r wkhtmltox/* /usr/ && \
#     rm -rf wkhtml*

# # Set working directory
# WORKDIR /rest/supermat

# # Create logs directory
# RUN mkdir -p logs /rest/logs

# # Copy requirements and install dependencies
# COPY ./requirements.txt ./requirements1.txt ./
# RUN pip3 install --no-cache-dir --upgrade pip && \
#     pip3 install --no-cache-dir -r requirements.txt && \
#     pip3 install --no-cache-dir -r requirements1.txt

# # NLTK and spaCy specific package installation
# RUN python3 -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger && \
#     python3 -m spacy download en_core_web_sm

# # Copy project files
# COPY ./ /rest/supermat

# # Django Model initial Migrate
# RUN python3 manage.py migrate

# # Expose Chroma port
# EXPOSE 9222

# # Start Chroma server
# CMD ["chromium-browser", "--headless", "--disable-gpu", "--no-sandbox", "--remote-debugging-port=9222", "about:blank"]
