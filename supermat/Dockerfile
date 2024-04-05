# FROM ubuntu:16.04
# ENV DEBIAN_FRONTEND=noninteractive
# RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
#     libxml2-dev \
#     libldap2-dev \
#     libxmlsec1-dev \
#     libsasl2-dev \
#     libffi-dev \
#     wget \
#     xorg

# RUN apt-get install python3-dev python3-pip python3-venv python3-wheel -y

# RUN apt-get install -y --no-install-recommends postgresql postgresql-contrib
# RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
#       tar -xJf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
#       cp -r wkhtmltox/* /usr/ && \
#       rm -rf wkhtml*
# WORKDIR /rest/supermat

# COPY ./requirements.txt ./
# RUN pip3 install wheel
# RUN pip3 install -r requirements.txt
# COPY ./ /rest/supermat

# RUN apt-get update && apt-get install -y gnupg
# RUN apt-get install -y vim

FROM ubuntu:16.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential \
    libxml2-dev \
    libldap2-dev \
    libxslt1-dev \
    unixodbc-dev \
    libxmlsec1-dev \
        libsasl2-dev \
        libffi-dev \
        wget\
        xorg
# RUN apt-get install -y --no-install-recommends \
#         python-dev \
#         python-pip \
#         python-setuptools \
#         libpq-dev \
#         xorg \
#         wget \ 
#        python-lxml
RUN apt-get install --no-install-recommends python3-dev python3-pip python3-venv python3-wheel -y

#RUN apt-get install -y --no-install-recommends postgresql postgresql-contrib
RUN wget https://github.com/wkhtmltopdf/wkhtmltopdf/releases/download/0.12.4/wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
      tar -xJf wkhtmltox-0.12.4_linux-generic-amd64.tar.xz && \
      cp -r wkhtmltox/* /usr/ && \
      rm -rf wkhtml*
WORKDIR /rest/supermat

COPY ./requirements.txt ./
# RUN pip install --upgrade pip
RUN pip3 install wheel && \
    pip3 install -r requirements.txt
COPY ./ /rest/supermat

RUN apt-get update && apt-get install gnupg
RUN apt-get install -y ca-certificates wget

RUN apt-get install -y vim --force-yes
