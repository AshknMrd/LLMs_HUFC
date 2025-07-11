FROM python:3.10-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# install dependencies
COPY --chown=user:user requirements.txt /opt/app
RUN python -m pip install --user -r /opt/app/requirements.txt

# install diag-radiology-report-anonymizer, which is a private package
# COPY --chown=user:user diag-radiology-report-anonymizer ./diag-radiology-report-anonymizer
# RUN pip install ./diag-radiology-report-anonymizer

# install dragon_prep
RUN mkdir -p /opt/app/dragon_prep
COPY --chown=user:user . /opt/app/dragon_prep
RUN python -m pip install --user /opt/app/dragon_prep
