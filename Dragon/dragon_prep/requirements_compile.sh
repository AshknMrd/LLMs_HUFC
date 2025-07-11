docker run --rm -it -v .:/opt/app \
    dragon_prep:latest pip-compile /opt/app/setup.cfg
