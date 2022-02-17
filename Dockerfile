FROM tensorflow/tensorflow:1.5.0-devel-gpu

RUN apt update
RUN apt install --no-install-recommends -y build-essential checkinstall libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev git wget vim openssl locales sudo python3-pip python3-dev
RUN wget https://www.python.org/ftp/python/3.6.15/Python-3.6.15.tar.xz && \
	tar xvf Python-3.6.15.tar.xz && \
	rm -f Python-3.6.15.tar.xz && \
	cd Python-3.6.15/ && \
	./configure && \
        make altinstall && \
	cd .. && \
	rm -rf Python-3.6.15
RUN locale-gen "en_US.UTF-8"
RUN update-locale LC_ALL="en_US.UTF-8"

RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install --upgrade setuptools
RUN python3.6 -m pip install tensorflow-gpu==1.5.0 Pillow scipy==1.1.0 scikit-learn matplotlib==2.1.2 tqdm
RUN ln -sf /usr/bin/python3.6 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

ENV USER docker

RUN echo "${USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers.d/${USER}
RUN chmod u+s /usr/sbin/useradd \
   && chmod u+s /usr/sbin/groupadd
ENV HOME /home/${USER}
ENV SHELL /bin/bash
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV TERM xterm-256color

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

WORKDIR /workspace
