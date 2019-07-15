FROM chaff800/opencv-tess-lept

WORKDIR /rob

COPY . /rob 

ENV DISPLAY=${DISPLAY}

RUN make MORE_FLAGS="-D DEBUG -D WAIT"

CMD ["make",  "run"]