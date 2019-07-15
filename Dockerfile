FROM chaff800/opencv-tess-lept

WORKDIR /rob

COPY . /rob 

RUN make MORE_FLAGS="-D DEBUG -D WAIT"

CMD ["make",  "run"]