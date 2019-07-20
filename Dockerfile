FROM chaff800/opencv-tess-lept

WORKDIR /rob

COPY . /rob 

ENV DISPLAY=${DISPLAY}

ARG FLAGS="MORE_FLAGS=\"-D DEBUG -D WAIT\""

#MORE_FLAGS="-D DEBUG -D WAIT"
RUN make 
RUN make test

CMD ["make",  "run"]