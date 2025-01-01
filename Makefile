CC = mpicc
CFLAGS = -std=c99 -g -O3
LIBS = -lm

BIN = serial snake_ordered snake_split snake_alltoall

all: serial snake_ordered snake_split snake_alltoall

snake_alltoall: snake_alltoall.c 
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

snake_split: snake_split.c 
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

snake_ordered: snake_ordered.c 
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)

serial: snake_serial.c 
	$(CC) $(CFLAGS) -o $@ $< $(LIBS)
	
clean:
	$(RM) $(BIN)

