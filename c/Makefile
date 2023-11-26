CC = gcc 
CFLAGS = -Wall -Wextra -Werror -std=c99 -pedantic -O3 -lm -fopenmp

.PHONY: all clean

all: seamcarving

clean:
	rm -rf *o

seamcarving: seamcarving.o stb_image_write.h stb_image.h
	$(CC) -o $@ $< $(CFLAGS)
seamcarving.o: seamcarving.c 
	$(CC) -c $< -fopenmp
