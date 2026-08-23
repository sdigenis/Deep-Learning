CC      = gcc
CFLAGS  = -Wall -O2
LDLIBS  = -lm

TARGET  = mlp
SRC     = mlp.c
DEPS    = data.c data.h nn_architecture.c nn_architecture.h

.PHONY: all run clean distclean

all: $(TARGET)

# mlp.c #includes data.c and nn_architecture.c directly,
# so the whole program is a single translation unit.
$(TARGET): $(SRC) $(DEPS)
	$(CC) $(CFLAGS) -o $@ $(SRC) $(LDLIBS)

run: $(TARGET)
	./$(TARGET)

# Remove the binary but keep the generated datasets.
clean:
	rm -f $(TARGET)

# Also drop the generated datasets; the next run regenerates them.
distclean: clean
	rm -f train_data.txt test_data.txt
