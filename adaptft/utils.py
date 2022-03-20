class CircularBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None for _ in range(buffer_size)]
        self.current_index = 0
        self.full = False

    def add(self, data):
        self.buffer[self.current_index] = data
        self.current_index += 1
        if not self.full:
            if self.current_index == self.buffer_size:
                self.full = True
        self.current_index %= self.buffer_size

    def get(self):
        # Warning... you definitely will be confused by the output if you have not filled the buffer.
        return self.buffer[self.current_index:] + self.buffer[:self.current_index]

    def is_full(self) -> bool:
        return self.full

