import numpy as np

class Dataloader():
    def __init__(self, data: np.ndarray, batch_size=16, shuffle=True) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data = data
        self.data_length = len(data)

    def __iter__(self):
        current = 0
        if self.shuffle:
            unique_numbers = np.random.choice(range(0, self.data_length),
                                                size=self.data_length, 
                                                replace=False)
        else:
            unique_numbers = np.arange(0, self.data_length)
            
        while current < self.data_length:
            yield self.data[unique_numbers[current:current + self.batch_size]]
            current += self.batch_size


    
    