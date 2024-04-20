import multiprocessing
from functools import partial

def square_number(x,y):
    return x ** 2 + y ** 2 

def main():
    # Define your list of numbers
    numbers = [1, 2, 3, 4, 5]

    # Create a partial function with square_number
    square_number_partial = partial(square_number,y=1)

    # Create a Pool object within a context manager
    with multiprocessing.Pool(5) as pool:
        # Apply the partial function to each number in the list using Pool.map
        squared_numbers = pool.map(square_number_partial, numbers)

    # Print the original numbers and their squares
    for num, squared_num in zip(numbers, squared_numbers):
        print(f"{num} squared is {squared_num}")

if __name__ == "__main__":
    main()
