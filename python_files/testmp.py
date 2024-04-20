import multiprocessing
import concurrent.futures
from functools import partial

def square_number(x,y):
    return x ** 2 + y ** 2 

def test_mp():
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



def test_thread():
    # Define your list of numbers
    numbers = [1, 2, 3, 4, 5]

    # Create a ThreadPoolExecutor object within a context manager
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Map the square_number function to each number in the list using executor.map
        squared_numbers = list(executor.map(square_number, numbers))

    # Print the original numbers and their squares
    for num, squared_num in zip(numbers, squared_numbers):
        print(f"{num} squared is {squared_num}")

if __name__ == "__main__":
    test_thread()
