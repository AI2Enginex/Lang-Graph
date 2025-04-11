# Import TypedDict from typing_extensions.
# TypedDict allows us to specify the expected structure of a dictionary with fixed keys and their types.
from typing_extensions import TypedDict

# Define a custom type 'Variables' using TypedDict.
# This type ensures that any dictionary passed to the functions must contain two integer keys: 'num1' and 'num2'.
class Variables(TypedDict):
    num1: int  # First number (integer)
    num2: int  # Second number (integer)
    var: str   # string input
    sequence: list # list input

# Define a class that performs basic arithmetic operations.
class ArathematicOperations:

    # Constructor method for the class. Currently, it does nothing but is defined for future extensibility.
    def __init__(self):
        pass

    # Method to add two numbers from the input dictionary.
    def add_data(self, parameters: Variables):
        try:
            # Access 'num1' and 'num2' from the input and return their sum.
            return parameters['num1'] + parameters['num2']
        except Exception as e:
            # In case of any error (e.g., missing keys), print an error message and return the error.
            print("function returns the following error")
            return e

    # Method to subtract the second number from the first number.
    def subtract_data(self, parameters: Variables):
        try:
            return parameters['num1'] - parameters['num2']
        except Exception as e:
            print("function returns the following error")
            return e

    # Method to multiply the two numbers.
    def multiply_data(self, parameters: Variables):
        try:
            return parameters['num1'] * parameters['num2']
        except Exception as e:
            print("function returns the following error")
            return e

    # Method to divide the first number by the second.
    def divide_data(self, parameters: Variables):
        try:
            return parameters['num1'] / parameters['num2']
        except Exception as e:
            print("function returns the following error")
            return e

    # Method to get the remainder when the first number is divided by the second.
    def get_remainder(self, parameters: Variables):
        try:
            return parameters['num1'] % parameters['num2']
        except Exception as e:
            print("function returns the following error")
            return e

# Placeholder class for string-related operations.
# Currently not implemented but defined for future expansion of the program.
class StringOperations:
    pass

# Placeholder class for list-related operations.
# Like StringOperations, it's a stub for future methods that may handle lists.
class ListOperations:
    pass

# This is the main entry point of the script.
# Code inside this block only runs when the script is executed directly.
if __name__ == '__main__':

    # Create an instance of the ArathematicOperations class to access its methods.
    a = ArathematicOperations()

    # Call the add_data method with a dictionary containing 'num1' and 'num2'.
    # The method returns the sum of the two numbers.
    add_result = a.add_data({'num1': 20, 'num2': 40})

    # Call the multiply_data method with the same input to get the product.
    multiplication_result = a.multiply_data({'num1': 20, 'num2': 40})

    # Print the result of the multiplication.
    print(multiplication_result)
