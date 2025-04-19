# Import TypedDict from typing_extensions.
# TypedDict allows us to specify the expected structure of a dictionary with fixed keys and their types.
from typing_extensions import TypedDict,List,Optional

# Define a custom type 'Variables' using TypedDict.
# This type ensures that any dictionary passed to the functions must contain two integer keys: 'num1' and 'num2'.
class Variables(TypedDict, total=False):
    num1: int  # First number (integer)
    num2: int  # Second number (integer)
    var: str   # string input
    sequence: Optional[List[int]] = None # list input
    str_sequence: Optional[List[str]] = None  #list of strings

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

    def __init__(self):
        pass

    # Method for converting string
    # to lowercase using pydantic
    def make_lower(self,parameter: Variables):

        try:
            return parameter['var'].lower()
        except Exception as e:
            return e
    
    # Method for converting string
    # to uppercase using pydantic
    def make_upper(self,parameter: Variables):

        try:
            return parameter['var'].upper()
        except Exception as e:
            return e
    
    # Method for converting first character 
    # uppercase using pydantic of string to
    def make_capitalize(self,parameter: Variables):

        try:
            return parameter['var'].capitalize()
        except Exception as e:
            return e
        

    # Method for splitting a string using a delimeter
    # and returning a list using pydantic module
    def split_characters(self,parameter: Variables,delimiter: None):

        try:
            return parameter['var'].split(delimiter)
        except Exception as e:
            return e

# Placeholder class for list-related operations.
# Like StringOperations, it's a stub for future methods that may handle lists.
class ListOperations:

    def __init__(self):
        pass

    # Method for implementing list 
    # comprehension using TypeDict
    def list_comprehension(self, parameter: Variables):

        try:
            return [data ** 2 for data in parameter['sequence']]
        except Exception as e:
            return e
    
    # Method for implementing list 
    # of string operations using TypeDict
    def list_of_strings(self, parameter: Variables):

        try:
            return [data.lower() for data in parameter['str_sequence']]
        except Exception as e:
            return e

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

    ls = ListOperations()

    lst_result = ls.list_comprehension({'sequence': [1,2,3,4,5]})

    print(lst_result)


    str_list_ops = ls.list_of_strings({'str_sequence': ["HELLO","HII","HOW"]})
    print(str_list_ops)
    

    st = StringOperations()

    capitalize_result = st.make_capitalize({'var': "mumbai indians"})

    print(capitalize_result)

    split_result = st.split_characters({'var': "mumbai indians"},delimiter=" ")

    print(split_result)