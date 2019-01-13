# Python Script

'''
	It is the multiline comment.
	It can serve a lot of purpose in your code.
	It should be used appropriatey.
'''
import sys

print("Arguments: {}".format(sys.argv))	
print('------------------------------')
# Returns the list whose first element will be the name of the file and rest 
# will be the other command line argument(s) provided by the user
# Example: python python.py -k mod -y yatch
# list will be: ['python.py', '-k', 'mod', '-y', 'yatch']

# Variables
myname = 'Vivek' 
num1 = 7
num2 = 5
print("'{}' type: {}".format(myname,type(myname)))
print("'{}' type: {}".format(num1,type(num1)))

# Mathematical Operations
print('------------------------------')
print("Addition: {}".format(num1 + num2))
print("Subtraction: {}".format(num1 - num2))
print("Multiplication: {}".format(num1 * num2))
print("Division: {}".format(num1 / num2))
print("Exponentiation: {}".format(num1 ** num2))
print("Floor Division: {}".format(num1 // num2))
print("Remainder: {}".format(num1 % num2))
print('------------------------------')

# Strings: It is immutable
# '\' is an escape character
str1 = 'It ain\'t me.'
str2 = "It ain't me again."
#str1[1] = 'F' # << immutable
print("String 1: {}".format(str1))
print("String 2: {}".format(str2))
print(r"Raw String: C:\some\name") # 'r' is used to treat it as a raw string
str3 = """\
Return: Nothing
Argument: int, int, float"""
print("Multiline String- {}".format(str3))
#String Operations
print("Concatenation: {}".format(str1 + str2))
# It supports Zero-based, -1 based indexing and also slicing
print("Lenght of '{}': {}".format("string",len("string")))
print('------------------------------')

# Lists: It is a mutable, compound datatype
list1 = [3, 'name', 2, 4.7]
print("Compound list: {}".format(list1))
list1 = list1 + [1, 2, 4]
print("Compound list: {}".format(list1))
print("Lenght: {}".format(len(list1)))
print('------------------------------')
# It also supports zero based, -1 based indexing and slicing operations

# while Loop
a,b = 0, 1
# while Loop demonstration
print("Fibonacci Series: ",end="")
while a<20:
	print(a,end=" ")
	a, b = b, a+b
print()
