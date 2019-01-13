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
list1.append(6)
list1.extend(['come','home'])
list1.insert(1,'okay')
#list1.remove() # first occurence of x
list1.pop() # last item or element at that index
print("Compound list: {}".format(list1))
list1 = list1 + [1, 2, 4]
print("Compound list: {}".format(list1))
print("Lenght: {}".format(len(list1)))
list1.clear()
# .count(x), .sort(key,reverse), .reverse, .copy
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

# if - elif - else statements
x = int(input("Enter a number: "))
if x>0:
	print("{} is greater than zero".format(x))
elif x<0:
	print("{} is less than zero".format(x))
else:
	print("{} is equal to zero".format(x))

# for Loop
print("List: ",end="")
for val in list1:
	print(val,end=" ")
print()
print("List: ",end="")
for idx in range(len(list1)):
	print(list1[idx],end=" ")
print()
print("List: ",end="")
for idx, val in enumerate(list1):
	print("({},{})".format(idx,val),end=" ")
print()
print('------------------------------')
# Another variation of range are: range(start,end), range(start, end, increment)
# list(range(4)) = [0, 1, 2, 3]
# break and continue statements as in C++ and C

# pass Statement
def setnum():
	pass
class abstract_class:
	pass

# Functions
def fib(n):	# Returns None
	a, b = 0, 1
	while a < n:
		print(a,end=" ")
		a, b = b, a+b
	print()

def fib_variation(n):
	res = []
	a, b = 0, 1
	while a < n:
		res.append(a)
		a, b = b, a+b
	return res

#print(fib_variation(10))
# default arguments, keyword arguments
def arguments(kind, *args, **kwargs):
	print("Arguments: ")
	print("-> Your kind:",kind)
	for arg in args:
		print('->', arg)
	for k, v in kwargs.items():
		print('->',k,':',v)

arguments("human", "I am a student", "I live in Dhanbad", fav_animal="dog")

def concatenate(*args, sep=":"):
	return sep.join(args)

print(concatenate("vivek","tiwari",sep = '.'))
# fun_name.__doc__ and fun_name.__annotations__
def make_incrementor(n):
	return lambda x: x + n
f = make_incrementor(10)
print('------------------------------')

# List comprehensions
squares = [x**2 for x in range(1,10)]
l = [(x,y) for x in range(1,5) for y in range(2,5) if x != y]
freshfruit = ['  banana', '  loganberry ', 'passion fruit  ']
fruits = [fruit.strip() for fruit in freshfruit]
k = [-4,-3,-2,-1,0]
k = [abs(i) for i in k]

from math import pi
pi_list = [round(pi,i) for i in range(1,5)]
print(pi_list)

matrix = [
	[1,11,43],
	[2,7,4],
	[3,0,8],
]
print(list(zip(*matrix)))

# Tuples: Immutable
# Set, Dictionaries

import json
l = [2,3,4,5]
f = open("myfile.txt",'r')
#json.dump(l,f)
#f.close()
x = json.load(f)
