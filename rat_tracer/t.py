from itertools import chain

values = [["a"], ["b"], ["c"]]
a = list(chain.from_iterable(values)) # works fine
print(a) # ['a', 'b', 'c']
b = sum(values, []) # TypeError: unsupported operand type(s) for +: 'int' and 'list'
print(b)