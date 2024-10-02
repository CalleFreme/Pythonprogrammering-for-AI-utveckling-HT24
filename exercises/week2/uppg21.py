## Uppgift 21: Medelsvår - Skapa en egen iterator

'''
Implementera en klass `FibonacciIterator` som genererar Fibonacci-sekvensen:

1. Använd `__iter__()` och `__next__()` metoder.
2. Låt iteratorn generera sekvensen upp till ett specificerat maxvärde.
3. Hantera `StopIteration` när sekvensen är klar.

'''

class FibonacciIterator:
    def __init__(self, max_value):
        self.max_value = max_value
        self.a, self.b = 0, 1

    def __iter__(self):
        # __iter__ gör objektet itererbart. D.v.s vi kan loopa över objektet.
        return self

    def __next__(self):
        # __next__ gör objektet 
        if self.a > self.max_value:
            raise StopIteration
        result = self.a
        self.a, self.b = self.b, self.a + self.b
        return result

# Använd iteratorn
fib = FibonacciIterator(100)
for num in fib:
    print(num, end=' ')

# Kommentarer:
# Vi skapar en iterator-klass som genererar Fibonacci-tal upp till ett maxvärde.
# __iter__() metoden returnerar self, vilket gör objektet itererbart.
# __next__() metoden genererar nästa tal i sekvensen och hanterar StopIteration. __next__() kallas varje gång vi loopar över objektet.
# I Python är ett objekt en itererbar (iterator) om det finns både en __iter__ och __next__-metod definierad. 
# Klassen håller reda på de två senaste talen (self.a och self.b) för att generera nästa.