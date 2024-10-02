# Skapa en klass `Stack` som implementerar en stack (sista in, första ut) datastruktur med metoderna:

# * `push(item)`: Lägger till ett element överst i stacken.
# * `pop()`: Tar bort och returnerar det översta elementet i stacken.
# * `peek()`: Returnerar det översta elementet utan att ta bort det.
# * `is_empty()`: Returnerar True om stacken är tom, annars False.

class Stack():

    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        raise IndexError("Stack is empty")
    
    def peek(self):
        if not self.is_empty():
            return self.items[-1]
        raise IndexError("Stack is empty")
    
    def is_empty(self):
        return len(self.items) == 0
    
    def size(self):
        return len(self.items)
    
# Testa Stack class
stack = Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())  # Should print 3
print(stack.peek())  # Should print 2
print(stack.size())  # Should print 2
print(stack.is_empty())  # Should print False

# Kommentarer:
# Stack är en typ av datastruktur, som fungerar som en "hög". En stack fungerar som en speciell typ av lista.
# Element i stacken läggs till och tas bort enligt last in, first out.
# Denna stack-implementering använder en lista som underliggande datastruktur.
# Vi implementerar standardoperationerna för en stack: push, pop, peek, is_empty, och size.
# Notera hur vi hanterar fallet när stacken är tom i pop() och peek() metoderna.

