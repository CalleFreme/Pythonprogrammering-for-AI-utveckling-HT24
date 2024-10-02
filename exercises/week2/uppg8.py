## Uppgift 8: Skapa en enkel todo-lista

'''
Implementera en klass `TodoList` med metoder för att:

* Lägga till uppgifter
* Markera uppgifter som slutförda
* Visa alla uppgifter
* Visa endast oavslutade uppgifter
'''

class TodoList:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        self.tasks.append({"task": task, "completed": False})
    
    def mark_completed(self, task_index):
        if 0 <= task_index < len(self.tasks):
            self.tasks[task_index]["completed"] = True
        else:
            print("Invalid task index")
    
    def show_all_tasks(self):
        for i, task in enumerate(self.tasks):
            status = "Completed" if task["completed"] else "Pending"
            print(f"{i+1}. {task['task']} - {status}")
    
    def show_pending_tasks(self):
        pending_tasks = [task for task in self.tasks if not task["completed"]]
        for i, task in enumerate(pending_tasks):
            print(f"{i+1}. {task['task']}")

# Testa TodoList class
todo = TodoList()
todo.add_task("Buy groceries")
todo.add_task("Do laundry")
todo.add_task("Clean the house")
todo.mark_completed(1)
todo.show_all_tasks()
print("\nPending tasks:")
todo.show_pending_tasks()

# Kommentarer:
# Vi använder en lista av dictionaries för att lagra uppgifter och deras status.
# Listan innehåller uppgifter (tasks)
# Tasks representeras som dictionaries. De har nycklar så som task och completed.
# mark_completed() metoden kontrollerar om index är giltigt innan den uppdaterar status.
# show_pending_tasks() använder en list comprehension för att filtrera uppgifter.