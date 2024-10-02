class BankAccount:
    def __init__(self, owner):
        self.owner = owner
        self.balance = 0
    
    def deposit(self, amount):
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def display_balance(self):
        print(f"Current balance for {self.owner}: ${self.balance:.2f}")

# Testa BankAccount class
account = BankAccount("Alice")
account.deposit(1000)
account.withdraw(500)
account.display_balance()

# Kommentarer:
# Denna klass implementerar grundläggande funktionalitet för ett bankkonto.
# Vi använder en konstruktor (med hjälp av __init__()) för att sätta ägaren och initialt saldo för ett konto-objekt.
# Metoderna deposit och withdraw kontrollerar att beloppen är giltiga innan de ändrar saldot.
# Genom att använda metoder (deposit och withdraw) för att ändra värden på attribut (balance), istället för att
# ändra värden direkt (t.ex. balance = 1000), så kan vi kontrollera hur attributets värde ska kunna ändras genom medotens logik.
# display_balance använder f-strings för att formatera utskriften snyggt.