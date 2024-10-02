## Uppgift 19: Enkel dataanalys med Pandas

'''
I denna uppgift kan ni använda CSV-filen `sales_data.csv` som finns här i repot.
Skapa ett program som använder Pandas för att analysera en CSV-fil med försäljningsdata:

1. Läs in en CSV-fil med kolumner för datum, produkt och försäljningsbelopp.
2. Visa de första 5 raderna.
3. Beräkna total försäljning per produkt.
4. Beräkna genomsnittlig försäljning per månad.
5. Hitta den dag med högst total försäljning.
6. Hitta produkten med högst total försäljning.
7. Skapa ett enkelt linjediagram över försäljningen över tid med matplotlib.

Om du vill:
8. Programmet sparar diagrammet som en PNG-fil
5. Programmet skriver en sammanfattning av analysen till en ny textfil (du får bestämma vad analysen ska inkludera)

Använd klasser för att strukturera koden och inkludera felhantering för filoperationer.
'''

import pandas as pd
import matplotlib.pyplot as plt
import os

class SalesAnalyzer:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        self.df['Month'] = self.df['Date'].dt.to_period('M')

    def show_first_rows(self, n=5):
        print(f"Första {n} raderna:")
        print(self.df.head(n))

    def total_sales_per_product(self):
        return self.df.groupby('Product')['SalesAmount'].sum().sort_values(ascending=False)

    def average_sales_per_month(self):
        return self.df.groupby('Month')['SalesAmount'].mean()

    def day_with_highest_sales(self):
        return self.df.groupby('Date')['SalesAmount'].sum().idxmax()

    def product_with_highest_sales(self):
        return self.total_sales_per_product().index[0]

    def plot_sales_over_time(self):
        plt.figure(figsize=(12, 6))
        self.df.groupby('Date')['SalesAmount'].sum().plot(kind='line')
        plt.title('Total försäljning över tid')
        plt.xlabel('Datum')
        plt.ylabel('Försäljning')
        plt.tight_layout()
        
        # 8. Spara diagrammet som en PNG-fil
        plt.savefig('sales_over_time.png')
        plt.close()

    def generate_summary(self):
        summary = f"""Försäljningsanalys Sammanfattning
        
Total försäljning: ${self.df['SalesAmount'].sum():.2f}
Genomsnittlig daglig försäljning: ${self.df.groupby('Date')['SalesAmount'].sum().mean():.2f}
Dag med högst försäljning: {self.day_with_highest_sales()}
Produkt med högst total försäljning: {self.product_with_highest_sales()}

Topp 5 produkter efter försäljning:
{self.total_sales_per_product().head().to_string()}

Genomsnittlig försäljning per månad:
{self.average_sales_per_month().to_string()}
        """
        return summary

    def save_summary(self, filename='sales_summary.txt'):
        with open(filename, 'w') as f:
            f.write(self.generate_summary())

def main():
    try:
        analyzer = SalesAnalyzer('sales_data.csv')
        
        analyzer.show_first_rows()
        
        print("\nTotal försäljning per produkt:")
        print(analyzer.total_sales_per_product())
        
        print("\nGenomsnittlig försäljning per månad:")
        print(analyzer.average_sales_per_month())
        
        print(f"\nDag med högst försäljning: {analyzer.day_with_highest_sales()}")
        
        print(f"\nProdukt med högst total försäljning: {analyzer.product_with_highest_sales()}")
        
        analyzer.plot_sales_over_time()
        print("\nDiagram över försäljning över tid har sparats som 'sales_over_time.png'")
        
        analyzer.save_summary()
        print("\nEn sammanfattning av analysen har sparats i 'sales_summary.txt'")
        
    except FileNotFoundError:
        print("Fel: CSV-filen kunde inte hittas. Kontrollera filnamn och sökväg.")
    except Exception as e:
        print(f"Ett oväntat fel inträffade: {e}")

if __name__ == "__main__":
    main()

# Kommentarer:
# Vi har skapat en SalesAnalyzer klass för att strukturera koden och inkapsla funktionaliteten.
# Felhantering har lagts till för att hantera eventuella problem med filinläsning eller andra oväntade fel.
# Metoden plot_sales_over_time() har uppdaterats för att spara diagrammet som en PNG-fil.
# En ny metod generate_summary() skapar en textsammanfattning av analysen.
# save_summary() metoden sparar denna sammanfattning till en textfil.
# main() funktionen orchestrerar hela analysprocessen och hanterar eventuella fel.