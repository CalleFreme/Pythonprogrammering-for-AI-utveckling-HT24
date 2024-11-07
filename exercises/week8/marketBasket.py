# Association rule mining används ofta i retail och e-handel för att hitta
# mönster i transaktionell data, så som att identifiera items som ofta köps tillsammans
# Mål: Identifiera item pairs som ofta köps tillsammans, m.h.a association rules
# Metod: Apriori algorithm (från mlxtend.frequent_patterns)

# 1. Ladda in och preprocessa transaktions-data till one-hot encoded format
# 2. Använd Apriori-algoritmen för att hitta frekventa "itemsets" över en minimum support threshold
# 3. Generera association rules (med parametrar support, confidence, lift, etc.) baserat på dessa itemsets

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Simulera transaction data i one-hot encoded format
data = {
    'transaction_id': [1, 1, 2, 2, 3, 3, 3, 4, 5, 5],
    'item': ['apple', 'banana', 'apple', 'milk', 'banana', 'milk', 'bread', 'milk', 'apple', 'bread']
}
df = pd.DataFrame(data)

# Pivot data till one-hot encoded format, varje transaction som en rad, items som kolumner
basket = df.pivot_table(index='transaction_id', columns='item', aggfunc=lambda x: 1, fill_value=0)

# Applicera Apriori för att hitta frequent itemsets med minimum support 0.3
frequent_itemsets = apriori(basket, min_support=0.3, use_colnames=True)
print("Frequent Itemsets:\n", frequent_itemsets)

# Generera association rules med minimum confidence 0.5
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
print("\nAssociation Rules:\n", rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

