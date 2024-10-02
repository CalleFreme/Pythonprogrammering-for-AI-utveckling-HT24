## Uppgift 24: Avancerad - Textanalysverktyg

'''
Skapa ett textanalysverktyg som kombinerar filhantering, OOP, och funktionell programmering. Det ska kunna:

1. Läsa in en textfil
2. Räkna ord, meningar och stycken
3. Identifiera de vanligaste orden och fraserna
4. Beräkna läsbarhetsindex (Googla)
5. Generera en rapport med resultaten
'''

import re
from collections import Counter
import matplotlib.pyplot as plt

class TextAnalyzer:
    def __init__(self, filename):
        with open(filename, 'r', encoding='utf-8') as file:
            self.text = file.read()
        self.words = re.findall(r'\b\w+\b', self.text.lower())
        self.sentences = re.split(r'[.!?]+', self.text)
        self.paragraphs = self.text.split('\n\n')

    def word_count(self):
        return len(self.words)

    def sentence_count(self):
        return len(self.sentences)

    def paragraph_count(self):
        return len(self.paragraphs)

    def most_common_words(self, n=10):
        return Counter(self.words).most_common(n)

    def most_common_phrases(self, n=5, phrase_length=2):
        phrases = [' '.join(self.words[i:i+phrase_length]) for i in range(len(self.words)-phrase_length+1)]
        return Counter(phrases).most_common(n)

    def calculate_readability(self):
        word_count = self.word_count()
        sentence_count = self.sentence_count()
        syllable_count = sum(self.count_syllables(word) for word in self.words)
        
        # Flesch-Kincaid Grade Level
        fkgl = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        return fkgl

    @staticmethod
    def count_syllables(word):
        # Enkel implementation, kan förbättras för mer exakthet
        return len(re.findall(r'[aeiou]', word)) + 1

    def generate_report(self):
        report = f"""Textanalysrapport

Antal ord: {self.word_count()}
Antal meningar: {self.sentence_count()}
Antal stycken: {self.paragraph_count()}

Vanligaste ord:
{self.format_list(self.most_common_words())}

Vanligaste fraser:
{self.format_list(self.most_common_phrases())}

Läsbarhetsindex (Flesch-Kincaid Grade Level): {self.calculate_readability():.2f}
        """
        return report

    @staticmethod
    def format_list(items):
        return '\n'.join(f"{item[0]}: {item[1]}" for item in items)

    def plot_word_frequency(self):
        words, counts = zip(*self.most_common_words(20))
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts)
        plt.title("Top 20 vanligaste ord")
        plt.xlabel("Ord")
        plt.ylabel("Frekvens")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('word_frequency.png')
        plt.close()

def main():
    analyzer = TextAnalyzer('sample_text_txtanalyzer.txt')
    print(analyzer.generate_report())
    analyzer.plot_word_frequency()
    print("En grafisk representation av ordfrekvensen har sparats som 'word_frequency.png'")

if __name__ == "__main__":
    main()

# Kommentarer:
# Denna klass implementerar ett avancerat textanalysverktyg med flera funktioner:
# - Räknar ord, meningar och stycken
# - Identifierar vanligaste ord och fraser
# - Beräknar ett läsbarhetsindex (Flesch-Kincaid Grade Level)
# - Genererar en detaljerad rapport
# - Skapar en visualisering av ordfrekvensen
# Biblitoek 're' låter oss använda reguljära uttryck (regex) för att dela upp texten i ord och meningar enligt ett visst mönster.
# Counter från collections används för att räkna förekomster effektivt.
# matplotlib används för att skapa en visualisering av ordfrekvensen.