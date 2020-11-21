### Min and Max values frm Dictionaries:
prices = {
   'ACME': 45.23,
   'AAPL': 612.78,
   'IBM': 205.55,
   'HPQ': 37.20,
   'FB': 10.75
}
# Find min and max price
min_price = min(zip(prices.values(), prices.keys()))
max_price = max(zip(prices.values(), prices.keys()))
print('min price:', min_price)
print('max price:', max_price)
# Sorted prices
print('sorted prices:')
prices_sorted = sorted(zip(prices.values(), prices.keys()))
for price, name in prices_sorted:
    print('    ', name, price)

### Determine the most/top common words in a list
words = [
   'look', 'into', 'my', 'eyes', 'look', 'into', 'my', 'eyes',
   'the', 'eyes', 'the', 'eyes', 'the', 'eyes', 'not', 'around', 'the',
   'eyes', "don't", 'look', 'around', 'the', 'eyes', 'look', 'into',
   'my', 'eyes', "you're", 'under']
from collections import Counter
word_counts = Counter(words)
top_three = word_counts.most_common(3)
print(top_three)
# outputs [('eyes', 8), ('the', 5), ('look', 4)]
# Example of merging in more words
morewords = ['why','are','you','not','looking','in','my','eyes']
word_counts.update(morewords)
print(word_counts.most_common(3))

### Extracting a subset from dictionary based on values or keys
# example of extracting a subset from a dictionary
from pprint import pprint
prices = {
   'ACME': 45.23,
   'AAPL': 612.78,
   'IBM': 205.55,
   'HPQ': 37.20,
   'FB': 10.75}
# Make a dictionary of all prices over 200
p1 = { key:value for key, value in prices.items() if value > 200 }
print("All prices over 200")
pprint(p1)
# Make a dictionary of tech stocks
tech_names = { 'AAPL', 'IBM', 'HPQ', 'MSFT' }
p2 = { key:value for key,value in prices.items() if key in tech_names }
print("All techs")
pprint(p2)










