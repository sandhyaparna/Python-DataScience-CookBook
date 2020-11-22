### Old style
'We have %d %s containing %.2f gallons of %s' %(2, 'bottles', 2.5, 'milk')
'We have %d %s containing %.2f gallons of %s' %(5.21, 'jugs', 10.86763, 'juice')

### New Style
'Hello {} {}, it is a great {} to meet you at {}'.format('Mr.', 'Jones', 'pleasure', 5)
'Hello {} {}, it is a great {} to meet you at {} o\' clock'.format('Sir', 'Arthur', 'honor', 9)
'I have a {food_item} and a {drink_item} with me'.format(drink_item='soda', food_item='sandwich')
'The {animal} has the following attributes: {attributes}'.format(animal='dog', attributes=['lazy', 'loyal'])

### Regular Expressions
s1 = 'Python is an excellent language'
s2 = 'I love the Python language. I also use Python to build applications at work!'

import re
pattern = 'python'
# match only returns a match if regex match is found at the beginning of the string
re.match(pattern, s1)

# pattern is in lower case hence ignore case flag helps
# in matching same pattern with different cases
re.match(pattern, s1, flags=re.IGNORECASE)

# printing matched string and its indices in the original string
m = re.match(pattern, s1, flags=re.IGNORECASE)
print('Found match {} ranging from index {} - {} in the string "{}"'.format(m.group(0), 
                                                                            m.start(), 
                                                                            m.end(), s1))
                                                                            
# match does not work when pattern is not there in the beginning of string s2
re.match(pattern, s2, re.IGNORECASE)





