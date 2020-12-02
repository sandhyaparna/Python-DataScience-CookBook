-- Missing in where statement
where Var IS NULL

-- IN in where
where Var IN ('X','Y')

-- If a var has matching string - https://www.sqlservertutorial.net/sql-server-basics/sql-server-like/
where Var LIKE '%DIAB1%' -- Word DIAB is present in any part of the sentence variable
where Var LIKE 'DIAB1%' -- starts with DIAB1
where Var LIKE '%DIAB1' -- ends with DIAB1
where Var LIKE 't%s' -- starts with the letter t and ends with the letter s
where Var LIKE '_u%' -- underscore represents a single character. Returns rows where second character is u
where Var LIKE '[YZ]%' -- If first charcter is Y or Z. The square brackets with a list of characters e.g., [ABC] represents a single character that must be one of the characters specified in the list.
where Var LIKE '[A-E]%' -- If first charcter is in range of A throug E
where Var LIKE '[^A-E]%' -- Negation -  If first charcter is NOT in range of A throug E
where Var LIKE '%30!%%'  ESCAPE '!' -- If row has 30%. ! is used as escape character to find % within a var















