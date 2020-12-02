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

-- Join string values using comma or a seperator within a group mentioned in Group by clause in the order of product name
string_agg(product,',') WITHIN GROUP (ORDER BY product) as products

-- Creating new variable or CASE statement
(case 
    when operation='Buy' then -1*price -- 
    else price
END) as New_Var

-- self looping employee-Manager
-- keep doing left join based on the needed number of times

-- when using order by with in sub query using sql server/T-sql
-- Find the name of the user who has rated the greatest number of movies.
-- In case of a tie, return lexicographically smaller user name.
-- Find the movie name with the highest average rating in February 2020.
-- In case of a tie, return lexicographically smaller movie name.
select * from (select Top 1 B.name as results from Movie_Rating as A LEFT JOIN Users as B On A.user_id=B.user_id group by B.name order by count(DISTINCT movie_id) DESC, B.name) as X
UNION ALL
select results from (select Top 1 B.title as results,MONTH(created_at) as Month,YEAR(created_at) as Year from Movie_Rating as A LEFT JOIN Movies as B  On A.movie_id=B.movie_id  group by B.title,MONTH(created_at),YEAR(created_at) having MONTH(created_at)=2 AND YEAR(created_at)=2020 order by avg(cast(A.rating as float)) DESC, B.title) as Y

-- To get floating number/value when applying aggregate functions
cast(A.rating as float)

-- Select top n rows
select Top 1 * from Table

-- to check consecutive values within a variable as in For loop. Use LEAD for next value
-- Use LAG for previous value
-- If we have to check for 3 consecutive values, calculate next and Second next. so we can compare current num, next num and second next num
LEAD(Num,1) OVER (ORDER BY ID) as NextNum -- LEAD of value in Num variable within ID variable by 1 row
LEAD(Num,2) OVER (ORDER BY ID) as NextNum -- LEAD of value in Num variable within ID variable by 2 rows --second next value
LAG(total_sale) OVER(ORDER BY year) AS previous_total_sale  -- LAG/previous value of total_sale within year variable









