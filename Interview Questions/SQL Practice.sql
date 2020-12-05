e-- Missing in where statement
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

-- string cleaning functions
LEFT(string, number of characters)
RIGHT(string, number of characters)
LENGTH(string)
TRIM function is used to remove characters from the beginning and end of a string
POSITION allows you to specify a substring, then returns a numerical value equal to the character number (counting from left) where that substring first appears in the target string. 
POSITION('A' IN descript) -- First position in descript column where A appears
STRPOS(descript, 'A') -- Similar to POSITION but not needed to use IN 
UPPER 
LOWER 
SUBSTR(*string*, *starting character position*, *# of characters*)
CONCAT -- combine strings from several columns together
Instead of concat function use || to join strings -- eg: "I am" || "Sandy"


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
cast(working_percentage as numeric(36,2)) -- Creating 2 decimal places for 0 also i.e produces 0.00
FORMAT(number, decimal_places) # coverts to a number with specified number of decimal places
-- cast should be applied on all numbers
cast(cast(count(DISTINCT q.player_id) as float)/cast(count(DISTINCT p.player_id) as float) as numeric(36,2)) as Day1_retention

-- The CAST() function converts a value (of any type) into a specified datatype.
SELECT CAST(25.65 AS varchar);
SELECT CAST('2017-08-25' AS datetime);

-- Format https://docs.microsoft.com/en-us/sql/t-sql/functions/format-transact-sql?view=sql-server-ver15
-- Returns a value formatted with the specified format and optional culture
FORMAT( value, format [, culture ] )  
FORMAT(number, decimal_places) #Converts numbers to float with specified number of decimal places

-- Select top n rows
select Top 1 * from Table

-- to check consecutive values within a variable as in For loop. Use LEAD for next value
-- Use LAG for previous value
-- If we have to check for 3 consecutive values, calculate next and Second next. so we can compare current num, next num and second next num
LEAD(Num,1) OVER (PARTITION BY ID ORDER BY Num) as NextNum -- LEAD of value in Num variable within ID variable by 1 row
LEAD(Num,2) OVER (PARTITION BY ID ORDER BY Num) as NextNum -- LEAD of value in Num variable within ID variable by 2 rows --second next value
LAG(total_sale) OVER(ORDER BY year) AS previous_total_sale  -- LAG/previous value of total_sale within year variable

-- IS NULL to 0
COALESCE(spampostsremoved,0) -- COALESCE is used in MySQL
ISNULL(spampostsremoved,0) -- ISNULL is used in SQL server/T-sql

-- Creating weeknames manually for pivoting
-- This works even when any of the weekname is not in the data but when all weeknames are needed
pivot (max(overall_quantity) for WEEKNAME IN (MONDAY,TUESDAY,WEDNESDAY,THURSDAY,FRIDAY,SATURDAY,SUNDAY)) pvt

-- Case statement can also be used to pivot 
https://avaldes.com/pivot_using_case/

-- Date times DATENAME ( datepart , date )  -- returns result as CHARACTER string
DATENAME(dw, DateVar) -- Returns Weekname i.e Sunday, Monday etc
--  Example. If datepart is year or yyyy or yy mentioned, year of the date is returned
SELECT DATENAME(datepart,'2007-10-30 12:15:32.1234567 +05:10');
Here is the result set.
EXAMPLES
datepart	Return value
year, yyyy, yy	2007
quarter, qq, q	4
month, mm, m	10 --Gives month as Number
dayofyear, dy, y	303
day, dd, d	30
week, wk, ww	44
weekday, dw	Tuesday
hour, hh	12
minute, n	15
second, ss, s	32
millisecond, ms	123
microsecond, mcs	123456
nanosecond, ns	123456700
TZoffset, tz	+05:10
ISO_WEEK, ISOWK, ISOWW	44
 
-- Select data related variables. DATEPART() function was built specifically for returning specified parts of a date. 
-- Returns return as Integer
DAY(@date) AS DAY
MONTH(@date) AS MONTH
YEAR(@date) AS YEAR
DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042';
DATEPART(day, @date) AS DAY, -- 2
DATEPART(weekday, @date) AS WEEKDAY,-- 7
DATEPART(month, @date) AS MONTH,-- 6
DATEPART(year, @date) AS YEAR;-- 2018

-- FORMAT() 
DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042';
SELECT 
    FORMAT(@date, 'd ') AS d,   -- 2
    FORMAT(@date, 'dd') AS dd, -- 02
    FORMAT(@date, 'ddd') AS ddd, -- Sat
    FORMAT(@date, 'dddd') AS dddd; -- Saturday
    FORMAT(@date, 'M ') AS M, -- 6
    FORMAT(@date, 'MM') AS MM, -- 06
    FORMAT(@date, 'MMM') AS MMM, -- Jun
    FORMAT(@date, 'MMMMM') AS MMMM; --  June
    FORMAT(@date, 'y ') AS y, -- 18
    FORMAT(@date, 'yy') AS yy, -- 18
    FORMAT(@date, 'yyy') AS yyy, -- 2018
    FORMAT(@date, 'yyyy') AS yyyy, -- 2018
    FORMAT(@date, 'yyyyy') AS yyyyy;  --  02018
    FORMAT(@date, 'd ') AS 'Space', --  2
    FORMAT(@date, 'd') AS 'No Space', -- 6/2/2018
    FORMAT(@date, 'M ') AS 'Space', -- 6
    FORMAT(@date, 'M') AS 'No Space', -- June 2
    FORMAT(@date, 'y ') AS 'Space', -- 8
    FORMAT(@date, 'y') AS 'No Space';  -- June 2018
    
    
-- Generate numbers between 0 and N --sql server
SELECT DISTINCT number
FROM master..spt_values
WHERE number BETWEEN 0 AND N
                                              
-- Generate numbers between 0 & max value obtained from a subquery
SELECT DISTINCT number
FROM master..spt_values
WHERE number BETWEEN 0 AND (select Top 1 ISNULL(count(transaction_date),0)  
-- Max of count(transaction_date)
(select Top 1 ISNULL(count(transaction_date),0)  
                            
-- Declare variables
DECLARE @variable_name datatype [ = initial_value ],
@variable_name datatype [ = initial_value ],
@variable_name datatype [ = initial_value ],
;
-- Declare string variable with initial value
DECLARE @techonthenet VARCHAR(50) = 'Example showing how to declare variable'; --DECLARE @techonthenet VARCHAR(max) -- use max instead of mentioning a number
-- Declare date variable with initial value
declare @pStartDate date = '01/01/2020'
DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042'; -- declare datetime var

 
-- Add months, days to date variable  DATEADD(interval, number, date)
DATEADD(year, 1, '2017/08/25') -- Adding 1 year to '2017/08/25'
-- Negative number implies subtraction
year, yyyy, yy = Year
quarter, qq, q = Quarter
month, mm, m = month
dayofyear, dy, y = Day of the year
day, dd, d = Day
week, ww, wk = Week
weekday, dw, w = Weekday
hour, hh = hour
minute, mi, n = Minute
second, ss, s = Second
millisecond, ms = Millisecond
 
-- Difference between two dates DATEDIFF(datepart , startdate , enddate )
SELECT DATEDIFF(year,        '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(quarter,     '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(month,       '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(dayofyear,   '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(day,         '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(week,        '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(hour,        '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(minute,      '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(second,      '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(millisecond, '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
SELECT DATEDIFF(microsecond, '2005-12-31 23:59:59.9999999', '2006-01-01 00:00:00.0000000');
 
-- Generating FirstMonth of date between any 2 given dates
declare @pStartDate date = '01/01/2020'
declare @pEndDate date   = '12/31/2020'
;with FirstDayOfMonth as(
    select @pStartDate as [firstDayOfMonth]
    union all
-- Generating first dates of consecutive months
    (select DATEADD(month, 1, [firstDayOfMonth]) from FirstDayOfMonth
    where DATEADD(month, 1, [firstDayOfMonth]) < @pEndDate))
select * from FirstDayOfMonth -- this statement is imp as runing only the code above it gives error
option (maxrecursion 0)
 
-- End of Month i.e last day of the month based on a date
EOMONTH('2007-10-25') --Gives '2007-10-31'

-- Ranking methods https://codingsight.com/methods-to-rank-rows-in-sql-server-rownumber-rank-denserank-and-ntile/
ROW_NUMBER() OVER( ORDER BY Student_Score) --Generates 1,2,3,4 without duplicates or gaps even if Student_Score is same
-- Student_Score is 770,770,770, 885, 900, 900, 1001 below code generates 1,1,1,4,5,5,7
RANK ()  OVER( ORDER BY Student_Score) --Generates dupliactes but misses the next numbers based on the duplicates
DENSE_RANK() OVER(ORDER BY Student_Score) -- Student_Score is 770,770,770, 885, 900, 900, 1001  geenrates 1,1,1,2,3,3,4
NTILE(n) Creates n number of groups

-- Cumulative sum of last 3 rows
-- Cumulative sum of salary with id var ordered by Month within id and last 3 rows (current & previous 2 rows) are used to calculate cumulative sum
sum(Salary) over (PARTITION BY A.Id order by Month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as Salary
-- In below code current and previous 3 rows are used
sum(Salary) over (PARTITION BY A.Id order by Month ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) as Salary

-- Use order by when using UNION ALL
select * from (select Top 2147483647 * from A order by Date) AB
UNION ALL 
select * from (select Top 2147483647 * from B order by Date) XY

-- declare a variable and use it near Top. Declared variable should be in brackets
-- cal total scores of all players
DECLARE @Top int = (select count(*) as rows from Players);
DECLARE @Top int = (select max(rows) from (select count(*) as rows from Players) as xy); -- Extracting number of rows in Players table
select * from (select Top (@Top) * from A order by Date) AB

-- To calculate duplicates in data. Use count(*) by the interested columns

-- Current date time
NOW()
GETDATE()
SYSDATETIME()

-- Create temporary table # for temp, ## for global temp
DROP TABLE #student (to make sure #student is not there)
CREATE TABLE #student
(   id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    gender VARCHAR(50) NOT NULL,
    age INT NOT NULL,
    total_score INT NOT NULL, )
INSERT INTO #student 
VALUES (1, 'Jolly', 'Female', 20, 500), 
(2, 'Jon', 'Male', 22, 545), 
(3, 'Sara', 'Female', 25, 600), 
(4, 'Laura', 'Female', 18, 400), 
(5, 'Alan', 'Male', 20, 500), 
(6, 'Kate', 'Female', 22, 500), 
(7, 'Joseph', 'Male', 18, 643), 
(8, 'Mice', 'Male', 23, 543), 
(9, 'Wise', 'Male', 21, 499), 
(10, 'Elis', 'Female', 27, 400);
select * from #student

-- Difference between UNION and UNION ALL
UNION selects only distinct values
UNION ALL selects all the values

-- Not equal <>

-- Negation 2 conditions in where (year and month not equal to Feb 20202)
WHERE NOT ( EXTRACT(YEAR from A.rental_ts)=2020 AND EXTRACT(MONTH from A.rental_ts)=2 )
-- Instead use date less than or equal to 31st Jan and greater than or equal to March 1st
where rental_ts<='2020-01-31' AND rental_ts>='2020-03-01' -- Less Time complexity is involved

-- Join 3 or more tables 
SELECT *
from rental as A LEFT JOIN inventory as B ON A.inventory_id=B.inventory_id
LEFT JOIN film as C On B.film_id=C.film_id

-- less than for Date in where
where DateVar<'2020-02-01' ## YYYY-MM-DD format

-- WHERE clause can be used before GROUP BY clause to restrict number of rows of the table in FROM clause




