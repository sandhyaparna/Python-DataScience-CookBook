# https://github.com/XD-DENG/SQL-exercise
# https://www.linkedin.com/posts/rakesh-thottathil-a67b6480_sql-cheat-sheet-activity-6807530589978181632-rv7w

### CTE
-- Recursive function to generate numbers between 1 to 50 
WITH   cte # Common table expression (disposable views)
AS     (SELECT 1 AS n -- anchor member
        UNION ALL
        SELECT n + 1 -- recursive member
        FROM   cte
        WHERE  n < 50 )-- terminator
SELECT n FROM   cte;

-- Recursive example to generate all weekday names: Monday, Tuesday, Wednesday etc
WITH cte_numbers(n, weekday) 
AS (
    SELECT  0, DATENAME(DW, 0)
    UNION ALL
    SELECT n + 1, DATENAME(DW, n + 1) FROM cte_numbers
    WHERE n < 6 )
SELECT  weekday
FROM cte_numbers;

-- Generating Firstday of all months between any 2 given dates
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

-- Generating all dates between 2 given dates
declare @pStartDate date = '01/01/2020'
declare @pEndDate date   = '12/31/2020'
;with DaysOfMonth as(
    select @pStartDate as [Day] --Creating varibale called Day
    union all
    (select DATEADD(day, 1, [Day]) from DaysOfMonth
    where DATEADD(day, 1, [Day]) < @pEndDate) )
select * from DaysOfMonth -- this statement is imp as runing only the code above it gives error
option (maxrecursion 0)


### Examples
-- self join
select *
from rest as A LEFT JOIN rest as B
ON A.x>B.x OR A.x<B.x (x of A is joined with all the other x)
-- self join using 2 variables
select A.x,A.y, B.x,B.y
from rest as A LEFT JOIN rest as B
ON (A.x<=B.x OR A.x>=B.x) AND (A.y<=B.y OR A.y>=B.y) -- where statement should be used along with this
where NOT(A.x=B.x AND A.y=B.y) -- Removing the same observation on either side

-- Use this self join 
ON p1.x != p2.x OR p1.y != p2.y

-- Join on x & y and remove the below observation
WHERE NOT (A.x=B.x and A.y=B.y)

-- Where this nor that but other
where condition=Other OR NOT (Condition=this OR condition=that)

-- Negation 2 conditions in where (year and month not equal to Feb 20202)
WHERE NOT ( EXTRACT(YEAR from A.rental_ts)=2020 AND EXTRACT(MONTH from A.rental_ts)=2 )
-- Instead use date less than or equal to 31st Jan and greater than or equal to March 1st
where rental_ts<='2020-01-31' AND rental_ts>='2020-03-01' -- Less Time complexity is involved

-- Generate numbers between 0 and N --sql server
SELECT DISTINCT number
FROM master..spt_values
WHERE number BETWEEN 0 AND N

-- Generate numbers between 0 & max value obtained from a subquery
SELECT DISTINCT number
FROM master..spt_values
WHERE number BETWEEN 0 AND (select Top 1 ISNULL(count(transaction_date),0). Max of count(transaction_date) is (select Top 1 ISNULL(count(transaction_date),0)   

-- self looping employee-Manager
-- keep doing left join based on the needed number of times

-- Second Highest value in MySQL
-- To generate actual 'null' in output write a select statment for the subquery table
SELECT
    IFNULL(
      (SELECT DISTINCT Salary
       FROM Employee
       ORDER BY Salary DESC
        LIMIT 1 OFFSET 1),
    NULL) AS SecondHighestSalary
    
-- when using order by with in sub query using sql server/T-sql
-- Find the name of the user who has rated the greatest number of movies.
-- In case of a tie, return lexicographically smaller user name.
-- Find the movie name with the highest average rating in February 2020.
-- In case of a tie, return lexicographically smaller movie name.
select * from (select Top 1 B.name as results 
                from Movie_Rating as A LEFT JOIN Users as B 
                On A.user_id=B.user_id 
                group by B.name 
                order by count(DISTINCT movie_id) DESC, B.name) as X -- After ORDER BY u can use LIMIT 1
UNION ALL
select results from (select Top 1 B.title as results,MONTH(created_at) as Month,YEAR(created_at) as Year 
                    from Movie_Rating as A LEFT JOIN Movies as B  
                    On A.movie_id=B.movie_id 
                    group by B.title,MONTH(created_at),YEAR(created_at) 
                    having MONTH(created_at)=2 AND YEAR(created_at)=2020 
                    order by avg(cast(A.rating as float)) DESC, B.title) as Y

-- to check consecutive values within a variable as in For loop. Use LEAD for next value & LAG for previous value

-- Creating weeknames manually for pivoting
-- This works even when any of the weekname is not in the data but when all weeknames are needed
pivot (max(overall_quantity) for WEEKNAME IN (MONDAY,TUESDAY,WEDNESDAY,THURSDAY,FRIDAY,SATURDAY,SUNDAY)) pvt

-- Case statement can also be used to pivot 
https://avaldes.com/pivot_using_case/

-- For runtime error use Top 100 in select statement etc

-- Use order by when using UNION ALL
select * from (select Top 2147483647 * from A order by Date) AB
UNION ALL 
select * from (select Top 2147483647 * from B order by Date) XY

-- To calculate duplicates in data. Use count(*) by the interested columns

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

-- Create INDEX for time efficiency
CREATE INDEX NewTableThatIsSorted ON OriginalTable(VarToBeSortedOn DESC);

-- Check for cost and run time (Time complexity) in MySQL
explain analyze select * from Sales -- Doesn't work in SQL server

-- Multiplication in case statement might give upto 2 decimals but to get integer
first use ROUND and use CAST(VAR as int)

-- to get true or false in CASE statement
-- use single quotes: 'true' or 'false'
-- use CAST(1 AS BIT) to get true  CAST(0 AS BIT) to get false

-- Calculate Median within each group
Order data within each group by increasing order of values
a)Select Top 50 percent of data within each group and select Max value
b)Select Bottom 50 percent of data within each group and select Min value
cal avg of a & b within each group

-- For decimal places 
Apply CAST(var as decimal(10,4)) at each var within sub queries
Var1+Var2/2.00 gives decimal places in the result

-- Numeric function 
sqrt
SQUARE 
id%2 for MOD(id,2)

-- select a random number : ceil( rand() * no of obs/max of id)







