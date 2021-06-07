### General Concepts
1. SELECT A.*, B.* EXCEPT(Var_k, Var_k6) FROM Table1 AS A LEFT JOIN TABLE2 as B ON A.var2=B.var2
   * SELECT * from rental as A LEFT JOIN inventory as B ON A.inventory_id=B.inventory_id LEFT JOIN film as C On B.film_id=C.film_id
2. SELECT DISTINCT, COUNT(DISTINCT Var), 
   * CASE </br>
    WHEN condition1 THEN result1 </br>
    WHEN condition2 THEN result2 </br>
    WHEN conditionN THEN resultN </br>
    ELSE result </br>
    END AS VAR_NAME; </br>
3. LEFT JOIN, INNER JOIN, RIGHT JOIN, FULL JOIN ON (t.key = ot.key)
4. GROUP BY, WHERE, HAVING, ORDER BY ASC | DESC
5. UNION - selects only distinct values, UNION ALL - selects all the values, EXCEPT/MINUS, INTERSECT
6. LIMIT some_value
7. SELECT TOP 10 * FROM table; SELECT TOP 10 percent FROM table
8. AND, OR, NOT, IN(val_1, ..., val_n), where Var IN ('X','Y'), NOT IN, IS NULL, IS NOT NULL, BETWEEN val_1 AND val_2, =, !=/<>, >=, >, <, <=
    * Price BETWEEN 10 AND 20
    * OrderDate BETWEEN '1996-07-01' AND '1996-07-31'
    * COALESCE(col_1, val_if_null), ISNULL((col_1, val_if_null), IFNULL((col_1, val_if_null),
    * DateVar<'2020-02-01' ## YYYY-MM-DD
9. WHERE col ANY (SELECT ...); WHERE col ALL (SELECT ...)
10. ORDER BY column_list [ASC |DESC] </br> OFFSET offset_row_count {ROW | ROWS} FETCH NEXT fetch_row_count {ROW | ROWS} ONLY (OFFSET is to specify number of rows to skip before stating to return number of rows mentioned in FETCH clause) 
11. MYSQL: LIMIT 1 OFFSET 1
12. AGG: AVG(col), SUM(col), COUNT(col), COUNT(DISTINCT col), MAX(col), MIN(col), VAR(col), STDEV(col), PERCENTILE_APPROX(col, p), collect_list(col)
    * STRING_AGG(FirstName,'-') All observation within FirstName column are concatenated using - symbol; https://www.sqlshack.com/string_agg-function-in-sql/
    * STRING_AGG(product,',') WITHIN GROUP (ORDER BY product) as products: Join string values using comma or a seperator within a group mentioned in Group by clause in the order of product name
13. STRING: LEFT, RIGHT, LENGTH, TRIM, POSITION, STRPOS, MID, CONCAT, LIKE '%val%', NOT LIKE
14. STRING matching https://www.sqlservertutorial.net/sql-server-basics/sql-server-like/
    * LIKE 'a%'	Finds any values that start with "a"
    * LIKE '%a'	Finds any values that end with "a"
    * LIKE '%or%'	Finds any values that have "or" in any position
    * LIKE '_r%'	Finds any values that have "r" in the second position
    * LIKE 'a_%'	Finds any values that start with "a" and are at least 2 characters in length
    * LIKE 'a__%'	Finds any values that start with "a" and are at least 3 characters in length
    * LIKE 'a%o'	Finds any values that start with "a" and ends with "o"
    * LIKE '[YZ]%' -- If first charcter is Y or Z. The square brackets with a list of characters e.g., [ABC] represents a single character that must be one of the characters specified in the list
    * LIKE '[A-E]%' -- If first charcter is in range of A throug E
    * LIKE '[^A-E]%' -- Negation -  If first charcter is NOT in range of A throug E
    * LIKE '%30!%%'  ESCAPE '!' -- If row has 30%. ! is used as escape character to find % within a var
15. Wild cards
    * %	Represents zero or more characters	Example: bl% finds bl, black, blue, and blob
    * _	Represents a single character	Example: h_t finds hot, hat, and hit
    * []	Represents any single character within the brackets	Example: h[oa]t finds hot and hat, but not hit
    * ^	Represents any character not in the brackets	Example: h[^oa]t finds hit, but not hot and hat
    * -	Represents a range of characters	Example: c[a-b]t finds cat and cbt
16. DECLARE @variable_name datatype [ = initial_value ], @variable_name datatype [ = initial_value ], @variable_name datatype [ = initial_value ],
    * DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042'; Variable called @date
    * DECLARE @techonthenet VARCHAR(50) = 'Example showing how to declare variable';
    * DECLARE @numVar INT = 9
    * DECLARE @pStartDate date = '01/01/2020'
    * DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042';
17. RANK() - Generates dupliactes but misses the next numbers based on the duplicates, DENSE_RANK(), ROW_NUMBER(), NTILE(n): Creates n number of groups, CUME_DIST, PERCENT_RANK
    * LEAD(col, n), LAG(col, n) OVER (PARTITION BY Var1 ORDER BY Var2, Var2)
    * Cumulative sum of 3 rows: sum(Var) OVER (PARTITION BY Var1 ORDER BY Var2, Var2 ROWS BETWEEN 2 PRECEEDING AND CURRENT ROW)
    * SUM(Var) OVER (PARTITION BY Var1 ORDER BY Var2, Var2 ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING)
18. CAST(Var as float) INT, decimal(10,2), numeric(36,2), numeric(36,4), string, real, char, varchar, text, datetime CAST('2017-08-25' AS datetime)
    * cast should be applied on all numbers: cast(cast(count(DISTINCT q.player_id) as float)/cast(count(DISTINCT p.player_id) as float) as numeric(36,2)) as Day1_retention
23. WITH cte_1 AS ( SELECT ...) SELECT ... FROM ...
24. 
25. INSERT INTO table_name VALUES (value1, value2, value3, ...);
26. INSERT INTO t(col_list) VALUES (value_list1), (value_list2)
27. INSERT INTO t(col_list) SELECT column_list FROM t2
28. Comments: -- ; Multi-line: /*  */
29. SQL STRING FUNCTIONS 
    * CONCAT(col_1, ..., col_n)  OR 'W3Schools' + '.com' OR  "I am" || "Sandy"
    * CONCAT_WS('.', 'www', 'W3Schools', 'com')  Add strings together. Use '.' to separate the concatenated string values
    * LEN(Var)
    * SUBSTRING('SQL Tutorial', 1, 3)  Extract 3 characters from a string, starting in position 1
    * CHARINDEX('t', 'Customer') Search for "t" in string "Customer", and return position
    * PATINDEX('%schools%', 'W3Schools.com')
    * POSITION('A' IN descript) -- First position in descript column where A appears
    * STRPOS(descript, 'A') -- Similar to POSITION but not needed to use IN 
    * LEFT('SQL Tutorial', 3) | RIGHT('SQL Tutorial', 3)
    * LTRIM(col) / RTRIM(col) / TRIM(col)
    * LOWER(col) / UPPER(col)
    * REPLACE(string, old_string, new_string) REPLACE('SQL Tutorial', 'T', 'M') 
    * REPLICATE(string, integer) repeats a string a specified number of times
    * REVERSE('SQL Tutorial')
    * STR(185) Return a number as a string
    * FORMAT(value, format, culture)  https://docs.microsoft.com/en-us/sql/t-sql/functions/format-transact-sql?view=sql-server-ver15
      * FORMAT(number, decimal_places) # coverts to a number with specified number of decimal places
      * DECLARE @date datetime2 = '2018-06-02 08:24:14.3112042';
      * FORMAT(@date, 'd ') AS d,   -- 2
      * FORMAT(@date, 'dd') AS dd, -- 02
      * FORMAT(@date, 'ddd') AS ddd, -- Sat
      * FORMAT(@date, 'dddd') AS dddd; -- Saturday
      * FORMAT(@date, 'M ') AS M, -- 6
      * FORMAT(@date, 'MM') AS MM, -- 06
      * FORMAT(@date, 'MMM') AS MMM, -- Jun
      * FORMAT(@date, 'MMMMM') AS MMMM; --  June
      * FORMAT(@date, 'y ') AS y, -- 18
      * FORMAT(@date, 'yy') AS yy, -- 18
      * FORMAT(@date, 'yyy') AS yyy, -- 2018
      * FORMAT(@date, 'yyyy') AS yyyy, -- 2018
      * FORMAT(@date, 'yyyyy') AS yyyyy;  --  02018
      * FORMAT(@date, 'd ') AS 'Space', --  2
      * FORMAT(@date, 'd') AS 'No Space', -- 6/2/2018
      * FORMAT(@date, 'M ') AS 'Space', -- 6
      * FORMAT(@date, 'M') AS 'No Space', -- June 2
      * FORMAT(@date, 'y ') AS 'Space', -- 8
      * FORMAT(@date, 'y') AS 'No Space';  -- June 2018
      * FORMAT (getdate(), 'dd/MM/yyyy ') as date	21/03/2018
      * FORMAT (getdate(), 'dd/MM/yyyy, hh:mm:ss ') as date	21/03/2018, 11:36:14
      * FORMAT (getdate(), 'dddd, MMMM, yyyy') as date	Wednesday, March, 2018
      * FORMAT (getdate(), 'MMM dd yyyy') as date	Mar 21 2018
      * FORMAT (getdate(), 'MM.dd.yy') as date	03.21.18
      * FORMAT (getdate(), 'MM-dd-yy') as date	03-21-18
      * FORMAT (getdate(), 'hh:mm:ss tt') as date	11:36:14 AM
      * FORMAT (getdate(), 'd','us') as date    
      * FORMAT(p.pay_date,'yyyy-MM') as pay_month
30. SQL Math
    * Abs(-243.5)
    * FLOOR(25.75)
    * RAND(6): a random decimal number (with seed value of 6)
    * ROUND(235.415, 2)
31. SQL DateTime
    * DAY(@date) AS DAY
    * MONTH(@date) AS MONTH
    * YEAR(@date) AS YEAR
    * DATEADD(year, 1, '2017/08/25')  Adding 1 year; Negative number implies subtraction  DATEADD(interval, number, date)
    * DATEDIFF(interval, startdate, enddate )
    * DATENAME(interval, DateVar)  returns result i.e year/month, etc as CHARACTER string. DATENAME(dw, DateVar) -- Returns Weekname i.e Sunday, Monday etc
    * DATEPART: DATEPART (interval, dateVar)  It is a Datetime function which helps to extract information from date. This function always returns result as integer type
    * EOMONTH(DateVar) as LastDayofMonth. EOMONTH('2007-10-25') --Gives '2007-10-31'
    * Intervals:
      * year, yyyy, yy = Year
      * quarter, qq, q = Quarter
      * month, mm, m = month
      * dayofyear, dy, y = Day of the year
      * day, dd, d = Day
      * week, ww, wk = Week
      * weekday, dw, w =Weekname Returns Weekname i.e Sunday, Monday etc
      * hour, hh = hour
      * minute, mi, n = Minute
      * second, ss, s = Second
      * millisecond, ms = Millisecond
      * microsecond, mcs
      * nanosecond, ns
      * TZoffset, tz	+05:10
      * ISO_WEEK, ISOWK, ISOWW	44
    * 

    *   ROUND(col, n), REPLACE(col, old, new), SUBSTR(col, start, length),  DATE_TRUNC(time_dimension, col_date), DATE_ADD(col_date, number_of_days)
