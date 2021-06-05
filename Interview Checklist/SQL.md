### General Concepts
1. SELECT A.*, B.* EXCEPT(Var_k, Var_k6) FROM Table1 AS A LEFT JOIN TABLE2 as B ON 
2. SELECT DISTINCT, COUNT(DISTINCT Var)
3. LEFT JOIN, INNER JOIN, RIGHT JOIN, FULL JOIN ON (t.key = ot.key)
4. GROUP BY, WHERE, HAVING, ORDER BY ASC | DESC
5. UNION, UNION ALL, EXCEPT/MINUS, INTERSECT
6. LIMIT some_value
7. SELECT TOP 10 * FROM table
8. AND, OR, NOT, IN(val_1, ..., val_n), NOT IN, IS NULL, IS NOT NULL, BETWEEN val_1 AND val_2, =, !=/<>, >=, >, <, <=
9. ORDER BY column_list [ASC |DESC] OFFSET offset_row_count {ROW | ROWS} FETCH NEXT fetch_row_count {ROW | ROWS} ONLY  (OFFSET is to specify number of rows to skip before stating to return number of rows mentioned in FETCH clause) 
10. MYSQL: LIMIT 1 OFFSET 1
11. 10. AGG: AVG(col), SUM(col), COUNT(col), COUNT(DISTINCT col), MAX(col), MIN(col), VAR(col), STDEV(col), PERCENTILE_APPROX(col, p), collect_list(col)
12. COALESCE(col_1, ..., col_n), CONCAT(col_1, ..., col_n), ROUND(col, n), LOWER(col), UPPER(col), REPLACE(col, old, new), SUBSTR(col, start, length), LTRIM(col) / RTRIM(col) / TRIM(col), LTRIM(col), RTRIM(col), LENGTH(col), DATE_TRUNC(time_dimension, col_date), DATE_ADD(col_date, number_of_days)
14. STRING: LEFT, RIGHT, LENGTH, TRIM, POSITION, STRPOS, MID, CONCAT, LIKE '%val%', NOT LIKE, 
15. RANK(), DENSE_RANK(), ROW_NUMBER(), CUME_DIST, PERCENT_RANK
16. LEAD(col, n), LAG(col, n) OVER (PARTITION BY Var1 ORDER BY Var2, Var2)
17. Cumulative sum of 3 rows: sum(Var) OVER (PARTITION BY Var1 ORDER BY Var2, Var2 ROWS BETWEEN 2 PRECEEDING AND CURRENT ROW)
18. SUM(Var) OVER (PARTITION BY Var1 ORDER BY Var2, Var2 ROWS BETWEEN CURRENT ROW AND 3 FOLLOWING)
19. CAST(Var as float) INT, decimal(10,2), numeric(36,4), string, real, char, varchar, text, datetime CAST('2017-08-25' AS datetime)
20. WITH cte_1 AS ( SELECT ...) SELECT ... FROM ...
21. INSERT INTO table_name VALUES (value1, value2, value3, ...);
22. INSERT INTO t(col_list) VALUES (value_list1), (value_list2)
23. INSERT INTO t(col_list) SELECT column_list FROM t2
