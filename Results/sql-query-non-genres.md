- **SQL Query (For tests exlcuding genres)**
```SQL

SELECT 
    tm.imdb_id,
    tb.primaryTitle,
    tm.release_date,
    tm.certification,
    tb.runtimeMinutes as runtime,
    tm.budget,
    tm.revenue
FROM
    tmdb tm
        JOIN
    title_basics tb ON tm.imdb_id = tb.tconst
WHERE
    tm.budget > 0 AND tm.revenue > 0
        AND tm.certification IS NOT NULL
        AND tm.certification NOT LIKE 'NR'
        AND tm.certification NOT LIKE 'NC%';

```



- **Preview of data:**


|    | imdb_id   | primaryTitle                                           | release_date   | certification   |   runtime |    budget |   revenue |
|---:|:----------|:-------------------------------------------------------|:---------------|:----------------|----------:|----------:|----------:|
|  0 | tt0168629 | Dancer in the Dark                                     | 2000-06-30     | R               |       140 |  12500000 |  45600000 |
|  1 | tt0314412 | My Life Without Me                                     | 2003-03-07     | R               |       106 |   2500000 |  12300000 |
|  2 | tt0325980 | Pirates of the Caribbean: The Curse of the Black Pearl | 2003-07-09     | PG-13           |       143 | 140000000 | 655011224 |
|  3 | tt0266697 | Kill Bill: Vol. 1                                      | 2003-10-10     | R               |       111 |  30000000 | 180906076 |
|  4 | tt0338013 | Eternal Sunshine of the Spotless Mind                  | 2004-03-19     | R               |       108 |  20000000 |  72258126 |