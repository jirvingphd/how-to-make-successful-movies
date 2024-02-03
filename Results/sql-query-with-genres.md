- **SQL Query (For tests including genre)**
```SQL

SELECT 
    tm.imdb_id,
    tb.primaryTitle,
    tm.budget,
    tm.revenue,
    g.genre_name
FROM
    tmdb tm
        JOIN
    title_basics tb ON tm.imdb_id = tb.tconst
        JOIN
    title_genres tg ON tg.tconst = tb.tconst
        JOIN
    genres g ON g.genre_id = tg.genre_id
WHERE
    tm.budget > 0 AND tm.revenue > 0
    AND g.genre_name IS NOT NULL
    AND g.genre_name NOT LIKE "News";

```



- **Preview of data:**
|    | imdb_id   | primaryTitle   |   budget |   revenue | genre_name   |
|---:|:----------|:---------------|---------:|----------:|:-------------|
|  0 | tt0035423 | Kate & Leopold | 48000000 |  76019048 | Comedy       |
|  1 | tt0035423 | Kate & Leopold | 48000000 |  76019048 | Fantasy      |
|  2 | tt0035423 | Kate & Leopold | 48000000 |  76019048 | Romance      |
|  3 | tt0118589 | Glitter        | 22000000 |   5271666 | Drama        |
|  4 | tt0118589 | Glitter        | 22000000 |   5271666 | Music        |