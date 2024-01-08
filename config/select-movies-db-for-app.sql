SELECT tb.tconst, tb.primaryTitle, tb.startYear, tb.runtimeMinutes,g.Genre_Name, tmdb.imdb_id, 
tmdb.id, tmdb.budget, tmdb.revenue, tmdb.certification, tmdb.original_title, tmdb.poster_path, tmdb.tagline, 
tmdb.original_language, tmdb.release_date, tmdb.popularity, tmdb.vote_average, tmdb.vote_count 
FROM title_basics tb
JOIN tmdb ON tb.tconst = tmdb.imdb_id 
JOIN title_genres tg ON tg.tconst = tmdb.imdb_id 
JOIN genres g ON g.Genre_ID  = tg.Genre_ID 
WHERE tmdb.budget>0 AND tmdb.revenue>0
