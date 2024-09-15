# test_wiki_vectorization
Goal is to take a set of text, vectorize it and make it searchable by indexing it in a vector DB

## How to run
1. `docker-compose build` - build container
2. `docker-compose up -d` - run  container with Redis and Redis Insight

### Steps to run
1. Create virtual env in directory
2. Activate it: `source /venv/bin/activate`
3. Run `pip install -r requirements.txt`
4. Run step 4 to create index
5. Run step 5 to save the data from articles
6. Run step 6 to search something

### Try to improve searching quality
1. Try smaller chunks, not 1000 chars, but 250 to improve relevancy - not works
