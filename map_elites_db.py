#
# A map elites implementation backed by a relational database
#

import random
import sqlite3
import json
from math import floor, ceil, log
from scipy.stats.qmc import LatinHypercube
from scipy.interpolate import interp1d

from numpy import ndarray



#
# Init table queries
#
create_bin_table_sql = "CREATE TABLE IF NOT EXISTS population_bin (bin_id INTEGER PRIMARY KEY AUTOINCREMENT, bin_code TEXT NOT NULL UNIQUE)"

trunc_bin_table_sql = "DELETE FROM population_bin"

create_sample_table_sql = """CREATE TABLE IF NOT EXISTS population_sample (
    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
    bin_id INTEGER NOT NULL,
    input_code TEXT NOT NULL UNIQUE,
    result TEXT,
    fitness NUMBER NOT NULL
)"""

trunc_sample_table_sql = "DELETE FROM population_sample"


#
# CRUD Queries
#

insert_bin = "INSERT INTO population_bin (bin_code) VALUES (?)"

insert_sample = "INSERT INTO population_sample (bin_id, input_code, result, fitness) VALUES (?, ?, ?, ?)"

select_bin_report = """SELECT 
    bin.bin_id,
    bin_code,
    sample_id,
    input_code,
    result,
    MAX(fitness)
    FROM population_sample AS sample
        INNER JOIN population_bin AS bin using (bin_id)
    GROUP BY bin_id ORDER BY MAX(fitness) DESC
"""

select_bin = "SELECT bin_id FROM population_bin WHERE bin_code = ?"

select_input = "SELECT sample_id FROM population_sample WHERE input_code = ?"

#
# Core functions
#


def encode_list(bin_code):
    if isinstance(bin_code, ndarray):
        return json.dumps(bin_code.tolist())
    else:
        return json.dumps(bin_code)

def decode_list(bin_code):
    return json.loads(bin_code)

class DBReaderSqlite():
    def __init__(self, db_con):
        self.db_cur = db_con.cursor()

    def elites(self, limit=None):
        db_cur = self.db_cur
        if not limit:
            db_cur.execute(select_bin_report)
        else:
            db_cur.execute(select_bin_report + " LIMIT ?", [limit])
        return db_cur.fetchall()


class MapElites():
    def __init__(self, spaces, db_con, num_buckets=10, resume_search=None):
        # Describes the number of buckets per dimension
        self.num_buckets = num_buckets
        self.spaces = spaces
        self.db_cur = db_con.cursor()
        self.default_search_scale = self.num_buckets ** ceil(log(len(self.spaces)))

        self.db_reader = DBReaderSqlite(db_con)
        # A function which closes over the given db_conenction and returns a closure. When the closure is called, it creates a DBReader object with the connection
        lazy_reader_on = lambda connection : lambda : DBReaderSqlite(connection)
        self.lazy_db_reader_factory = lazy_reader_on(db_con)

        # DB init
        db_cur = self.db_cur
        db_cur.execute(create_bin_table_sql)
        db_cur.execute(create_sample_table_sql)

        if not resume_search:
            db_cur.execute(trunc_bin_table_sql)
            db_cur.execute(trunc_sample_table_sql)

        db_cur.connection.commit()


    def input_to_bin_code(self, input):
        bin_code = []
        for input, bound in zip(input, self.spaces):
            if input < bound[0]:
                bin_code.append(0)
            elif input > bound[1]:
                bin_code.append(self.num_buckets)
            else:
                bin_code.append(floor(input / (bound[1] / self.num_buckets)))
        return bin_code

    # Add a sample to a population bin, this will create the bin if it does not already exist
    def add_sample(self, input, fitness, sample):
        bin_code = encode_list(self.input_to_bin_code(input))
        db_cur = self.db_cur

        # Collect bin_id or create id if not exist
        bin_id = -1
        db_cur.execute(select_bin, [bin_code])
        id = self.db_cur.fetchone()
        if id:
            bin_id = id[0]
        else:
            db_cur.execute(insert_bin, [bin_code])
            bin_id = db_cur.lastrowid

        try:
            # print(insert_sample, (bin_id, encode_list(input), encode_list(sample), fitness))
            db_cur.execute(insert_sample, (bin_id, encode_list(
                input), encode_list(sample), fitness))
            db_cur.connection.commit()
            return db_cur.lastrowid
        except:
            print("Skipped duplicate input")
            db_cur.execute(select_input, [encode_list(input)])
            id = db_cur.fetchone()
            return id[0]

    def check_input_collision(self, input):
        self.db_cur.execute(select_input, [encode_list(input)])
        return self.db_cur.fetchone()
    
    def search_noise(self, process, fitness, limit=None):
        """Randomly distribute samples across the space to initialise the search."""
        limit = limit or self.default_search_scale
        sampler = LatinHypercube(d=len(self.spaces))
        for input in sampler.random(limit):
            for n, dim in enumerate(self.spaces):
                range_mapping = interp1d([0, 1], dim)
                input[n] = range_mapping(input[n])
            value = process(input)
            search.add_sample(input, fitness(value), value)

    def search_elites_weighted(self, process, mutatate, fitness, limit=None):
        limit = limit or self.default_search_scale
        for (rank, row) in enumerate(search.elites(limit=limit)):
            elite_input = decode_list(row[3])
            # Place more focus on current elite
            for _ in range(rank, limit):
                # print(noise)
                new_input = mutatate(elite_input, row[5], self.lazy_db_reader_factory)
                if not self.check_input_collision(new_input):
                    value = process(new_input)
                    search.add_sample(new_input, fitness(value), value)
    
    def search_elites(self, process, mutatate, fitness, mutations=None, limit=None):
        mutations = mutations or self.default_search_scale
        limit = limit or self.default_search_scale
        for row in search.elites(limit=limit):
            elite_input = decode_list(row[3])
            noise = (abs(MAX_RESULT - row[5]) * 0.0005) # Derrive a narrowing window of noise as the fitness apraoches the limit
            for _ in range(1, mutations):
                new_input = mutatate(elite_input, row[5], self.lazy_db_reader_factory)
                if not self.check_input_collision(new_input):
                    value = process(new_input)
                    search.add_sample(new_input, fitness(value), value)

    #
    # Delegate reader calls
    #

    # Select back the elites of each population in sorted high to low fitness
    def elites(self, limit=None):
        return self.db_reader.elites(limit=limit)


#
# Synopsis
#

# Conenct to a database (use ":memory:" for transient or "[filename].sqlite" for persistent storage)
db_con = sqlite3.connect(":memory:")

#
# Define process fitness and mutation functions
#
MAX_VALUE = 100

# Process an input vector into an output structure
def result_list(l):
    return [l[0] * l[1]]

MAX_RESULT = result_list([MAX_VALUE, MAX_VALUE])[0]

# Calculate a fitness value (float) for a given value (Input structure)
def fitness(result):
    # abs imported from math
    return MAX_RESULT / abs(42 - result[0])

def mutation_noise(fitness):
    return abs(MAX_RESULT - fitness) * 0.0005 # Derrive a narrowing window of noise as the fitness apraoches the limit

# Create a callable function to perform a genetic operation on the sample
def mutator(values, fitness, _):
    magnitude = mutation_noise(fitness)
    new_value = []
    for val in values:
        new_value.append(val + random.uniform(-magnitude, magnitude))
    return new_value


# Initialise the search with a list of tuples for bounding the x and y and more dimensions of the search
search = MapElites([(-MAX_VALUE, MAX_VALUE), (-MAX_VALUE, MAX_VALUE)], db_con)

# Init with a noise based search for braod coverage of the space
search.search_noise(result_list, fitness, limit=30)

print("Initial State:", search.elites(limit=10))
# print(f"Searching with scale of: {search.default_search_scale}")

# Run a sequence of searches to look broadly and then narrow the scope

# Go deep into the population pool
search.search_elites(result_list, mutator, fitness, mutations=3, limit=30)
# Equal energy search across the top of the pool
search.search_elites(result_list, mutator, fitness, mutations=5, limit=15)
# Weighted search on the top of the pool
search.search_elites_weighted(result_list, mutator, fitness, limit=10)
    

print("Final State:", search.elites(limit=10))