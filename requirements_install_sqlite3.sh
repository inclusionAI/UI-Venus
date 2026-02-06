# Navigate to a temporary directory
cd /tmp

# Remove any old downloads to start fresh
rm -rf sqlite pysqlite3

# Download the source code for SQLite and pysqlite3
git clone https://github.com/sqlite/sqlite.git
git clone https://github.com/coleifer/pysqlite3.git

# Go into the SQLite directory and prepare the source code
cd sqlite
./configure

# Compile the SQLite source code with FTS4 enabled
make sqlite3.c CFLAGS="-DSQLITE_ENABLE_FTS3 -DSQLITE_ENABLE_FTS4"

# Go to the pysqlite3 directory
cd ../pysqlite3

# Copy the compiled SQLite source code into the pysqlite3 directory
cp ../sqlite/sqlite3.c .
cp ../sqlite/sqlite3.h .

# Build and install the pysqlite3 library
python setup.py build_static
python setup.py install