import wrds
import pandas as pd

if __name__ == "__main__":
    # Establish a connection to WRDS
    db = wrds.Connection(wrds_username='doullia')  # It will ask you your username and password

    # List of PERMNO numbers for which you want to retrieve ticker names
    permno_list = [10001, 10002, 10003]  # Just a test before passing the 40k permno

    # Convert the list to a string format for SQL query
    permno_str = ', '.join(map(str, permno_list))

    # Define your query to retrieve ticker names associated with the given PERMNOs
    query = f"""
            SELECT permno, ticker
            FROM crsp.msenames
            WHERE permno IN ({permno_str})
            """

    # Execute the query and retrieve the data
    data = db.raw_sql(query)

    # Close the connection
    db.close()

    # Display the retrieved data
    print(data)
