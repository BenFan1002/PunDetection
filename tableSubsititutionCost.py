def getCosts():
    import numpy as np
    import pandas as pd

    # Initialize the substitute_costs matrix
    substitute_costs = np.ones((128, 128), dtype=np.float64)
    delete_costs = np.ones(128, dtype=np.float64)
    insert_costs = np.ones(128, dtype=np.float64)
    df = pd.read_excel('Prouncation Table.xlsx', header=0, index_col=0)

    # Iterate through the DataFrame to update the substitute_costs

    for row_key in df.index:
        if row_key == ".":  # Skip the space row
            continue
        for col_key in df.columns:
            # Convert the row and column labels to ASCII if they are single characters
            row_index = ord(str(row_key))
            col_index = ord(str(col_key))
            if col_key == ".":
                delete_costs[row_index] = df.at[row_key, col_key]
                insert_costs[col_index] = df.at[row_key, col_key]
                continue
            # Update the substitute_costs matrix
            substitute_costs[row_index, col_index] = df.at[row_key, col_key]
    return substitute_costs, delete_costs, insert_costs


substitute_costs, delete_costs, insert_costs = getCosts()
print(substitute_costs[ord("&"), [ord(str(4))]])
print(ord(str(4)))