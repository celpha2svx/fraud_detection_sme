import pandas as pd
import pandas.api.types
import numpy as np
import warnings
import gc


warnings.filterwarnings('ignore')


def reduce_mem_usage(df):
    """Iterates through all the columns of a dataframe and downcasts its numeric types
       to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Initial Memory usage: {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        # Check if the column is NOT a generic object and NOT a categorical type
        is_numeric_type = col_type != object and not pd.api.types.is_categorical_dtype(col_type)

        if is_numeric_type:
            # We can safely calculate min/max only for numeric columns
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                # Integer downcasting logic
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                # Float downcasting to float16
                df[col] = df[col].astype(np.float16)

        elif col_type == object:
            # Convert low-cardinality object columns to category for memory savings
            if len(df[col].unique()) < 0.5 * len(df):
                df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f'Final Memory usage: {end_mem:.2f} MB (Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df