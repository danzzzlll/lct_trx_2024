from typing import List

import torch
import pandas as pd

from ptls.preprocessing.base import ColTransformer
from ptls.preprocessing.pandas.col_transformer import ColTransformerPandasMixin
from ptls.preprocessing.pandas.user_group_transformer import UserGroupTransformer

from tqdm.auto import tqdm


class CustomUserGroupTransformer(UserGroupTransformer):    
    def df_to_feature_arrays(self, df):
        def decide(k, v):
            if k in self.cols_first_item:
                return v.iloc[0]
            elif isinstance(v.iloc[0], torch.Tensor):
                return torch.vstack(tuple(v))
            elif v.dtype == 'object':
                return v.values
            else:
                return torch.from_numpy(v.values)

        return {k: decide(k, v) for k, v in df.to_dict(orient='series').items()}
    
    def transform(self, x: pd.DataFrame):
        # Ensure event_time is attached and set as an index with user ID

        x['et_index'] = x['event_time']
        x.set_index([self.col_name_original, 'et_index'], inplace=True)
        x.sort_index(inplace=True)

        results = []

        # Get a list of unique user IDs
        unique_users = x.index.get_level_values(self.col_name_original).unique()

        # Iterate over each user ID and process their respective data
        for user in tqdm(unique_users):
            user_data = x.loc[user]  # Filter data for the current user
            result = self.df_to_feature_arrays(user_data)
            result[self.col_name_original] = user
            results.append(result)
        
        del x
        
        # Combine results into a DataFrame
        x = pd.DataFrame(results)
        
        # If return_records is True, convert DataFrame to a list of dictionaries
        if self.return_records:
            x = x.to_dict(orient='records')

        return x
