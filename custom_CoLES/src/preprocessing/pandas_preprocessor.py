import logging
from typing import List, Union

import numpy as np
import pandas as pd

from ptls.preprocessing.pandas_preprocessor import PandasDataPreprocessor

from ptls.preprocessing.base import DataPreprocessor, ColTransformer
from ptls.preprocessing.base.col_category_transformer import ColCategoryTransformer
from ptls.preprocessing.pandas.category_identity_encoder import CategoryIdentityEncoder
from ptls.preprocessing.pandas.col_identity_transformer import ColIdentityEncoder
from ptls.preprocessing.pandas.event_time import DatetimeToTimestamp
from ptls.preprocessing.pandas.user_group_transformer import UserGroupTransformer


from .pandas.frequency_encoder import CustomFrequencyEncoder
from .pandas.user_group_transformer import CustomUserGroupTransformer

logger = logging.getLogger(__name__)


class CustomPandasDataPreprocessor(PandasDataPreprocessor):
    """Data preprocessor based on pandas.DataFrame

    During preprocessing it
        * transform datetime column to `event_time`
        * encodes category columns into indexes;
        * groups flat data by `col_id`;
        * arranges data into list of dicts with features

    Preprocessor don't modify original dataframe, but links to his data.

    Parameters
    ----------
    col_id : str
        name of column with ids. Used for groups
    col_event_time : str
        name of column with datetime
        or `ColTransformer` implementation with datetime transformation
    event_time_transformation: str
        name of transformation for `col_event_time`
        - 'dt_to_timestamp': datetime (string of datetime64) to timestamp (long) with `DatetimeToTimestamp`
            Original column is dropped by default cause target col `event_time` is the same information
            and we can not use as feature datetime column itself.
        - 'none': without transformation, `col_event_time` is in correct format. Used `ColIdentityEncoder`
            Original column is kept by default cause it can be any type and we may use it in the future
    cols_category : list[str]
        list of category columns. Each can me column name or `ColCategoryTransformer` implementation.
    category_transformation: str
        name of transformation for column names from `cols_category`
        - 'frequency': frequency encoding with `FrequencyEncoder`
        - 'none': no transformation with `CategoryIdentityEncoder`
    cols_numerical : list[str]
        list of columns to be mentioned as numerical features. No transformation with `ColIdentityEncoder`
    cols_identity : list[str]
        list of columns to be passed as is without any transformation
    cols_first_item: List[str]
        Only first value will be taken for these columns
        It can be user-level information joined to each transaction
    return_records:
        False: Result is a `pandas.DataFrame`.
            You can:
            - join any additional information like user-level features of target
            - convert it to `ptls` format using `.to_dict(orient='records')`
        True: Result is a list of dicts - `ptls` format

    """

    def __init__(self,
                 col_id: str,
                 col_event_time: Union[str, ColTransformer],
                 event_time_transformation: str = 'dt_to_timestamp',
                 cols_category: List[Union[str, ColCategoryTransformer]] = None,
                 category_transformation: str = 'frequency',
                 cols_numerical: List[str] = None,
                 cols_identity: List[str] = None,
                 cols_first_item: List[str] = None,
                 return_records: bool = True,
                 ):
        if cols_category is None:
            cols_category = []
        if cols_numerical is None:
            cols_numerical = []
        if cols_identity is None:
            cols_identity = []
        if cols_first_item is None:
            cols_first_item = []


        if type(col_event_time) is not str:
            ct_event_time = col_event_time  # use as is
        elif event_time_transformation == 'dt_to_timestamp':
            ct_event_time = DatetimeToTimestamp(col_name_original=col_event_time)
        elif event_time_transformation == 'none':
            ct_event_time = ColIdentityEncoder(
                col_name_original=col_event_time,
                col_name_target='event_time',
                is_drop_original_col=False,
            )
        else:
            raise AttributeError(f'incorrect event_time parameters combination: '
                                 f'`ct_event_time` = "{col_event_time}" '
                                 f'`event_time_transformation` = "{event_time_transformation}"')

        cts_category = []
        for col in cols_category:
            if type(col) is not str:
                cts_category.append(col)  # use as is
            elif category_transformation == 'frequency':
                cts_category.append(CustomFrequencyEncoder(col_name_original=col))
            elif category_transformation == 'none':
                cts_category.append(CategoryIdentityEncoder(col_name_original=col))
            else:
                raise AttributeError(f'incorrect category parameters combination: '
                                     f'`cols_category[i]` = "{col}" '
                                     f'`category_transformation` = "{category_transformation}"')

        cts_numerical = [ColIdentityEncoder(col_name_original=col) for col in cols_numerical]
        t_user_group = CustomUserGroupTransformer(
            col_name_original=col_id, 
            cols_first_item=cols_first_item, 
            return_records=return_records
        )

        DataPreprocessor.__init__(
            self,
            ct_event_time=ct_event_time,
            cts_category=cts_category,
            cts_numerical=cts_numerical,
            cols_identity=cols_identity,
            t_user_group=t_user_group,
        )
