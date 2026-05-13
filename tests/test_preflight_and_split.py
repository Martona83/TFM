import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import config_from_user_config
from src.data import split_data
import pandas as pd
import pytest


def _df():
    return pd.DataFrame({
        'target':[0,1,0,1,0,1],
        'event_time':pd.date_range('2020-01-01', periods=6),
        'id':['a','b','c','d','e','f'],
        'age_group':['A','A','B','B','A','B']
    })


def test_temporal_required_when_event_columns_present():
    df = _df()
    cfg = config_from_user_config({'event_time_columns':['event_time'], 'split_strategy':'random_stratified'})
    with pytest.raises(ValueError):
        split_data(df, ('age_group',), cfg)


def test_temporal_split_is_deterministic_ordered():
    df = _df()
    cfg = config_from_user_config({'event_time_columns':['event_time'], 'split_strategy':'temporal', 'split_time_col':'event_time', 'validation_size':0.2, 'test_size':0.2})
    splits = split_data(df, ('age_group',), cfg)
    assert splits['train']['event_time'].max() <= splits['validation']['event_time'].min()
    assert splits['validation']['event_time'].max() <= splits['test']['event_time'].min()
