import pytest
from unittest.mock import patch, MagicMock
from train_multi import find_latest_checkpoint

def test_find_latest_checkpoint_empty_dir(tmp_path):
    assert find_latest_checkpoint(str(tmp_path)) is None

def test_find_latest_checkpoint_no_match(tmp_path):
    (tmp_path / "other_run_100_steps.txt").touch()
    assert find_latest_checkpoint(str(tmp_path)) is None

def test_find_latest_checkpoint_single_match(tmp_path):
    f = tmp_path / "test_run_100_steps.zip"
    f.touch()
    assert find_latest_checkpoint(str(tmp_path)) == str(f)

def test_find_latest_checkpoint_multiple_matches(tmp_path):
    f1 = tmp_path / "test_run_100_steps.zip"
    f2 = tmp_path / "test_run_200_steps.zip"
    f3 = tmp_path / "test_run_50_steps.zip"
    import time
    f1.touch()
    time.sleep(0.01)
    f3.touch()
    time.sleep(0.01)
    f2.touch()
    assert find_latest_checkpoint(str(tmp_path)) == str(f2)
