import pytest
import os
import json
from unittest.mock import patch, mock_open

from offline_analysis import analyze_actions

def test_analyze_actions():
    mock_data = '{"episode": 1, "action": [1, 0, 0, 0], "reward": 10.0, "ep_reward": 10.0, "total_steps": 1}\n'

    with patch('builtins.open', mock_open(read_data=mock_data)):
        with patch('offline_analysis.os.path.exists', return_value=True):
            with patch('offline_analysis.plt') as mock_plt:
                analyze_actions()

                assert mock_plt.figure.call_count > 0
                assert mock_plt.savefig.call_count > 0

def test_analyze_actions_no_file():
    with patch('offline_analysis.os.path.exists', return_value=False):
        # Should print and return early
        with patch('builtins.print') as mock_print:
            analyze_actions()
            mock_print.assert_called_once()
