import torchph.torchph.pershom.pershom_backend as ph
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import unittest


class TestCalculatePersistence(unittest.TestCase):
    def test_calculate_persistence(self):
        compr_desc_sort_ba = torch.randn(10, 10)  # 10x10のランダムなテンソル
        ba_row_i_to_bm_col_i = torch.randint(0, 10, (10,))  # 10要素のランダムな整数テンソル
        simplex_dimension = torch.randint(0, 3, (10,))  # 10要素のランダムな次元テンソル
        max_dim_to_read_of_reduced_ba = 2
        max_pairs = 5

        # 期待する出力の例を生成します
        expected_output = [
            [torch.randn(3), torch.randn(3)],  # 3次元のbirth/death timesのリスト
            [torch.randn(3), torch.randn(3)]   # 3次元のbirth timesのリスト
        ]


        # 関数を実行します
        result = ph.calculate_persistence(
            compr_desc_sort_ba,
            ba_row_i_to_bm_col_i,
            simplex_dimension,
            max_dim_to_read_of_reduced_ba,
            max_pairs,
        )

        # 出力が期待されるものと一致するかテストします
        self.assertEqual(result, expected_output)


if __name__ == "__main__":
    unittest.main()
