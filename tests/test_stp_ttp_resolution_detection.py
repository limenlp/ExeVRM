import unittest
import torch

class TestResolutionDetection(unittest.TestCase):
    def is_low_res_video(self, video_grid_thw, spatial_merge_size=2, threshold=500):
        if video_grid_thw is None:
            return False
        for i in range(video_grid_thw.size(0)):
            t, h, w = video_grid_thw[i].tolist()
            tokens_per_frame = (h * w) // (spatial_merge_size ** 2)
            if tokens_per_frame < threshold:
                return True
        return False

    def test_low_res_360p_like(self):
        grid_360p = torch.tensor([[50, 26, 46]])  # ~360p merged tokens: 26*46/4=299
        self.assertTrue(self.is_low_res_video(grid_360p))

    def test_high_res_720p_like(self):
        grid_720p = torch.tensor([[50, 52, 90]])  # ~720p merged tokens: 52*90/4=1170
        self.assertFalse(self.is_low_res_video(grid_720p))

    def test_mixed_batch(self):
        grid_mixed = torch.tensor([[50, 52, 90], [50, 26, 46]])
        self.assertTrue(self.is_low_res_video(grid_mixed))

    def test_imports_exist(self):
        from llamafactory.model.model_utils.ttp import apply_ttp_forward_patch
        from llamafactory.model.model_utils.stp import (
            apply_stp_forward_patch,
            patch_stp_qwen3vl_vision_encoder_with_pruning,
        )
        # Just ensure they are callables
        self.assertTrue(callable(apply_ttp_forward_patch))
        self.assertTrue(callable(apply_stp_forward_patch))
        self.assertTrue(callable(patch_stp_qwen3vl_vision_encoder_with_pruning))

if __name__ == "__main__":
    unittest.main()

