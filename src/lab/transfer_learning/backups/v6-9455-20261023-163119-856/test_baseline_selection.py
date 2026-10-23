"""Offline regression tests; no pretrained download or competition images."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline as b


class IndexedDataset(Dataset):
    def __init__(self, split):
        self.targets = [0, 0, 0, 1, 1, 1]
        self.offset = 0 if split == 'train' else 100

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return torch.full((3, 4, 4), float(self.offset + index)), self.targets[index]


class PixelClassifier(nn.Module):
    def __init__(self, sign=1):
        super().__init__()
        self.sign = nn.Parameter(torch.tensor(float(sign)))

    def forward(self, x):
        value = x.mean((1, 2, 3)) * self.sign
        return torch.stack((-value, value), 1)


class BaselineTests(unittest.TestCase):
    def test_bn_reestimation_excludes_selection_images(self):
        with patch.object(b, 'labeled', side_effect=lambda split, tf: IndexedDataset(split)):
            fit, views, bn = b.prepare_datasets(None, merged=True)
        ids = lambda ds: {int(ds[i][0][0, 0, 0]) for i in range(len(ds))}
        self.assertEqual(ids(views[0]), {0, 3})
        self.assertEqual(ids(views[1]), {0, 3})
        self.assertEqual(ids(fit), {1, 2, 4, 5, 100, 101, 102, 103, 104, 105})
        self.assertEqual(ids(bn), ids(fit))
        self.assertFalse(ids(bn) & ids(views[0]))

    def test_tta_uses_all_views_and_eval_mode(self):
        model = PixelClassifier().train()
        # First view predicts both incorrectly; second view outweighs it.
        labels = torch.tensor([0, 1])
        first = torch.tensor([0.1, -0.1]).reshape(2, 1, 1, 1)
        second = torch.tensor([-3., 3.]).reshape(2, 1, 1, 1)
        views = [DataLoader(TensorDataset(x, labels), batch_size=2) for x in (first, second)]
        with patch.object(b, 'N_CLASSES', 2):
            score = b.evaluate_tta([model], torch.device('cpu'), views)
        self.assertEqual(score, 1.)
        self.assertFalse(model.training)

    def test_tta_rejects_inconsistent_view_labels(self):
        x = torch.ones(2, 1, 1, 1)
        views = [DataLoader(TensorDataset(x, torch.tensor(y))) for y in ([0, 1], [1, 0])]
        with patch.object(b, 'N_CLASSES', 2), self.assertRaises(ValueError):
            b.evaluate_tta([PixelClassifier()], torch.device('cpu'), views)

    def test_candidate_selection_keeps_the_better_model(self):
        x = torch.tensor([-1., 1.]).reshape(2, 1, 1, 1)
        views = [DataLoader(TensorDataset(x, torch.tensor([0, 1])), batch_size=2)]
        good, bad = PixelClassifier(), PixelClassifier(-1)
        with patch.object(b, 'N_CLASSES', 2):
            selected, name, scores = b.select_candidate(
                {'raw': good, 'swa': bad}, views, torch.device('cpu'))
        self.assertIs(selected, good)
        self.assertEqual(name, 'raw')
        self.assertEqual(scores, {'raw': 1., 'swa': 0.})

    def test_bn_reestimation_restores_modes_and_momentum(self):
        model = nn.Sequential(nn.BatchNorm2d(1, momentum=0.23), nn.Dropout(0.5)).train()
        model[0].eval()
        modes = [m.training for m in model.modules()]
        weights = model[0].weight.detach().clone()
        data = DataLoader(TensorDataset(torch.full((4, 1, 2, 2), 3.), torch.zeros(4)), batch_size=2)
        b.reestimate_bn(model, data, torch.device('cpu'))
        self.assertEqual([m.training for m in model.modules()], modes)
        self.assertEqual(model[0].momentum, 0.23)
        self.assertTrue(torch.equal(model[0].weight, weights))
        self.assertTrue(torch.allclose(model[0].running_mean, torch.tensor([3.])))

    def test_checkpoint_roundtrip_keeps_an_independent_cpu_snapshot(self):
        model = PixelClassifier()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'A.pt'
            b.save_checkpoint(path, model, {'kind': 'raw', 'score': 1.})
            with torch.no_grad():
                model.sign.fill_(99.)
            saved = torch.load(path, map_location='cpu', weights_only=True)
        self.assertEqual(saved['state_dict']['sign'].item(), 1.)
        self.assertEqual(saved['metadata']['kind'], 'raw')

    def test_short_training_saves_selected_model_and_candidate_scores(self):
        # Exercise real gradient updates and candidate/BN/evaluation paths on tiny data.
        model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten())
        simple_loader = lambda ds, batch=64, shuffle=False: DataLoader(ds, batch_size=batch, shuffle=shuffle)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp, \
             patch.object(b, 'labeled', side_effect=lambda split, tf: IndexedDataset(split)), \
             patch.object(b, 'loader', side_effect=simple_loader), \
             patch.object(b, 'EPOCHS', 2), patch.object(b, 'SWA_EPOCHS', 1), \
             patch.object(b, 'WARMUP', 0), patch.object(b, 'N_CLASSES', 2):
            result = b.train(model, torch.device('cpu'), None, tag='A', merged=True, run_dir=Path(tmp))
            saved = torch.load(Path(tmp) / 'A.pt', map_location='cpu', weights_only=True)
            report = json.loads((Path(tmp) / 'A.json').read_text(encoding='utf-8'))
            self.assertEqual(set(report['candidate_scores']), {'raw', 'raw_bn', 'swa', 'swa_bn'})
            self.assertEqual(len(report['epochs']), 2)
            self.assertEqual(saved['metadata']['kind'], report['selected'])
            for key, value in result.state_dict().items():
                self.assertTrue(torch.equal(value.cpu(), saved['state_dict'][key]))


if __name__ == '__main__':
    unittest.main()
