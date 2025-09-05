from tqdm import tqdm
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image
import os


class FeatureDataset(Dataset):
    def __init__(self, model, features_layer, dataset, device):
        super().__init__()
        self.model = model
        self.device = device
        self.hook_handle=None
        self.layer=features_layer
        hook_out = {}

        # Define the hook
        def get_hook_fn(layer):
            def hook_fn(module, input, output):
                if len(output.shape)!=2:
                    output=torch.flatten(output,1)
                hook_out[layer] = output.detach().cpu()
            return hook_fn
        layer = dict(model.named_modules()).get(features_layer, None)
        if layer is None:
            raise ValueError(f"Layer '{features_layer}' not found in model.")
        # Register the hook
        self.hook_handle = layer.register_forward_hook(get_hook_fn(features_layer))

        self.samples = torch.empty(size=(0, model.classifier.in_features), device=self.device)
        self.labels = torch.empty(size=(0,), device=self.device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        for x, y in tqdm(iter(loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                x = model(x)
                self.samples = torch.cat((self.samples, hook_out[features_layer]), 0)
                self.labels = torch.cat((self.labels, y), 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]
    
    def __del__(self):
        self.hook_handle.remove()


class RestrictedDataset(Dataset):
    def __init__(self, dataset, indices, return_indices=False):
        self.dataset = dataset
        self.indices = indices
        self.return_indices = return_indices
        if hasattr(dataset, "name"):
            self.name = dataset.name
        else:
            self.name = dataset.dataset.name

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        d = self.dataset[self.indices[item]]
        if self.return_indices:
            return d, self.indices[item]
        return d