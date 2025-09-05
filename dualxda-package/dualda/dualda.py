import torch
import os
import time
from sklearn.svm import (
    LinearSVC,
)  ##modified LIBLINEAR MCSVM_CS_Solver which returns the dual variables
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image


class FeatureDataset(Dataset):
    def __init__(self, model, features_layer, dataset, device):
        super().__init__()
        self.model = model
        self.device = device
        self.hook_handle = None
        self.layer = features_layer
        hook_out = {}
        hook_in = {}

        # Define the hook
        def get_hook_fn(layer):
            def hook_fn(module, input, output):
                if len(output.shape) != 2:
                    output = torch.flatten(output, 1)
                hook_out[layer] = output.detach().cpu()

            return hook_fn

        layer = dict(model.named_modules()).get(features_layer, None)
        if layer is None:
            raise ValueError(f"Layer '{features_layer}' not found in model.")
        # Register the hook
        self.hook_handle = layer.register_forward_hook(get_hook_fn(features_layer))

        self.samples = (
            []
        )  # torch.empty(size=(0, model.classifier.in_features), device=self.device)
        self.labels = torch.empty(size=(0,), device=self.device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        for x, y in tqdm(iter(loader)):
            x = x.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                x = model(x)
                # self.samples = torch.cat((self.samples, hook_out[features_layer].to(self.device)), 0)
                self.samples.append(hook_out[features_layer].to(self.device))
                self.labels = torch.cat((self.labels, y), 0)
        self.samples = torch.concat(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]

    def __del__(self):
        self.hook_handle.remove()


class DualDA:


    def __init__(
        self,
        model,
        dataset,
        feature_layer,
        device,
        cache_dir,
        C=1.0,
        max_iter=1000000,
    ):
        # caching parameters
        if cache_dir[-1] == "\\":
            cache_dir = cache_dir[:-1]
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)

        # sklearn training parameters
        self.C = C
        self.max_iter = max_iter

        # core parameters
        self.vanilla_ds = dataset
        self.feature_layer = feature_layer
        self.device = device
        self.model = model
        dev = torch.device(device)
        self.model.to(dev)

        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self._active_indices = None
        self.samples = None
        self.labels = None

        if not (
            os.path.isfile(os.path.join(cache_dir, self.name, "weights"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "coefficients"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "_active_indices"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "samples"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "labels"))
        ):
            # Need training
            feature_ds = FeatureDataset(self.model, feature_layer, dataset, device)
            self.samples = feature_ds.samples.to(self.device)
            self.labels = torch.tensor(
                feature_ds.labels, dtype=torch.int, device=self.device
            )
            # TODO: validate that none of the files exist?
            # assert self._needs_training()
            self.train()
        else:
            # Read all values here
            self._read_variables()

    def _read_variables(self):
        self.learned_weights = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "weights"), map_location=self.device
            )
            .to(torch.float)
            .to(self.device)
        )
        self.coefficients = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "coefficients"), map_location=self.device
            )
            .to(torch.float)
            .to(self.device)
        )
        self.train_time = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "train_time"), map_location=self.device
            )
            .to(torch.float)
            .to(self.device)
        )
        self._active_indices = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "_active_indices"), map_location=self.device
            )
            .to(torch.bool)
            .to(self.device)
        )
        self.samples = (
            torch.load(os.path.join(self.cache_dir, self.name, "samples"))
            .to(torch.float)
            .to(self.device)
        )
        self.labels = (
            torch.load(os.path.join(self.cache_dir, self.name, "labels"))
            .to(torch.int)
            .to(self.device)
        )

    def train(self):
        tstart = time.time()
        model = LinearSVC(
            multi_class="crammer_singer",
            max_iter=self.max_iter,
            C=self.C,
            fit_intercept=True,
        )
        print("Training via sklearn.LinearSVC(multi_class='crammer_singer')")
        model.fit(self.samples.cpu(), self.labels.cpu())
        print("SVM training finished")

        coefficients = torch.tensor(
            model.alpha_.T, dtype=torch.float, device=self.device
        )
        self.learned_weights = torch.tensor(
            model.coef_, dtype=torch.float, device=self.device
        )

        self._active_indices = ~(
            torch.all(
                torch.isclose(
                    coefficients,
                    torch.tensor(0.0, device=self.device),
                ),
                dim=-1,
            )
        )
        # assert torch.all(
        #     torch.any(
        #         ~torch.isclose(
        #             coefficients[self._active_indices], torch.tensor(0.0, device="cuda")
        #         ),
        #         dim=-1,
        #     )
        # )
        # assert torch.all(
        #     torch.isclose(
        #         coefficients[~self._active_indices], torch.tensor(0.0, device="cuda")
        #     )
        # )
        self.coefficients = coefficients[self._active_indices]
        torch.save(self.learned_weights.cpu(), os.path.join(self.cache_dir, "weights"))
        torch.save(
            self.coefficients.cpu(), os.path.join(self.cache_dir, "coefficients")
        )
        torch.save(
            self._active_indices.cpu(), os.path.join(self.cache_dir, "_active_indices")
        )

        self.samples = self.samples[self._active_indices]
        self.labels = self.labels[self._active_indices]
        torch.save(self.samples.cpu(), os.path.join(self.cache_dir, "samples"))
        torch.save(self.labels.cpu(), os.path.join(self.cache_dir, "labels"))

        self.train_time = torch.tensor(time.time() - tstart, device=self.device)
        torch.save(self.train_time.cpu(), os.path.join(self.cache_dir, "train_time"))

    def explain(self, x, xpl_targets, drop_zero_columns=False):
        with torch.no_grad():
            assert self.coefficients is not None
            x = x.to(self.device)
            # TODO: use hook
            f = self.model.features(x)
            crosscorr = torch.matmul(f, self.samples.T)
            crosscorr = crosscorr[:, :, None]
            xpl = self.coefficients * crosscorr
            indices = xpl_targets[:, None, None].expand(-1, self.samples.shape[0], 1)
            xpl = torch.gather(xpl, dim=-1, index=indices)
            xpl = torch.squeeze(xpl)
            if not drop_zero_columns:
                total_xpl = torch.zeros(
                    x.shape[0], len(self.vanilla_ds), device=self.device
                )
                total_xpl[:, self._active_indices] = xpl
                xpl = total_xpl
            return xpl

    def self_influences(self, only_coefs=False, drop_zero_columns=False):
        self_coefs = self.coefficients[
            torch.arange(self.coefficients.shape[0]), self.labels
        ]
        ret = self_coefs if only_coefs else (self.samples.norm(dim=-1) * self_coefs)
        if not drop_zero_columns:
            total_ret = torch.zeros(len(self.vanilla_ds), device=self.device)
            total_ret[self._active_indices]=ret
            ret=total_ret
        return ret
    
    @property
    def active_indices(self):
        return torch.where(self._active_indices)[0]

    @property
    def name(self):
        return f"DualDA_C={str(self.C)}"