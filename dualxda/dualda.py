# Copyright (c) 2024, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & the authors: Galip Ümit Yolcu, Moritz Weckbecker, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin.
# Licensed under CC-BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
from .utils import display_img, colourgradarrow, InitRelevanceRule

import os
import time

import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn_dual.svm import (
    LinearSVC,
)  ## We modified LIBLINEAR MCSVM_CS_Solver to return the dual variables after training
from tqdm import tqdm
from zennit.composites import (
    EpsilonPlusFlat,
    EpsilonPlus,
    EpsilonAlpha2Beta1,
    EpsilonAlpha2Beta1Flat,
    NameMapComposite,
    MixedComposite,
)
from zennit.rules import Flat
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.image import imgify


class DualDA:

    def remove_hook(self):
        self.hook_handle.remove()

    def __exit__(self, exc_type, exc, tb):
        self.remove_hook()

    def __enter__(self):
        return self

    def __init__(
        self,
        model,
        dataset,
        classifier_layer,
        device,
        cache_dir,
        C=1.0,
        max_iter=1000000,
    ):
        # caching parameters
        if cache_dir[-1] == "\\":
            cache_dir = cache_dir[:-1]
        self.cache_dir = cache_dir

        # sklearn training parameters
        self.C = C
        self.max_iter = max_iter

        # core parameters
        self.dataset = dataset
        self.classifier = classifier_layer
        self.device = device
        self.model = model
        dev = torch.device(device)
        self.model.to(dev)

        # hook params
        self.hook_out = {}

        # Define the hook
        def get_hook_fn(layer):
            def hook_fn(module, input, output):
                input = input[0]
                if len(input.shape) != 2:
                    input = torch.flatten(input, 1)
                self.hook_out[layer] = input.detach().to(self.device)

            return hook_fn

        # Register the hook
        layer = dict(model.named_modules()).get(classifier_layer, None)
        if layer is None:
            raise ValueError(f"Layer '{classifier_layer}' not found in model.")
        self.hook_handle = layer.register_forward_hook(get_hook_fn(classifier_layer))

        self.coefficients = None  # the coefficients for each training datapoint x class
        self.learned_weights = None
        self._active_indices = None
        self.samples = None
        self.labels = None

        if not (
            os.path.isfile(os.path.join(cache_dir, self.name, "weights"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "coefficients"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "active_indices"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "samples"))
            and os.path.isfile(os.path.join(cache_dir, self.name, "labels"))
        ):
            # assert self._needs_training()
            self.samples = []
            self.labels = torch.empty(size=(0,), device=self.device, dtype=torch.int)
            loader = torch.utils.data.DataLoader(dataset, batch_size=32)
            for x, y in tqdm(iter(loader)):
                x = x.to(self.device)
                y = y.to(self.device)
                with torch.no_grad():
                    x = model(x)
                    # self.samples = torch.cat((self.samples, hook_out[classifier_layer].to(self.device)), 0)
                    self.samples.append(self.hook_out[classifier_layer].to(self.device))
                    self.labels = torch.cat((self.labels, y), 0)
            self.samples = torch.concat(self.samples)

            self.train()
        else:
            # Read all values here
            self._read_variables()

    def _read_variables(self):
        self.learned_weights = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "weights"),
                map_location=self.device,
            )
            .to(torch.float)
            .to(self.device)
        )
        self.coefficients = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "coefficients"),
                map_location=self.device,
            )
            .to(torch.float)
            .to(self.device)
        )
        self.train_time = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "train_time"),
                map_location=self.device,
            )
            .to(torch.float)
            .to(self.device)
        )
        self._active_indices = (
            torch.load(
                os.path.join(self.cache_dir, self.name, "active_indices"),
                map_location=self.device,
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

        os.makedirs(os.path.join(self.cache_dir, self.name), exist_ok=True)

        torch.save(
            self.learned_weights.cpu(),
            os.path.join(self.cache_dir, self.name, "weights"),
        )
        torch.save(
            self.coefficients.cpu(),
            os.path.join(self.cache_dir, self.name, "coefficients"),
        )
        torch.save(
            self._active_indices.cpu(),
            os.path.join(self.cache_dir, self.name, "active_indices"),
        )

        self.samples = self.samples[self._active_indices]
        self.labels = self.labels[self._active_indices]
        torch.save(
            self.samples.cpu(), os.path.join(self.cache_dir, self.name, "samples")
        )
        torch.save(self.labels.cpu(), os.path.join(self.cache_dir, self.name, "labels"))

        self.train_time = torch.tensor(time.time() - tstart, device=self.device)
        torch.save(
            self.train_time.cpu(), os.path.join(self.cache_dir, self.name, "train_time")
        )

    def attribute(self, x, xpl_targets, drop_zero_columns=False):
        with torch.no_grad():
            assert self.coefficients is not None
            x = x.to(self.device)
            xpl_targets = xpl_targets.to(self.device)
            _ = self.model(x)
            f = self.hook_out[self.classifier]
            crosscorr = torch.matmul(f, self.samples.T)
            crosscorr = crosscorr[:, :, None]
            xpl = self.coefficients * crosscorr
            indices = xpl_targets[:, None, None].expand(-1, self.samples.shape[0], 1)
            xpl = torch.gather(xpl, dim=-1, index=indices)
            xpl = torch.squeeze(xpl)
            if not drop_zero_columns:
                total_xpl = torch.zeros(
                    x.shape[0], len(self.dataset), device=self.device
                )
                total_xpl[:, self._active_indices] = xpl
                xpl = total_xpl
            return xpl

    def self_influences(self, only_coefs=False, drop_zero_columns=False):
        self_coefs = self.coefficients[
            torch.arange(self.coefficients.shape[0]), self.labels
        ]
        ret = (
            self_coefs
            if only_coefs
            else ((self.samples.norm(dim=-1) ** 2) * self_coefs)
        )
        if not drop_zero_columns:
            total_ret = torch.zeros(len(self.dataset), device=self.device)
            total_ret[self._active_indices] = ret
            ret = total_ret
        return ret

    @classmethod
    def _resolve_composite(cls, composite, canonizer="SequentialMergeBatchNorm", flat_layers=None):
        canonizer_dict = {
            "SequentialMergeBatchNorm": SequentialMergeBatchNorm(),
        }
        composite_dict = {
            "EpsilonPlus": EpsilonPlus,
            "EpsilonPlusFlat": EpsilonPlusFlat,
            "EpsilonAlpha2Beta1": EpsilonAlpha2Beta1,
            "EpsilonAlpha2Beta1Flat": EpsilonAlpha2Beta1Flat,
            # "EpsilonGammaBox": EpsilonGammaBox(zero_params='bias'),
        }
        canonizer = (
            canonizer_dict[canonizer] if isinstance(canonizer, str) else canonizer
        )
        composite = (
            composite_dict[composite](
                zero_params="bias",
                canonizers=[canonizer] if canonizer is not None else None,
            )
            if isinstance(composite, str)
            else composite
        )

        if isinstance(flat_layers, list) and len(flat_layers) > 0:
            flat_composite = NameMapComposite(
                name_map=[(flat_layers, Flat(zero_params="bias"))]
            )
            return MixedComposite([flat_composite, composite])
        else:
            return composite

    @classmethod
    def get_fontsize_from_nsamples(cls, nsamples):
        if nsamples <= 4:
            return 12
        elif nsamples == 5:
            return 17
        else:
            return 20

    def lrp(self, test_input, class_to_explain, composite):
        num_classes = self.coefficients.shape[-1]

        if class_to_explain == None:
            class_to_explain = test_input[1]

        with Gradient(model=self.model, composite=composite) as attributor:
            _, relevance = attributor(
                test_input[None],
                torch.eye(num_classes, device=test_input.device)[
                    None, class_to_explain
                ],
            )

        relevance = relevance[0].sum(dim=0).detach().cpu()

        return relevance

    def xda_heatmap(
        self, test_input, train_idx, attribution, mode="train", composite=EpsilonPlus()
    ):
        train_input, _ = self.dataset[train_idx]
        num_classes = self.coefficients.shape[-1]
        train_input = train_input.to(self.device)
        with torch.no_grad():
            self.model(train_input[None])
            train_features = self.hook_out[self.classifier]
            self.model(test_input[None])
            test_features = self.hook_out[self.classifier]
            attr_output = (
                test_features
                * train_features
                * (attribution / (train_features @ test_features.T))
            )
        new_composite = NameMapComposite(
            name_map=[([self.classifier], InitRelevanceRule(attr_output))]
        )
        new_composite = MixedComposite([new_composite, composite])
        to_attribute = train_input[0] if mode == "train" else test_input[0]

        if len(to_attribute.shape) == 2:
            to_attribute = to_attribute[None]  # Add color dimension for greyscale

        # Make a new composite that registers a special Rule to override initial relevances
        with Gradient(model=self.model, composite=new_composite) as attributor:
            _, relevance = attributor(
                to_attribute[None], torch.zeros(1, num_classes, device=self.device)
            )

        relevance = relevance[0].sum(0).detach().cpu()

        return relevance

    def xda(
        self,
        test_sample,
        inv_transform,
        class_names,
        attr,
        attr_target,
        fname,
        nsamples=5,
        composite="EpsilonPlusFlat",
        canonizer="SequentialMergeBatchNorm",
        flat_layers=None,
        save_path=None,
    ):
        test_sample = test_sample.to(self.device)
        attr = attr.to(self.device)
        composite = self._resolve_composite(
            composite=composite, canonizer=canonizer, flat_layers=flat_layers
        )
        size = 2
        proponent_idxs = torch.topk(attr, nsamples).indices[:nsamples]
        opponent_idxs = torch.topk(-attr, nsamples).indices[:nsamples]
        # Create figure with dualxda-package/test_attribution.pya specific size ratio to keep squares
        fig = plt.figure(figsize=((2 * nsamples + 2) * size, 4 * size))

        # Create a custom grid
        gs = gridspec.GridSpec(6, 2 * nsamples + 2 + 1 + 2, figure=fig)

        # Set spacing between subplots
        gs.update(wspace=0.05, hspace=0.2)

        # Big square 2x2 in the middle
        ax_big = fig.add_subplot(gs[1:3, nsamples + 1 : nsamples + 2 + 1])
        display_img(ax_big, test_sample.cpu(), inv_transform)
        # ax_big.set_title("Test Sample", fontsize=20)
        # Add black frame but still hide axes ticks
        ax_big.axis("on")
        ax_big.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Set spines (borders) to black
        for spine in ax_big.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2)

        # Big LRP square 2x2 in the middle
        ax_big = fig.add_subplot(gs[1:3, nsamples + 1 + 2 : nsamples + 2 + 1 + 2])
        relevance = self.lrp(test_sample, attr_target, composite=composite)
        img = imgify(relevance, cmap="bwr", symmetric=True)
        display_img(ax_big, test_sample.cpu(), inv_transform)
        ax_big.imshow(img, alpha=0.9)
        # ax_big.set_title("Test Sample", fontsize=20)
        # Add black frame but still hide axes ticks
        ax_big.axis("on")
        ax_big.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Set spines (borders) to black
        for spine in ax_big.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2)

        # Middle row: Picture of proponents / opponents
        for i in range(nsamples):
            # Proponent
            ax = fig.add_subplot(gs[2, nsamples + 2 + i + 1 + 2])
            display_img(ax, self.dataset[proponent_idxs[i]][0], inv_transform)
            # Add black frame but still hide axes ticks
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Set spines (borders) to black
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            # Opponent
            ax = fig.add_subplot(gs[2, nsamples - 1 - i + 1])
            display_img(ax, self.dataset[opponent_idxs[i]][0], inv_transform)
            # Add black frame but still hide axes ticks
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Set spines (borders) to black
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)

        # Middle row: Add Relevance and label
        for i in range(nsamples):
            # Proponent
            ax = fig.add_subplot(gs[1, nsamples + 2 + i + 1 + 2])
            ax.axis("off")
            # Add title lower in the cell (at y=0.3 instead of 0.5)
            ax.text(
                0.5,
                0.05,
                f"Attribution: {attr[proponent_idxs[i]]:.2f},\nLabel: {class_names[self.dataset[proponent_idxs[i]][1]]}",
                ha="center",
                va="center",
                fontsize=9,
            )
            # Opponent
            ax = fig.add_subplot(gs[1, nsamples - 1 - i + 1])
            ax.axis("off")
            # Add title lower in the cell (at y=0.3 instead of 0.5)
            ax.text(
                0.5,
                0.05,
                f"Attribution: {attr[opponent_idxs[i]]:.2f},\nLabel: {class_names[self.dataset[opponent_idxs[i]][1]]}",
                ha="center",
                va="center",
                fontsize=9,
            )
        # Middle row: Add titles (proponents, opponents)
        # Proponent
        ax = fig.add_subplot(gs[1, nsamples + 2 + 1 + 2 : 2 * nsamples + 2 + 1 + 2])
        plt.text(
            0.5,
            0.8,
            "$\\bf{POSITIVELY}$ relevant training samples",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=self.get_fontsize_from_nsamples(nsamples),
        )
        ax.axis("off")
        # Opponent
        ax = fig.add_subplot(gs[1, 0 + 1 : nsamples + 1])
        plt.text(
            0.5,
            0.8,
            "$\\bf{NEGATIVELY}$  relevant training samples",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=self.get_fontsize_from_nsamples(nsamples),
        )
        ax.axis("off")

        # Add hline between Row 1 and 2 and between Row 3 and 4
        ax = fig.add_subplot(gs[0:2, 0 : 2 * nsamples + 2 + 1 + 2])
        ax.axhline(
            y=1 / 2, xmin=0.04, xmax=1, linestyle="-", linewidth=2, color="black"
        )
        ax.axis("off")

        ax = fig.add_subplot(gs[2:4, 0 : 2 * nsamples + 2 + 1 + 2])
        ax.axhline(
            y=1 / 2, xmin=0.04, xmax=1, linestyle="-", linewidth=2, color="black"
        )
        ax.axis("off")

        # Row 1 + Row 4: 'XDA' title
        ax = fig.add_subplot(gs[0, nsamples + 1 : nsamples + 2 + 1 + 2])
        # ax.text(0.5, 0.5, "$\\bf{XDA}$", fontsize=30, ha="center", va="center")
        ax.axis("off")
        ax = fig.add_subplot(gs[3, nsamples + 1 : nsamples + 2 + 1 + 2])
        # ax.text(0.5, 0.5, "$\\bf{XDA}$", fontsize=30, ha="center", va="center")
        ax.axis("off")

        # Row 1 + Row 5: XDA
        for i in range(nsamples):
            # Proponents
            # Train
            ax = fig.add_subplot(gs[0, nsamples + 2 + i + 1 + 2])
            relevance = self.xda_heatmap(
                test_sample,
                proponent_idxs[i],
                attr[proponent_idxs[i]],
                mode="train",
                composite=composite,
            )
            img = imgify(relevance, cmap="bwr", symmetric=True)
            display_img(ax, self.dataset[proponent_idxs[i]][0], inv_transform)
            ax.imshow(img, alpha=0.9)
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            # Test
            ax = fig.add_subplot(gs[3, nsamples + 2 + i + 1 + 2])

            relevance = self.xda_heatmap(
                test_sample,
                proponent_idxs[i],
                attr[proponent_idxs[i]],
                mode="test",
                composite=composite,
            )
            img = imgify(relevance, cmap="bwr", symmetric=True)
            display_img(ax, test_sample.cpu(), inv_transform)
            ax.imshow(img, alpha=0.9)
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)

            # Opponents
            # Train
            ax = fig.add_subplot(gs[0, nsamples - 1 - i + 1])

            relevance = self.xda_heatmap(
                test_sample,
                opponent_idxs[i],
                attr[opponent_idxs[i]],
                mode="train",
                composite=composite,
            )
            img = imgify(relevance, cmap="bwr", symmetric=True)
            display_img(ax, self.dataset[opponent_idxs[i]][0], inv_transform)
            ax.imshow(img, alpha=0.9)
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            # Test
            ax = fig.add_subplot(gs[3, nsamples - 1 - i + 1])

            relevance = self.xda_heatmap(
                test_sample,
                opponent_idxs[i],
                attr[opponent_idxs[i]],
                mode="test",
                composite=composite,
            )
            img = imgify(relevance, cmap="bwr", symmetric=True)
            display_img(ax, test_sample.cpu(), inv_transform)
            ax.imshow(img, alpha=0.9)
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)

        # Add proponent arrow (red)
        ax = fig.add_subplot(gs[1, nsamples + 2 + 1 + 2 : 2 * nsamples + 2 + 1 + 2])
        colourgradarrow(ax, (0, 0.5), (nsamples, 0.5), cmap="Reds_r", n=100, lw=10)
        ax.axis("off")

        # Add opponent arrow (blue)
        ax = fig.add_subplot(gs[1, 0 + 1 : nsamples + 1])
        colourgradarrow(ax, (nsamples, 0.5), (0, 0.5), cmap="Blues_r", n=100, lw=10)
        ax.axis("off")

        # Add train/text
        ax = fig.add_subplot(gs[0, 0])
        ax.text(0.8, 0.5, "Train", rotation=90, fontsize=30, ha="center", va="center")
        ax.axis("off")

        ax = fig.add_subplot(gs[3, 0])
        ax.text(0.8, 0.5, "Test", rotation=90, fontsize=30, ha="center", va="center")
        ax.axis("off")

        # Add vertical line
        ax = fig.add_subplot(gs[0:4, 0:2])
        ax.axvline(x=1 / 2, ymin=0.0, ymax=1, linestyle="-", linewidth=2, color="black")
        ax.axis("off")

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, f"{fname}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    def da_figure(
        self,
        test_sample,
        inv_transform,
        class_names,
        attr,
        fname,
        nsamples=5,
        save_path=None,
    ):
        test_sample = test_sample.to(self.device)
        attr = attr.to(self.device)
        size = 2
        proponent_idxs = torch.topk(attr, nsamples).indices
        opponent_idxs = torch.topk(-attr, nsamples).indices
        # Create figure with dualxda-package/test_attribution.pya specific size ratio to keep squares
        fig = plt.figure(figsize=((2 * nsamples + 2) * size, 2 * size))

        # Create a custom grid
        gs = gridspec.GridSpec(2, 2 * nsamples + 2, figure=fig)

        # Set spacing between subplots
        gs.update(wspace=0.05, hspace=0.05)

        # Big square 2x2 in the middle
        ax_big = fig.add_subplot(gs[0:2, nsamples : nsamples + 2])
        display_img(ax_big, test_sample.cpu(), inv_transform)
        # ax_big.set_title("Test Sample", fontsize=20)
        # Add black frame but still hide axes ticks
        ax_big.axis("on")
        ax_big.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        # Set spines (borders) to black
        for spine in ax_big.spines.values():
            spine.set_color("black")
            spine.set_linewidth(2)

        # Middle row: Picture of proponents / opponents
        for i in range(nsamples):
            # Proponent
            ax = fig.add_subplot(gs[1, nsamples + 2 + i])
            display_img(ax, self.dataset[proponent_idxs[i]][0], inv_transform)
            # Add black frame but still hide axes ticks
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Set spines (borders) to black
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)
            # Opponent
            ax = fig.add_subplot(gs[1, nsamples - 1 - i])
            display_img(ax, self.dataset[opponent_idxs[i]][0], inv_transform)
            # Add black frame but still hide axes ticks
            ax.axis("on")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            # Set spines (borders) to black
            for spine in ax.spines.values():
                spine.set_color("black")
                spine.set_linewidth(2)

        # Middle row: Add Relevance and label
        for i in range(nsamples):
            # Proponent
            ax = fig.add_subplot(gs[0, nsamples + 2 + i])
            ax.axis("off")
            # Add title lower in the cell (at y=0.3 instead of 0.5)
            ax.text(
                0.5,
                0.05,
                f"Attribution: {attr[proponent_idxs[i]]:.2f},\nLabel: {class_names[self.dataset[proponent_idxs[i]][1]]}",
                ha="center",
                va="center",
                fontsize=9,
            )
            # Opponent
            ax = fig.add_subplot(gs[0, nsamples - 1 - i])
            ax.axis("off")
            # Add title lower in the cell (at y=0.3 instead of 0.5)
            ax.text(
                0.5,
                0.05,
                f"Attribution: {attr[opponent_idxs[i]]:.2f},\nLabel: {class_names[self.dataset[opponent_idxs[i]][1]]}",
                ha="center",
                va="center",
                fontsize=9,
            )
        # Middle row: Add titles (proponents, opponents)
        # Proponent
        ax = fig.add_subplot(gs[0, nsamples + 2 : 2 * nsamples + 2])
        plt.text(
            0.5,
            0.8,
            "$\\bf{POSITIVELY}$ relevant training samples",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=self.get_fontsize_from_nsamples(nsamples),
        )
        ax.axis("off")
        # Opponent
        ax = fig.add_subplot(gs[0, 0:nsamples])
        plt.text(
            0.5,
            0.8,
            "$\\bf{NEGATIVELY}$  relevant training samples",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=self.get_fontsize_from_nsamples(nsamples),
        )
        ax.axis("off")

        # Add proponent arrow (red)
        ax = fig.add_subplot(gs[0, nsamples + 2 : 2 * nsamples + 2])
        colourgradarrow(ax, (0, 0.5), (nsamples, 0.5), cmap="Reds_r", n=100, lw=10)
        ax.axis("off")

        # Add opponent arrow (blue)
        ax = fig.add_subplot(gs[0, 0:nsamples])
        colourgradarrow(ax, (nsamples, 0.5), (0, 0.5), cmap="Blues_r", n=100, lw=10)
        ax.axis("off")

        plt.tight_layout()
        os.makedirs(save_path, exist_ok=True)
        # plt.show(block=True)
        plt.savefig(
            os.path.join(save_path, f"{fname}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    @property
    def active_indices(self):
        return torch.where(self._active_indices)[0]

    @property
    def name(self):
        return f"DualDA_C={str(self.C)}"
