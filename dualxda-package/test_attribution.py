from dualda import DualDA
from models import BasicConvModel
from test.MNIST import MNIST
import torch


def load_model():
    params = {
        "convs": {
            "num": 3,
            "padding": 0,
            "kernel": 3,
            "stride": 1,
            "features": [5, 10, 5],
        },
        "fc": {"num": 2, "features": [500, 100]},
        "input_shape": (1, 28, 28),
    }

    return BasicConvModel(
        input_shape=params["input_shape"],
        convs=params["convs"],
        fc=params["fc"],
        num_classes=10,
    )


def load_datasets():
    ds = None
    evalds = None
    data_root = "/home/yolcu/Documents/Code/dualxda-package/dataroot"
    validation_size = 2000

    ds = MNIST(
        root=data_root, split="train", validation_size=validation_size, transform=None
    )
    evalds = MNIST(
        root=data_root, split="test", validation_size=validation_size, transform=None
    )

    return ds, evalds


def main():
    device="cuda" if torch.cuda.is_available() else "cpu"
    
    model = load_model()
    ckpt_path="/home/yolcu/Documents/Code/dualxda-package/checkpoints/MNIST_basic_conv_best"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    train_ds, evalds = load_datasets()
    
    da = DualDA(
        model,
        train_ds,
        "features",
        device,
        cache_dir="/home/yolcu/Documents/Code/dualxda-package/cache",
    )

    ldr=torch.utils.data.DataLoader(evalds, batch_size=32, shuffle=False)
    x,y=next(iter(ldr))
    preds=model(x.to("cuda")).argmax(dim=-1)
    xpl=da.explain(x=x, xpl_targets=preds)
    xpl_2=da.explain(x=x, xpl_targets=preds, drop_zero_columns=True)
    assert torch.all(xpl[:,da.active_indices]==xpl_2)
    si=da.self_influences()
    si_2=da.self_influences(drop_zero_columns=True)
    xpl=da.explain(da.samples, da.labels)
    pass


if __name__ == "__main__":
    main()
    print("exited")
