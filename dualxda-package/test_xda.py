from dualxda import DualDA
from models import BasicConvModel
from test.MNIST import MNIST
import torch
from test_attribution import load_datasets, load_model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model()
    ckpt_path = (
        "/home/yolcu/Documents/Code/dualxda-package/checkpoints/MNIST_basic_conv_best"
    )
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    train_ds, evalds = load_datasets()
    ldr = torch.utils.data.DataLoader(evalds, batch_size=32, shuffle=False)
    with DualDA(
        model,
        train_ds,
        "features.fc-1",
        device,
        cache_dir="/home/yolcu/Documents/Code/dualxda-package/cache",
    ) as da:
        for i in range(3):
            x, y = next(iter(ldr))
            preds = model(x.to("cuda")).argmax(dim=-1)
            xpl = da.attribute(x=x, xpl_targets=preds)

            da.xda(
                test_sample=x[i],
                inv_transform=train_ds.inverse_transform,
                class_names=range(10),
                attr=xpl[i],
                attr_target=preds[i],
                fname=f"test_{i}_features",
                composite="EpsilonPlus",
                canonizer="SequentialMergeBatchNorm",
                save_path="./xda_figures",
                nsamples=5)
            da.xda(
                test_sample=x[i],
                inv_transform=train_ds.inverse_transform,
                class_names=range(10),
                attr=xpl[i],
                attr_target=preds[i],
                fname=f"test_{i}_model",
                composite="EpsilonPlus",
                canonizer="SequentialMergeBatchNorm",
                save_path="./xda_figures",
                nsamples=5)
        

if __name__ == "__main__":
    main()
    print("exited")
