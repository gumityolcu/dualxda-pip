from torchvision.datasets import MNIST as tvMNIST
from torchvision import transforms
import torch
import os

class MNIST(tvMNIST):
    default_class_groups = [[i] for i in range(10)]
    class_labels=[i for i in range(10)]
    name='MNIST'
    default_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    inverse_transform = transforms.Compose([transforms.Normalize(mean=(0.,),
                                                                 std=(1 / 0.3081,)),
                                            transforms.Normalize(mean=(-0.1307,),
                                                                 std=(1.,)),
                                            ])
    def __init__(
            self,
            root,
            split="train",
            transform=None,
            inv_transform=None,
            target_transform=None,
            download=True,
            validation_size=0
    ):
        if transform is None:
            transform=MNIST.default_transform
        else:
            transform=transforms.Compose([transform, MNIST.default_transform])
        if inv_transform is not None:
            self.inverse_transform = inv_transform  # MUST HAVE THIS FOR MARK DATASET TO WORK
        train=(split=="train")
        super().__init__(root,train,transform,target_transform,download)
        self.split=split
        self.classes=[i for i in range(10)]
        N = super(MNIST, self).__len__()

        if not train:
            if (os.path.isfile(f"test/{self.name}_val_ids")and os.path.isfile(f"test/{self.name}_test_ids")):
                self.val_ids=torch.load(f"test/{self.name}_val_ids")
                self.test_ids=torch.load(f"test/{self.name}_test_ids")
            else:
                torch.manual_seed(42)  # THIS SHOULD NOT BE CHANGED BETWEEN TRAIN TIME AND TEST TIME
                perm = torch.randperm(N)
                self.val_ids = torch.tensor([i for i in perm[:validation_size]])
                self.test_ids = torch.tensor([i for i in perm[validation_size:]])
                torch.save(self.val_ids, f'test/{self.name}_val_ids')
                torch.save(self.test_ids, f'test/{self.name}_test_ids')


            print("Validation ids:")
            print(self.val_ids)
            print("Test ids:")
            print(self.test_ids)
            self.test_targets=self.targets.clone().detach()[self.test_ids]


    def __getitem__(self, item):
        if self.split=="train":
            id=item
        elif self.split=="val":
            id=self.val_ids[item]
        else:
            id=self.test_ids[item]

        x,y=super().__getitem__(id)
        return x,y


    def __len__(self):
        if self.split=="train":
            return super(MNIST, self).__len__()
        elif self.split=="val":
            return len(self.val_ids)
        else:
            return len(self.test_ids)