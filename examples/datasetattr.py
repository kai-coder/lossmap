from torch.utils.data import Dataset


class DatasetAttr(Dataset):
    def __init__(self, batch_size: int, shuffle: bool) -> None:
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self) -> None:
        print("user must define on_epoch_end")

    def __len__(self):
        print("user must define __len__")