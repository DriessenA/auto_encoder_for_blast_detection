from math import sqrt, log
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader


class SimpleDataModule(LightningDataModule):
    def __init__(
        self, train, val, test, trained_scaler=None, batch_size=32, num_workers=1
    ):
        super(SimpleDataModule, self).__init__()
        self.train = train
        self.val = val
        self.test = test
        self.scaler = trained_scaler
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def predict_dataloader(self):
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()

    def state_dict(self):
        # track whatever you want here
        state = {"trained_scaler": self.scaler}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.scaler = state_dict["trained_scaler"]


def asinh_transform(raw_data, cf_dict: dict, c: int = 5):
    """Perform asinh transformation given the raw data and
    co-factors for each marker (column) in the dataframe.

    Args:
        raw_data:
            dataframe with cells as rows and markers as columns.
            Column names should match entries in `cf_dict`.
        cf_dict:
            a dictonary with marker names : co factors
        c:
            default co-factor in case a marker doesn't
            have a co-factor in the dictonairy

    Returns: data frame with transformed values

    """

    transformed_data = raw_data.copy()
    print("Transforming values...")
    for m in range(0, len(raw_data.columns)):
        marker_name = raw_data.columns[m]
        try:
            c = cf_dict[marker_name]
        except KeyError:
            print("No co-factor defined for marker " + marker_name + " using '5' ")
            c = 5

        if not c == 0:
            temp = raw_data.iloc[:, m] / c
            transformed_data.iloc[:, m] = (temp + (temp ** 2 + 1).apply(sqrt)).apply(
                log
            )
            if marker_name == "new":
                print(transformed_data)
    return transformed_data


def get_co_factors():
    d = {
        "FSC_A": 0,
        "SSC_A": 0,
        "CD38": 5460,
        "CD117": 2610,
        "CD14": 6220,
        "CD71": 2900,
        "HLA_DR": 2180,
        "CD56": 2650,
        "CD3": 3400,
        "CD13": 6640,
        "CD34": 3570,
        "CD45RA": 3700,
        "CD64": 2770,
        "CD123": 3820,
        "CD11b": 3820,
        "CD8": 3820,
        "CD35": 2860,
        "CD33": 1930,
        "CD11c": 2860,
        "CD42b": 2860,
        "CD203c": 5460,
        "CD4": 4960,
        "CD45": 4960,
        "CD19": 3030,
    }
    return d
