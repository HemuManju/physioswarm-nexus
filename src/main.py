import yaml
from pathlib import Path


from datasets.bids_dataset import convert_to_bids_dataset

from utils import skip_run

# The configuration file
config_path = Path(__file__).parents[1] / "src/config.yml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

with skip_run("run", "convert_to_bids_dataset") as check, check():
    convert_to_bids_dataset(config)
    pass
