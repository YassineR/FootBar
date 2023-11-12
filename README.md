# BootFar Challenge


## Files

- **Description.ipynb**: Jupyter Notebook containing the description of my work.
- **Exploration.ipynb**: Jupyter Notebook for data exploration.
- **actions_prepro.csv**: CSV file containing the data after the post-processing.
- **generate_match.py**: Python script to generate matches.
- **model.pt**: Model file saved.
- **scaler.pkl**: Pickle file for scaler.

## Getting Started

### Prerequisites

Before you start, make sure you have the following prerequisites installed:

- [Conda](https://docs.conda.io/en/latest/)
- [Python 3.9](https://www.python.org/downloads/)

### Installation

Follow these steps to set up the environment and install the necessary dependencies:

1. Create a Conda environment named "FB" with Python 3.9:

    ```bash
    conda create --name FB python==3.9
    ```

2. Activate the Conda environment:

    ```bash
    conda activate FB
    ```

3. Install the required Python package :

    ```bash
    pip install gretel-synthetics
    ```

Now you have set up the environment and installed the necessary dependencies. You can proceed with running the project.

## Usage

To generate a match, run the `generate_match.py` script with the following command-line arguments:

```bash
python generate_match.py <duration_in_minutes> <style>
```

duration_in_minutes: Duration of the match in minutes.
style: Style of the match. Choose one of the following: 'defensif', 'offensif', 'neutre'.

Example : 
```bash
python generate_match.py 90 offensif
```

