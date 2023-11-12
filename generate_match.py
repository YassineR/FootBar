import numpy as np
import pandas as pd
from pickle import load
import sys

from gretel_synthetics.timeseries_dgan.dgan import DGAN
import json

def buildMatch(dataframe, total_rows, style):    
    
    match_style = {}
    if style == 'defensif':
        match_style = {'walk': 0.43 * total_rows,
                        'run': 0.36 * total_rows,
                        'pass': 0.05 * total_rows,
                        'dribble': 0.02 * total_rows,
                        'rest': 0.04 * total_rows,
                        'tackle': 0.06 * total_rows,
                        'shot': 0.01 * total_rows,
                        'cross': 0.03 * total_rows,
                       }
    elif style == 'offensif':
         match_style = {'walk': 0.43 * total_rows,
                    'run': 0.36 * total_rows,
                    'pass': 0.05 * total_rows,
                    'dribble': 0.06 * total_rows,
                    'rest': 0.04 * total_rows,
                    'tackle': 0.02 * total_rows,
                    'shot': 0.03 * total_rows,
                    'cross': 0.01 * total_rows,
                   }
    
    elif style == 'neutre':
        match_style = {'walk': 0.43 * total_rows,
                        'run': 0.36 * total_rows,
                        'pass': 0.05 * total_rows,
                        'dribble': 0.04 * total_rows,
                        'rest': 0.04 * total_rows,
                        'tackle': 0.04 * total_rows,
                        'shot': 0.02 * total_rows,
                        'cross': 0.02 * total_rows,
                       }

    # Calculate required counts for each label
    selected_rows = pd.DataFrame(columns=dataframe.columns)  # Empty DataFrame to store selected rows

    # Loop through labels and select rows based on proportions
    for label, count in match_style.items():
        label_df = dataframe[dataframe['label'] == label]
        if len(label_df) > 0:
            if len(label_df) < count:
                # If there are not enough rows for the current label, oversample the existing rows
                oversampled_rows = label_df.sample(int(count), replace=True, random_state=42)
                selected_rows = pd.concat([selected_rows, oversampled_rows])
            else:
                # If there are enough rows, sample rows without replacement
                selected_rows = pd.concat([selected_rows, label_df.sample(int(count))])

    # Shuffle the selected rows
    selected_rows = selected_rows.sample(frac=1).reset_index(drop=True)

    return selected_rows


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_match.py <duration_in_minutes> <style>")
        print("  - duration_in_minutes: Duration of the match in minutes.")
        print("  - style: Style of the match. Choose one of the following: 'defensif', 'offensif', 'neutre'.")
        sys.exit(1)

    duration = int(sys.argv[1])
    style = sys.argv[2]

    if style not in ['defensif', 'offensif', 'neutre']:
        print("Invalid style. Choose one of the following: 'defensif', 'offensif', 'neutre'.")
        sys.exit(1)
		
    total_rows = int(duration*60/3)

    model = DGAN.load(file_name='model.pt')

    # load the scaler
    scaler = load(open('scaler.pkl', 'rb'))
    generated = model.generate_dataframe(total_rows*20)

    # Extract column names containing 'norm'
    norm_columns = [col for col in generated.columns if col != 'label']
    # Unscale 
    generated_unscaled = pd.DataFrame(columns = generated.columns)
    generated_unscaled['label'] = generated['label']
    generated_unscaled[norm_columns] = scaler.inverse_transform(generated[norm_columns])

    match = buildMatch(generated_unscaled, total_rows, style)


    # Rebuild the 'norm' array using norm_X columns
    match['norm'] = match[norm_columns].values.tolist()
    match = match[['label', 'norm']]

    # Convert the DataFrame to a list of dictionaries and then to a JSON string
    json_data = match.to_dict(orient='records')

    file_name = 'match_' + str(duration) + 'min_' + style + '.json'
    # Save the generated match data to a JSON file
    with open(file_name, 'w') as json_file:
        json.dump(json_data, json_file)