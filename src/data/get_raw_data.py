import pandas as pd
import numpy as np

def load_eeg_data(file_path, skiprows=2):
    """Load EEG data from CSV file."""
    return pd.read_csv(file_path, skiprows=skiprows)

def load_triggers(file_path):
    """Load triggers from the triggers.txt file."""
    triggers = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                letter = parts[0]
                event = parts[1]
                timestamp = float(parts[2])
                triggers.append((letter, event, timestamp))
    return triggers

def separate_inquiries(triggers):
    """Separate the triggers into inquiries based on 'fixation' events."""
    inquiries = []
    current_inquiry = []
    for event in triggers:
        if event[1] == 'fixation':
            if current_inquiry:
                inquiries.append(current_inquiry)
            current_inquiry = []
        current_inquiry.append(event)
    if current_inquiry:
        inquiries.append(current_inquiry)
    return inquiries

def generate_stimuli_signal(inquiry_df, events, time_window=0.1):
    """Generate stimuli signal for the inquiry."""
    signal = np.zeros(len(inquiry_df))
    for event, timestamp in events:
        if event == 'target':
            signal_value = 1
        elif event == 'nontarget':
            signal_value = 0.5
        else:
            continue

        # Find the index range for the signal centered around the timestamp
        half_window = time_window / 2
        indices = (inquiry_df['lsl_timestamp'] >= timestamp - half_window) & (inquiry_df['lsl_timestamp'] <= timestamp + half_window)
        signal[indices] = signal_value

    return signal

def process_inquiries(eeg_data, inquiries, max_rows=3030):
    """Process the inquiries and separate the EEG data."""
    inquiry_data = []
    for i, inquiry in enumerate(inquiries):
        start_time = inquiry[0][2]  # Timestamp of the 'fixation' event
        if i < len(inquiries) - 1:
            end_time = inquiries[i + 1][0][2]  # Timestamp of the next 'fixation' event
        else:
            end_time = inquiry[-1][2] + 4  # Last letter timestamp + 4 seconds for the last inquiry

        inquiry_df = eeg_data[(eeg_data['lsl_timestamp'] >= start_time) & (eeg_data['lsl_timestamp'] <= end_time)].copy()
        inquiry_df['relative_time'] = inquiry_df['lsl_timestamp'] - start_time

        # Generate stimuli signal
        events = [(e[1], e[2]) for e in inquiry if e[1] in ['target', 'nontarget']]
        stimuli_signal = generate_stimuli_signal(inquiry_df, events)
        inquiry_df['stimuli_signal'] = stimuli_signal

        inquiry_data.append(inquiry_df.iloc[:max_rows])  # Append only the first max_rows rows

    return inquiry_data


def process_eeg_and_triggers(eeg_file_path, triggers_file_path, skiprows=2, max_rows=3030):
    """Main function to process EEG data and triggers and save the results."""
    eeg_data = load_eeg_data(eeg_file_path, skiprows)
    triggers = load_triggers(triggers_file_path)
    inquiries = separate_inquiries(triggers)
    inquiry_data = process_inquiries(eeg_data, inquiries, max_rows)
    return inquiry_data

# Example usage:
# process_eeg_and_triggers(
#     eeg_file_path='/path/to/eeg_data.csv',
#     triggers_file_path='/path/to/triggers.txt',
#     output_dir='/path/to/output_directory'
# )

