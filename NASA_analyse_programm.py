# Import libraries
from scipy import signal
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
from matplotlib import cm
from obspy.signal.invsim import cosine_taper
from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset


def analyze_csv(test_filename, data_directory, output_data, show_plots=True):
    #TODO: dodać dynamiczny treshold 0. dla danych e-09, w lewo dla danych e-07, w prawo dla danych e-10

    # Load data from CSV file
    csv_file = f'{data_directory}{test_filename}.csv'
    data_cat = pd.read_csv(csv_file)

    # Load time steps and velocities
    csv_date = np.array(data_cat['time_abs(%Y-%m-%dT%H:%M:%S.%f)'].tolist())
    csv_times = np.array(data_cat['time_rel(sec)'].tolist())
    csv_data = np.array(data_cat['velocity(m/s)'].tolist())

    # Calculate maximum value of velocity(m/s)
    max_impact = np.max(np.abs(csv_data))

    # Set minimum and maximum frequency for bandpass filter
    minfreq = 0.5
    maxfreq = 1.0

    # Implementation of high-pass filter (Butterworth)
    def high_pass_filter(data, cutoff_frequency, sample_rate):
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_frequency / nyquist
        b, a = signal.butter(5, normal_cutoff, btype='high', analog=False)
        return signal.filtfilt(b, a, data)

    # Implementation of bandpass filter
    def bandpass_filter(data, lowcut, highcut, sample_rate):
        nyquist = 0.5 * sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = signal.butter(5, [low, high], btype='band')
        return signal.filtfilt(b, a, data)

    # Apply filters
    sample_rate = 1 / (csv_times[1] - csv_times[0])
    cutoff_frequency = 1.0
    csv_data_highpass = high_pass_filter(csv_data, cutoff_frequency, sample_rate)
    csv_data_bandpass = bandpass_filter(csv_data, minfreq, maxfreq, sample_rate)

    if show_plots:
        # Draw plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Function for plotting data and spectrogram
        def plot_data_and_spectrogram(ax_data, ax_spec, times, data, title, sta_len, lta_len):
            ax_data.plot(times, data)
            ax_data.set_ylabel('Velocity (m/s)')
            ax_data.set_title(title, fontweight='bold')
            
            # Drawing envelope
            analytic_signal = signal.hilbert(data)
            amplitude_envelope = np.abs(analytic_signal)
            
            # STA/LTA algorithm
            df = sample_rate
            cft = classic_sta_lta(amplitude_envelope, int(sta_len * df), int(lta_len * df))
            
            # Setting thresholds for trigger
            thr_on = 0.9
            thr_off = 0.1
            
            # Adjusting classic_sta_lta result
            on_off = trigger_onset(cft, thr_on, thr_off)
            
            # Marking triggers on the plot
            for i in range(len(on_off)):
                triggers = on_off[i]
                trigger_on_abs = datetime.strptime(csv_date[triggers[0]], '%Y-%m-%dT%H:%M:%S.%f')
                trigger_off_abs = datetime.strptime(csv_date[triggers[1]], '%Y-%m-%dT%H:%M:%S.%f')
                ax_data.axvline(x=times[triggers[0]], color='red', label=f'Trig. On: {trigger_on_abs}' if i == 0 else "")
                ax_data.axvline(x=times[triggers[1]], color='purple', label=f'Trig. Off: {trigger_off_abs}' if i == 0 else "")
            
            ax_data.legend()
            
            f, t, sxx = signal.spectrogram(data, fs=sample_rate)
            ax_spec.pcolormesh(t, f, sxx, cmap=cm.jet, vmax=5e-17)
            ax_spec.set_ylabel('Frequency (Hz)')
            ax_spec.set_xlabel('Time (s)')
            
            # Marking triggers on the spectrogram
            for i in range(len(on_off)):
                triggers = on_off[i]
                ax_spec.axvline(x=times[triggers[0]], color='red')
                ax_spec.axvline(x=times[triggers[1]], color='purple')

            # Save detections to CSV file
            starttime = datetime.strptime(csv_date[0], '%Y-%m-%dT%H:%M:%S.%f')
            detection_times = []
            fnames = []
            for i in range(len(on_off)):
                triggers = on_off[i]
                on_time = starttime + timedelta(seconds=csv_times[triggers[0]])
                on_time_str = datetime.strftime(on_time, '%Y-%m-%dT%H:%M:%S.%f')
                detection_times.append(on_time_str)
                fnames.append(test_filename)

            detect_df = pd.DataFrame(data={'filename': fnames, 'time_abs(%Y-%m-%dT%H:%M:%S.%f)': detection_times, 'time_rel(sec)': csv_times[triggers[0]]})
            detect_df.to_csv(f'{output_data}detections.csv', index=False)

        # STA/LTA algorithm
        sta_len = 120
        lta_len = 900


        # Original data
        plot_data_and_spectrogram(axes[0, 0], axes[1, 0], csv_times, csv_data, 'Original data', sta_len, lta_len)

        # Data after bandpass filter
        plot_data_and_spectrogram(axes[0, 1], axes[1, 1], csv_times, csv_data_bandpass, f'After bandpass filter ({minfreq}-{maxfreq} Hz)', sta_len, lta_len)

        # Beautify plots
        for ax_row in axes:
            for ax in ax_row:
                ax.set_xlim([min(csv_times), max(csv_times)])

        plt.tight_layout()
        plt.savefig(f'{output_data}{test_filename}_plots.png')
        plt.close()

    return max_impact

# Example usage of the function (can be done in a loop):
test_filename = 'xa.s12.00.mhz.1969-12-16HR00_evid00006'
data_directory = 'C:/Users/popos/Desktop/moon/test/data/S12_GradeB/'
output_data = 'C:/Users/popos/Desktop/out/'
max_impact = analyze_csv(test_filename, data_directory, show_plots=True, output_data=output_data)
print(f"Maximum absolute value of velocity(m/s): {max_impact}")

