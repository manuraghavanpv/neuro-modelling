import numpy as np
import matplotlib.pyplot as plt
import time

# Function to generate synthetic EEG data
def generate_eeg_signal(length, fs):
    t = np.arange(0, length, 1/fs)
    # Generate random EEG signal
    eeg_signal = np.random.randn(len(t))
    return t, eeg_signal

# Function to simulate cursor movement based on EEG signal
def move_cursor(eeg_signal):
    cursor_position = 0
    for signal_value in eeg_signal:
        # Update cursor position based on EEG signal
        cursor_position += signal_value
        # Ensure cursor stays within bounds
        if cursor_position < 0:
            cursor_position = 0
        elif cursor_position > 100:
            cursor_position = 100
        # Display cursor position
        print(f"Cursor position: {cursor_position}")
        # Add delay to simulate real-time processing
        time.sleep(0.1)

# Main function
def main():
    # Generate synthetic EEG signal
    length = 10  # Length of EEG signal in seconds
    fs = 100     # Sampling frequency (Hz)
    t, eeg_signal = generate_eeg_signal(length, fs)

    # Plot EEG signal
    plt.figure(figsize=(10, 4))
    plt.plot(t, eeg_signal)
    plt.title('Synthetic EEG Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Simulate cursor movement based on EEG signal
    move_cursor(eeg_signal)

if __name__ == "__main__":
    main()
