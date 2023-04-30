% Load data
input_path = 'C:/Users/Alireza/Desktop/BMI 532/Data/pure_training_data/'; % Replace with path to your input data directory
output_path = 'C:/Users/Alireza/Desktop/BMI 532/Data/ChatGPT_data/'; % Replace with path to your output data directory
files = dir(fullfile(input_path, '*.wav'));
num_files = length(files);

% Define filter parameters
fc = 1000; % Cutoff frequency in Hz
fs = 4000; % Sample rate in Hz
[b, a] = butter(6, fc / (fs / 2)); % 6th order Butterworth filter

% Define time stretching parameters
stretch_factor = 1.2;

% Loop over files and apply filtering and time stretching
for i = 1:num_files
    % Load audio file
    filename = fullfile(input_path, files(i).name);
    [y, fs] = audioread(filename);
    
    % Apply filtering
    y_filtered = filtfilt(b, a, y);
    
    % Apply time stretching
    y_stretched = resample(y, stretch_factor*fs, fs);
    
    % Save synthetic data
    filename_filtered = fullfile(output_path, strcat('filtered_', files(i).name));
    filename_stretched = fullfile(output_path, strcat('stretched_', files(i).name));
    audiowrite(filename_filtered, y_filtered, fs);
    audiowrite(filename_stretched, y_stretched, stretch_factor*fs);
end
