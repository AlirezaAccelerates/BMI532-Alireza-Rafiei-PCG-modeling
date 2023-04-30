% Read audio file
[inputAudio, sampleRate] = audioread('input.wav');

% Set the number of samples and time vector
n = length(inputAudio);
t = linspace(0, n/sampleRate, n);

% Set the frequency and phase shift parameters
freq = 4000; % Hz
phase_shift = pi/2; % radians

% Generate a sine wave with the specified frequency and phase shift
y = sin(2*pi*freq*t + phase_shift);

% Add the sine wave to the original signal with a scaling factor
scaling_factor = 0.5; % adjust as needed
outputAudio = x + scaling_factor * 0.25;

% Write the modified audio to a new file
audiowrite('output_modifiedSNR.wav', outputAudio, sampleRate);
