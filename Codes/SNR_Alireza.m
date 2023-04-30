% Read audio file
[inputAudio, sampleRate] = audioread('input.wav');

% Target SNR in dB
targetSNR = 10; % Change this value to your desired SNR

% Calculate the current signal and noise powers
signalPower = rms(inputAudio).^2;
noisePower = signalPower / (10^(targetSNR/10));

% Create white noise with the desired power
noise = sqrt(noisePower) * randn(size(inputAudio));

% Mix the original audio with the generated noise
outputAudio = inputAudio + noise;

% Normalize the output audio to the original audio's maximum value
outputAudio = outputAudio / max(abs(outputAudio)) * max(abs(inputAudio));

% Write the modified audio to a new file
audiowrite('output_modifiedSNR.wav', outputAudio, sampleRate);
