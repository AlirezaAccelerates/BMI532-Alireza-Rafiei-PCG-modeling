import librosa
import librosa.display

# Load data
def load_data(file):

    samplerate, data = wavfile.read(file)

    return samplerate, data
  
input_directory = '/content/drive/MyDrive/BMI 532/Data/pure/the-circor-digiscope-phonocardiogram-dataset-1.0.3/training_data/'

murmurs_sec = []
outcomes_sec = []
features_sec = []
data_sec = []
data_mel_all = []
S_DB_all = []
data_sec_array = []

files = []
for f in os.listdir(input_directory):
    if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('wav'):
        files.append(f)
num_files = len(files)
for i, f in enumerate(files):
    print('    {}/{}...'.format(i+1, num_files))
    input_file = os.path.join(input_directory, f)
    samplerate, data = load_data(input_file)
    if i ==9:
      break

    sec = np.shape(data)[0]//4000

    for j in range(sec//3):

      murmurs_sec.append(murmurs[i])
      outcomes_sec.append(outcomes[i])
      features_sec.append(features[i])

      data_3s = data[3*4000*j:(4000*((3*j)+3)-1)]/1
      data_sec.append(data_3s)

      data_mel = librosa.feature.melspectrogram(y=data_3s, sr=4000, n_fft=512, hop_length=50, win_length=100, window='hamming', fmin = 0, fmax = 800, n_mels = 32)
      data_mel_all.append(data_mel)

      S_DB = librosa.power_to_db(data_mel)
      S_DB_all.append(S_DB)
      
np.save('/content/drive/MyDrive/BMI 532/Data/data_mel_all_phase.npy',data_mel_all_phase)
np.save('/content/drive/MyDrive/BMI 532/Data/data_sec_phase.npy',data_sec_phase)
np.save('/content/drive/MyDrive/BMI 532/Data/S_DB_all_phase.npy',S_DB_all_phase)
np.save('/content/drive/MyDrive/BMI 532/Data/murmurs_sec_phase.npy',murmurs_sec_phase)
np.save('/content/drive/MyDrive/BMI 532/Data/outcomes_sec_phase.npy',outcomes_sec_phase)
np.save('/content/drive/MyDrive/BMI 532/Data/features_sec_phase.npy',features_sec_phase)
