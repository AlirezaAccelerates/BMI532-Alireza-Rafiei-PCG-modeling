import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
S_DB = librosa.power_to_db(data_mel)
librosa.display.specshow(S_DB, sr=4000, hop_length=50, x_axis='time', y_axis='mel', fmin = 0, fmax = 800,  cmap ='jet');
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Frequency (HZ)', fontsize=14)
plt.yticks([0,200,400,600,800]) 
plt.colorbar(format='%+2.0f dB')
plt.savefig("/content/drive/MyDrive/BMI 532/Data/orginal_bar.jpg",dpi = 300)
