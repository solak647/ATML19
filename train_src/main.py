from generate_data.spectrogram import Spectrogram
from generate_data.data_folder import DataFolder


def main():

    # Do you want to generate the data folder from wav file? If yes, you need to have wav files.
    generate_data = True

    if generate_data:
        wav_directory = '../genres_wav'
        spectrogram_directory = '../genres_spectrograms'
        data_directory = 'data'
        # Create spectrogram from wav files
        #spectrogram = Spectrogram(spectrogram_directory)
        #spectrogram.create(wav_directory)

        # Create data folder with train, test and val directory
        data_folder = DataFolder(data_directory, False)
        data_folder.generate(spectrogram_directory, train_pourcent=0.6, val_pourcent=0.1)

    


if __name__ == '__main__':
	main()
