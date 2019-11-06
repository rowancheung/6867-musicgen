import tensorflow as tf
import tensorflow_datasets as tfds
import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta.models.music_vae.trained_model import NoExtractedExamplesError
import pickle as pkl
import numpy as np

tf.enable_eager_execution()

# Load the full GMD with MIDI only (no audio) as a tf.data.Dataset
dataset = tfds.load(
    name="groove/4bar-midionly",
    split=tfds.Split.TRAIN,
    try_gcs=True)

# features = dataset.take(20)
# for f in features:
# 	print(f['style'])

mel_16bar_models = {}
groovae_4bar_config = configs.CONFIG_MAP['groovae_4bar']
mel_16bar_models['groovae_4bar'] = TrainedModel(groovae_4bar_config, batch_size=1, 
	checkpoint_dir_or_path='/Users/rowancheung/Downloads/groovae_4bar.tar')

def play(note_sequence):
  mm.play_sequence(note_sequence, synth=mm.fluidsynth)

def convert_midi(midi):
	return mm.midi_io.midi_to_note_sequence(midi)

def gen_genre_vec(genre):
	vecs = None
	count = 0
	for f in dataset:
		if f['style']['secondary'].numpy() == genre:
			try: 
				if vecs is None:
					latent = mel_16bar_models['groovae_4bar'].encode([convert_midi(f['midi'].numpy())])
					vecs = np.array(latent[0])
					print(vecs.shape)
				else:
					latent = mel_16bar_models['groovae_4bar'].encode([convert_midi(f['midi'].numpy())])
					vecs = np.vstack((vecs, latent[0]))
				count += 1
			except NoExtractedExamplesError:
				pass
	print(count)
	print(vecs.shape)
	average =  np.mean(vecs, axis=0)
	print(average.shape)
	return average

def decode_genre_vec(vec):
	print('decoding')
	vec = tuple([vec[i, :] for i in range(vec.shape[0])])
	midi = mel_16bar_models['groovae_4bar'].decode(vec, length=50)
	play(midi[0])
	
if __name__ == '__main__':
	attribute_vec = gen_genre_vec(b'swing')
	vec = pkl.dump(attribute_vec, open('swing_attribute_vec.pkl', 'wb'))
	# decode_genre_vec(vec)


## IDEA: look for other attribute vectors - length, genre, time measurement, types of drums used
## IDEA: shared domain latent space with images or text
## IDEA: attribute vectors in other MusicVAE, i.e more complex music and other genres