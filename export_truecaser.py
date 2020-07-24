import sys

import tensorflow as tf
#sys.path.insert(1, './truecaser')
import truecaser_tf
import pickle
tf.compat.v1.enable_eager_execution()

def load_truecasing_model(model_filename):
    with open(model_filename, 'rb') as bin_file:
        uni_dist = pickle.load(bin_file)
        backward_bi_dist = pickle.load(bin_file)
        forward_bi_dist = pickle.load(bin_file)
        trigram_dist = pickle.load(bin_file)
        word_casing_lookup = pickle.load(bin_file)
        return word_casing_lookup, uni_dist, backward_bi_dist, forward_bi_dist, trigram_dist

truecaser_weights = 'en_truecasing_model.obj'
export_path = './truecaser_serving/1/'
wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist = load_truecasing_model(truecaser_weights)
tf_model = truecaser_tf.NGramTF()
tf_model.fit(wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist)
signature_def = tf_model.get_true_case.get_concrete_function(
        tf.TensorSpec(shape=(None), dtype=tf.string, name="input_text"))
tf.saved_model.save(tf_model,export_path,signatures={'serving_default': signature_def})
