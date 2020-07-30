import tensorflow as tf
import numpy as np

class NGramTF(tf.Module):
    def __init__(self):
        self.oov_score = 0
        self.pseudo_count = 5.0
        pass

    def fit_tf_lookup_table(self, dist):
        keys_tensor = tf.constant(list(dist.keys()))
        vals_tensor = tf.constant(list(dist.values()))
        initializer = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
        table = tf.lookup.StaticHashTable(initializer, self.oov_score)
        return table

    def fit_word_casing(self, wordCasingLookup):
        indices = []
        values = []
        tokens = []
        for idx, (token, items) in enumerate(wordCasingLookup.items()):
            tokens.append(token)
            for j, item in enumerate(items):
                indices.append([idx, j])
                values.append(item)
        word_casing_lookup_shape = [len(tokens), max([len(item) for item in wordCasingLookup.values()])]
        word_casing_lookup_tf = tf.SparseTensor(indices=indices, values=values, dense_shape=word_casing_lookup_shape)
        word_casing_indices = tf.range(0, len(tokens))
        dense_word_casing_lookup_tf = tf.sparse.to_dense(word_casing_lookup_tf, default_value='')

        initializer = tf.lookup.KeyValueTensorInitializer(tf.constant(tokens), word_casing_indices)

        # TODO: this does ot work
        table = tf.lookup.StaticHashTable(initializer, -1)
        return table, dense_word_casing_lookup_tf

    def fit(self, wordCasingLookup, uniDist, backwardBiDist, forwardBiDist, trigramDist):
        self.word_casing, self.word_casing_lookup = self.fit_word_casing(wordCasingLookup)
        self.tf_uni_dist = self.fit_tf_lookup_table(uniDist)
        self.backwardBiDist = self.fit_tf_lookup_table(backwardBiDist)
        self.forwardBiDist = self.fit_tf_lookup_table(forwardBiDist)
        self.trigramDist = self.fit_tf_lookup_table(trigramDist)

    def get_unigram_score(self, possible_token):

        input_tensor = tf.constant(possible_token)

        nominator = tf.dtypes.cast(self.tf_uni_dist.lookup(input_tensor), tf.float32) + self.pseudo_count

        input_tensor_lower = tf.strings.lower(input_tensor)
        idx = self.word_casing.lookup(input_tensor_lower)
        alternative_tokens = self.word_casing_lookup[idx]
        alternative_tokens = tf.gather(alternative_tokens, tf.where(alternative_tokens != b''))

        # t_idx = self.word_casing_lookup.indices[idx]
        # alternative_tokens = tf.gather(self.word_casing_lookup.values, t_idx)

        denominator = self.tf_uni_dist.lookup(alternative_tokens)
        denominator_sum = tf.math.reduce_sum(tf.dtypes.cast(denominator, tf.float32) + self.pseudo_count)
        unigram_score = nominator / denominator_sum

        return unigram_score

    def get_alternative_tokens(self, possible_token_tensor):
        idx = self.word_casing.lookup(possible_token_tensor)
        if tf.get_static_value(tf.equal(idx, tf.constant([[-1]])))[0]:
            return tf.constant([[b'']])
        alternative_tokens = self.word_casing_lookup[idx]
        alternative_tokens = tf.gather(alternative_tokens, tf.where(alternative_tokens != b''))
        return alternative_tokens

    def compute_unigram_score(self, possible_token_tensor, alternative_tokens):
        nominator = tf.dtypes.cast(self.tf_uni_dist.lookup(possible_token_tensor), tf.float32) + self.pseudo_count
        denominator = self.tf_uni_dist.lookup(alternative_tokens)
        denominator_sum = tf.math.reduce_sum(tf.dtypes.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_bigram_backward_score(self, possible_token_tensor, prev_token_tensor, alternative_tokens):
        x = prev_token_tensor + tf.constant('_') + possible_token_tensor
        nominator = tf.dtypes.cast(self.backwardBiDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = prev_token_tensor + tf.constant('_') + alternative_tokens
        denominator = self.backwardBiDist.lookup(alternative_tokens_m)
        denominator_sum = tf.math.reduce_sum(tf.dtypes.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_bigram_forward_score(self, possible_token_tensor, next_token_tensor, alternative_tokens):
        x = possible_token_tensor + tf.constant('_') + tf.strings.lower(next_token_tensor)
        nominator = tf.dtypes.cast(self.forwardBiDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = alternative_tokens + tf.constant('_') + tf.strings.lower(next_token_tensor)
        denominator = self.forwardBiDist.lookup(alternative_tokens_m)
        denominator_sum = tf.math.reduce_sum(tf.dtypes.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def compute_trigram_score(self, possible_token_tensor, prev_token_tensor, next_token_tensor, alternative_tokens):
        x = prev_token_tensor + tf.constant('_') + possible_token_tensor + tf.constant('_') + tf.strings.lower(
            next_token_tensor)
        nominator = tf.dtypes.cast(self.trigramDist.lookup(x), tf.float32) + self.pseudo_count
        alternative_tokens_m = prev_token_tensor + tf.constant('_') + alternative_tokens + tf.constant(
            '_') + tf.strings.lower(next_token_tensor)
        denominator = self.trigramDist.lookup(alternative_tokens_m)
        denominator_sum = tf.math.reduce_sum(tf.dtypes.cast(denominator, tf.float32) + self.pseudo_count)
        score = nominator / denominator_sum
        return score

    def get_score(self, prev_token, possible_token, next_token):

        possible_token_l = tf.strings.lower(possible_token)
        alternative_tokens = self.get_alternative_tokens(possible_token_l)

        unigram_score = self.compute_unigram_score(possible_token, alternative_tokens)
        result = tf.math.log(unigram_score)

        if prev_token is not None:
            bigram_backward_score = self.compute_bigram_backward_score(possible_token, prev_token, alternative_tokens)
            result += tf.math.log(bigram_backward_score)

        if next_token is not None:
            bigram_forward_score = self.compute_bigram_forward_score(possible_token, next_token, alternative_tokens)
            result += tf.math.log(bigram_forward_score)

        if prev_token is not None and next_token is not None:
            trigram_score = self.compute_trigram_score(possible_token, prev_token, next_token, alternative_tokens)
            result += tf.math.log(trigram_score)
        return result

    @staticmethod
    def capitalaize_str(token_tensor):
        token_tensor = tf.reshape(token_tensor, [1])
        char_tensor = tf.compat.v1.string_split(token_tensor, delimiter='').values

        first_char = char_tensor[0]
        first_char_cap = tf.strings.upper(first_char)

        cap_char_tensor = tf.concat([tf.reshape(first_char_cap, [1]), char_tensor[1:]], 0)
        cap_tensor = tf.strings.reduce_join(cap_char_tensor)
        return tf.reshape(cap_tensor, [1])

    #@tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.string, name="input_text")])
    @tf.function
    def get_true_case(self, tokens_tensor):
        #tokens_tensor = tf.constant(tokens)

        # get first token capitalized
        cap_first_token = self.capitalaize_str(tokens_tensor[0])
        trueCasedTokens = cap_first_token
        tokens_tensor = tf.slice(tokens_tensor, [1], [-1])

        condition = lambda tokens_tensor, trueCasedTokens: tf.greater(tf.size(tokens_tensor), 0)

        def body(tokens_tensor, trueCasedTokens):
            cur_tokens_tensor = tf.concat([trueCasedTokens[-1:], tokens_tensor], 0)
            curToken = tf.get_static_value(tf.slice(cur_tokens_tensor, [1], [1]))[0]
            curToken = tf.get_static_value(tf.strings.lower(curToken))

            prevToken = tf.slice(cur_tokens_tensor, [0], [1])
            if tf.get_static_value(tf.greater(tf.size(cur_tokens_tensor), [3]))[0]:
                nextToken = tf.slice(cur_tokens_tensor, [2], [1])
            else:
                nextToken = None

            wordCasingLookup = tf.reshape(self.get_alternative_tokens(tf.constant(curToken)), [-1])
            if tf.get_static_value(tf.equal(tf.size(wordCasingLookup), [0]))[0]:
                trueCasedTokens = tf.concat([trueCasedTokens, tf.constant(curToken)], 0)
            if tf.get_static_value(tf.equal(tf.size(wordCasingLookup), [1]))[0]:
                trueCasedTokens = tf.concat([trueCasedTokens, wordCasingLookup], 0)
            else:
                scores = tf.map_fn(lambda x: self.get_score(prevToken, x, nextToken),
                                   wordCasingLookup, dtype=tf.float32)
                maxElementIndx = tf.get_static_value(tf.argmax(scores))[0]
                trueVariant = tf.slice(wordCasingLookup, [maxElementIndx], [1])
                trueCasedTokens = tf.concat([trueCasedTokens, trueVariant], 0)

            tokens_tensor = tf.slice(tokens_tensor, [1], [-1])
            return tokens_tensor, trueCasedTokens

        res = tf.while_loop(condition,
                            lambda tokens_tensor, trueCasedTokens: body(tokens_tensor, trueCasedTokens),
                            [tokens_tensor, trueCasedTokens])                    
        res = res[1]
        res = tf.gather(res, tf.where(res != b''))

        return res
