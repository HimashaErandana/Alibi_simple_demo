import numpy as np
import tensorflow as tf
def getAlibiModifiedEmeddings(slope,scores,distance,sequnece):
    head_outs = []
    num_heads = len(slope)
    for h in range(num_heads):
        #get slopes for each
        alibi_bias = slope[h] * distance

        biased_scores = scores + alibi_bias

        attention_weights = tf.nn.softmax(biased_scores, axis=-1) #scaling the attention scores into prob vals (0-1) suing softmax
        attention_weights = attention_weights.numpy()

        output = attention_weights @ sequnece
        head_outs.append(output)

        final_out = np.mean(head_outs, axis=0)
        return final_out