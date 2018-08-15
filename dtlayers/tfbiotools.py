import tensorflow as tf

def prob_to_logit(p):
    return tf.log(p/(1-p))


def complement_onehot_tcag(nuc):
    """Complement a nucleotide sequence encoded as an 
    onehot array (batch_size,seq_len,4)
    :param nuc: (batch_size,seq_len,4) onehot nucleotide tensor
    :returns:  (batch_size,seq_len,4) onehot nucleotide tensor
    :rtype: A tensor
    """
    shape = nuc.get_shape().as_list()
    nuc_len = shape[2]
    batch_size=shape[0]
    tc = tf.slice(nuc,[0,0,0],[-1,-1,2])
    ag = tf.slice(nuc,[0,0,2],[-1,-1,2])
    #print "tc shape", tc.get_shape().as_list()
    #print "ag shape", ag.get_shape().as_list()
    agtc = tf.concat([ag,tc],axis=2)
    #print "agtc shape", agtc.get_shape().as_list()
    return agtc


def reverse_nucs(nucs):
    """Reverse a nucleotide sequence encoded as an  
    onehot array (batch_size,seq_len,4)
    :param nuc: (batch_size,seq_len,4) onehot nucleotide tensor
    :returns:  (batch_size,seq_len,4) onehot nucleotide tensor
    :rtype: A tensor
    """
    #Note: Using this to confirm the reverse op is copy and not in place
    return tf.reverse(nucs,[1])

def revcom_onehot_tcag(nuc):
    """Reverse complement a nucleotide sequence encoded as an NHWC 
    onehot array (batch_size,seq_len,4)
    :param nuc: (batch_size,seq_len,4) onehot nucleotide tensor
    :returns:  (batch_size,seq_len,4) onehot nucleotide tensor
    :rtype: A tensor
    """

    return tf.reverse(complement_onehot_tcag(nuc),[1])



