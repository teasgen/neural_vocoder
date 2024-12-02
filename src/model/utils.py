def init_norm_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if "Conv" in classname:
        m.weight.data.normal_(mean, std)