import os
import warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
import tensorflow as tf


def run():
    print('state=gradient_tracking(fake_quant gradient op)')
    print('about to call fake_quant_with_min_max_vars_gradient with invalid min/max shapes')
    try:
        tf.quantization.fake_quant_with_min_max_vars_gradient(gradients=1, inputs=1, min=[1, 1], max=[1, 1])
    except Exception as exc:
        print(f'raised Python exception instead of abort: {type(exc).__name__}: {exc}')
        print('BUG_REPRODUCED' if False else 'NOT_REPRODUCED')
        return
    print('call returned normally')
    print('BUG_REPRODUCED' if False else 'NOT_REPRODUCED')
    return


run()
