import tensorflow as tf

from models import efficientnet


def convert_to_lite(file_name, export_dir):
    model = tf.saved_model.load(export_dir)
    concrete_func = model.signatures[
        tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([1, 224, 224, 3])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_model = converter.convert()
    with open(file_name, 'wb') as f:
        f.write(tflite_model)


def convert_to_saved_model(checkpiont, export_dir):
    model = efficientnet.efficient_net_b0()
    model.build(input_shape=(None, 224, 224, 3))
    model.summary()
    model.load_weights(filepath=checkpiont)
    tf.saved_model.save(model, export_dir)


if __name__ == '__main__':
    export_dir = "saved_model/final"
    filename = 'models/tflite_models/bamboo.tflite'
    checkpiont = "saved_model/1699_0.95312"
    convert_to_saved_model(checkpiont, export_dir)
    convert_to_lite(filename, export_dir)
