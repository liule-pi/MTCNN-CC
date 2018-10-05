import tensorflow as tf
import numpy as np
import save_model
import cv2
from tensorflow.python.platform import gfile


def q_net(input_img):
    g1 = tf.Graph()
    with tf.Session(graph=g1) as sess:
        pnet, _, _ = save_model.create_mtcnn(sess, None)
        out = pnet(input_img)
    return out

def q_net_frozen(input_img):
    g2 = tf.Graph()
    with tf.Session(graph=g2) as sess:
        with gfile.FastGFile("./ckpt/pnet/pnet_frozen.pb",'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            # Add the graph to the session
            tf.import_graph_def(graph_def, name='')

       # Get tensor from graph
        output0 = g2.get_tensor_by_name("pnet/conv4-2/BiasAdd:0")
        output1 = g2.get_tensor_by_name("pnet/prob1:0")

        out = sess.run([output0, output1], feed_dict={'pnet/input:0': input_img})
        return out

def main():
    filename = "./test.jpg"
    draw = cv2.imread(filename)
    img = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    img_data = (img - 127.5) * 0.0078125
    img_x = np.expand_dims(img_data, 0)
    img_y = np.transpose(img_x, (0,2,1,3))

    output = q_net(img_y)
    output_frozen = q_net_frozen(img_y)

    check_reg = (output[0] == output_frozen[0]).all()
    check_prob = (output[1] == output_frozen[1]).all()
    print(check_reg and check_prob)

if __name__ == "__main__":
    main()
