from chainmrp.chain_answer_net import ChainAnswerNet
from chainmrp.chain_mrp import ChainMRP
import numpy as np
import tensorflow as tf

MAX_TIME = 6
env = ChainMRP()
sess = tf.Session()
tdNet = ChainAnswerNet(load_path="/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/TDmodel_1482985007.2686365.ckpt",
                       max_time=MAX_TIME, obs_dim=env.obs_dim, scope="TDnet")
supNet = ChainAnswerNet(load_path="/home/john/code/pythonfiles/TDnets/chainmrp/checkpoints/supervisedmodel_1482983389.5431712.ckpt",
                        max_time=MAX_TIME, obs_dim=env.obs_dim, scope="Supervised")


steps = 0
loss_vals = []
episode_num = 0

while steps < 1000:
    obs_sequence = np.zeros(shape=[MAX_TIME, tdNet.input_size])
    obs = env.reset()
    obs_sequence = np.append(obs_sequence, [obs], axis=0)
    obs_sequence = np.delete(obs_sequence, [0], axis=0)
    episode_num += 1
    done = False
    term = False
    state = env.state

    while not done:
        td_predictions = tdNet.sess.run(tdNet.final_prediction, feed_dict={tdNet.input: [obs_sequence], tdNet.batch_size: 1})
        supervised_predictions = supNet.sess.run(supNet.final_prediction, feed_dict={supNet.input: [obs_sequence], supNet.batch_size: 1})
        print("TD Predictions = ", td_predictions)
        print("Supervised Predictions = ", supervised_predictions)
        loss = np.sum(np.square(td_predictions-supervised_predictions))
        print(loss)
        print("State = ", state)
        done=term
        a = input("Press Enter to continue")
        new_obs, reward, term = env.step()
        obs = new_obs
        obs_sequence = np.append(obs_sequence, [obs], axis=0)
        obs_sequence = np.delete(obs_sequence, [0], axis=0)
        steps +=1
        state = env.state