import tensorflow as tf
import os
from datetime import datetime

class Evaluation:

    def __init__(self, store_dir, name, stats = []):
        """
        Creates placeholders for the statistics listed in stats to generate tensorboard summaries.
        e.g. stats = ["loss"]
        """
        tf.reset_default_graph()
        self.sess = tf.Session()
        self.tf_writer = tf.summary.FileWriter(os.path.join(store_dir, "%s-%s" % (name, datetime.now().strftime("%Y%m%d-%H%M%S")) ))

        self.stats = stats
        self.pl_stats = {}
        
        for s in self.stats:
            self.pl_stats[s] = tf.placeholder(tf.float32, name=s)
            tf.summary.scalar(s, self.pl_stats[s])
            
        self.performance_summaries = tf.summary.merge_all()

    def write_episode_data(self, episode, eval_dict):
       """
        Write episode statistics in eval_dict to tensorboard, make sure that the entries in eval_dict are specified in stats.
        e.g. eval_dict = {"loss" : 1e-4}
       """
       my_dict = {}
       for k in eval_dict:
          assert(k in self.stats)
          my_dict[self.pl_stats[k]] = eval_dict[k]

       summary = self.sess.run(self.performance_summaries, feed_dict=my_dict)

       self.tf_writer.add_summary(summary, episode)
       self.tf_writer.flush()

    def close_session(self):
        self.tf_writer.close()
        self.sess.close()

