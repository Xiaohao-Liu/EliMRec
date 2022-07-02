"""
@author: Zhongchuan Sun
"""
import numpy as np
import pandas as pd
from util import DataIterator
from util import typeassert
from .cpp_evaluator import CPPEvaluator
from util.cython.tools import float_type, is_ndarray
from util import pad_sequences
from util.cython.arg_topk import arg_topk


metric_dict = {"Precision": 1, "Recall": 2, "MAP": 3, "NDCG": 4, "MRR": 5}
re_metric_dict = {value: key for key, value in metric_dict.items()}


class UniEvaluator(CPPEvaluator):
    """Cpp implementation `UniEvaluator` for item ranking task.

    Evaluation metrics of `UniEvaluator` are configurable and can
    automatically fit both leave-one-out and fold-out data splitting
    without specific indication:

    * **First**, evaluation metrics of this class are configurable via the
      argument `metric`. Now there are five configurable metrics: `Precision`,
      `Recall`, `MAP`, `NDCG` and `MRR`.

    * **Second**, this class and its evaluation metrics can automatically fit
      both leave-one-out and fold-out data splitting without specific indication.
      In **leave-one-out** evaluation, 1) `Recall` is equal to `HitRatio`;
      2) The implementation of `NDCG` is compatible with fold-out; 3) `MAP` and
      `MRR` have same numeric values; 4) `Precision` is meaningless.
    """

    @typeassert(user_train_dict=dict, user_test_dict=(dict, None.__class__))
    def __init__(self, dataset, user_train_dict, user_test_dict, user_neg_test=None,
                 metric=None, top_k=50, batch_size=1024, num_thread=8):
        """Initializes a new `UniEvaluator` instance.

        Args:
            user_train_dict (dict): Each key is user ID and the corresponding
                value is the list of **training items**.
            user_test_dict (dict): Each key is user ID and the corresponding
                value is the list of **test items**.
            metric (None or list of str): If `metric == None`, metric will
                be set to `["Precision", "Recall", "MAP", "NDCG", "MRR"]`.
                Otherwise, `metric` must be one or a sublist of metrics
                mentioned above. Defaults to `None`.
            top_k (int or list of int): `top_k` controls the Top-K item ranking
                performance. If `top_k` is an integer, K ranges from `1` to
                `top_k`; If `top_k` is a list of integers, K are only assigned
                these values. Defaults to `50`.
            batch_size (int): An integer to control the test batch size.
                Defaults to `1024`.
            num_thread (int): An integer to control the test thread number.
                Defaults to `8`.

        Raises:
             ValueError: If `metric` or one of its element is invalid.
        """
        # super(UniEvaluator, self).__init__(user_test_dict)
        super(UniEvaluator, self).__init__()
        if metric is None:
            metric = ["Precision", "Recall", "MAP", "NDCG", "MRR"]
        elif isinstance(metric, str):
            metric = [metric]
        elif isinstance(metric, (set, tuple, list)):
            pass
        else:
            raise TypeError("The type of 'metric' (%s) is invalid!" % metric.__class__.__name__)

        for m in metric:
            if m not in metric_dict:
                raise ValueError("There is not the metric named '%s'!" % metric)

        self.dataset = dataset
        self.user_pos_train = user_train_dict
        self.user_pos_test = user_test_dict
        self.user_neg_test = user_neg_test
        self.metrics_num = len(metric)
        self.metrics = [metric_dict[m] for m in metric]
        self.num_thread = num_thread
        self.batch_size = batch_size

        self.max_top = top_k if isinstance(top_k, int) else max(top_k)
        if isinstance(top_k, int):
            self.top_show = np.arange(top_k) + 1
        else:
            self.top_show = np.sort(top_k)

    def metrics_info(self):
        """Get all metrics information.

        Returns:
            str: A string consist of all metrics information， such as
                `"Precision@10    Precision@20    NDCG@10    NDCG@20"`.
        """
        metrics_show = ['\t'.join([("%s@"%re_metric_dict[metric] + str(k)).ljust(12) for k in self.top_show])
                        for metric in self.metrics]
        metric = '\t'.join(metrics_show)
        return "metrics:\t%s" % metric

    def evaluate(self, model, test_users=None):
        """Evaluate `model`.

        Args:
            model: The model need to be evaluated. This model must have
                a method `predict_for_eval(self, users)`, where the argument
                `users` is a list of users and the return is a 2-D array that
                contains `users` rating/ranking scores on all items.

        Returns:
            str: A single-line string consist of all results, such as
                `"0.18663847    0.11239596    0.35824192    0.21479650"`.
        """
        # B: batch size
        # N: the number of items
        test_users = test_users if test_users is not None else list(self.user_pos_test.keys())
        if not isinstance(test_users, (list, tuple, set, np.ndarray)):
            raise TypeError("'test_user' must be a list, tuple, set or numpy array!")

        test_users = DataIterator(test_users, batch_size=self.batch_size, shuffle=False, drop_last=False)
        batch_result = []
        
        # if model.pretrain:
        # #     top_items = self.dataset.top_items
        # #     longtail_items = self.dataset.longtail_items
        # #     count_dict = {}
        #     batch_group = []
        for batch_users in test_users:
            if self.user_neg_test is not None:
                candidate_items = [list(self.user_pos_test[u]) + self.user_neg_test[u] for u in batch_users]
                test_items = [set(range(len(self.user_pos_test[u]))) for u in batch_users]

                ranking_score = model.predict(batch_users, candidate_items)  # (B,N)
                ranking_score = pad_sequences(ranking_score, value=-np.inf, dtype=float_type)

                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)
            else:
                test_items = [self.user_pos_test[u] for u in batch_users]
                ranking_score = model.predict(batch_users, None)  # (B,N)
                if not is_ndarray(ranking_score, float_type):
                    ranking_score = np.array(ranking_score, dtype=float_type)

                # set the ranking scores of training items to -inf,
                # then the training items will be sorted at the end of the ranking list.
                for idx, user in enumerate(batch_users):
                    if user in self.user_pos_train:
                        train_items = self.user_pos_train[user]
                    else:
                        train_items  = []
                    ranking_score[idx][train_items] = -np.inf
            # if model.pretrain:
            #     # batch_top_items_rec = arg_topk(ranking_score, 20)
            #     # unique, counts = np.unique(batch_top_items_rec, return_counts=True)
            #     # for item, freq in zip(unique, counts):
            #     #     if item not in count_dict:
            #     #         count_dict[item] = 0
            #     #     count_dict[item] += freq

            #     ###### For grouping
            #     # 1/|l_test|
            #     num_test_per_user = np.array([len(test_items_per_user) for test_items_per_user in test_items])
            #     num_test_per_user_inv = np.power(num_test_per_user, -1.0)
            #     num_test_per_user_inv[np.isinf(num_test_per_user_inv)] = 0.
            #     diag_num_test_per_user_inv = np.diag(num_test_per_user_inv) 
            #     # |l_rec \intersection l_test|
            #     batch_count_per_group = np.zeros([len(test_items), 10], dtype=np.int32)
            #     batch_top_items_rec = arg_topk(ranking_score, 20)
            #     for u in range(len(test_items)):
            #         for i in batch_top_items_rec[u]:
            #             if i in test_items[u]:
            #                 batch_count_per_group[u, self.dataset.item_group_idx[i]] += 1
            #     # batch_group_result = self.dataset.item_group_idx[batch_top_items_rec]
            #     # batch_count_per_group = np.apply_along_axis(lambda x: np.bincount(x, minlength=10), axis=1, arr=batch_group_result)

            #     # |l_rec \intersection l_test| / |l_test|
            #     batch_count_per_group = np.matmul(diag_num_test_per_user_inv, batch_count_per_group)
            #     batch_group.append(batch_count_per_group)

            result = self.eval_score_matrix(ranking_score, test_items, self.metrics,
                                            top_k=self.max_top, thread_num=self.num_thread)  # (B,k*metric_num)
            batch_result.append(result)

        # if model.pretrain:
        #     # num_clicks_top_items = [count_dict[item] if item in count_dict else 0 for item in top_items]
        #     # df = pd.DataFrame(data={'itemID': top_items, 'freq': num_clicks_top_items})
        #     # df.to_csv(self.dataset.statistics_folder + 'top_items_%s_l%d.csv' % (model.model_name, model.n_layers), index=False)
        #     all_group = np.concatenate(batch_group, axis=0)
        #     final_group = np.mean(all_group, axis=0)
        #     group_buf = '\t'.join([("%.5f" % x).ljust(12) for x in final_group])
        #     print(group_buf)
        # concatenate the batch results to a matrix
        all_user_result = np.concatenate(batch_result, axis=0)  # (num_users, metrics_num*max_top)
        final_result = np.mean(all_user_result, axis=0)  # (1, metrics_num*max_top)

        final_result = np.reshape(final_result, newshape=[self.metrics_num, self.max_top])  # (metrics_num, max_top)
        final_result = final_result[:, self.top_show - 1]
        final_result = np.reshape(final_result, newshape=[-1])
        buf = '\t'.join([("%.8f" % x).ljust(12) for x in final_result])
        return final_result, buf