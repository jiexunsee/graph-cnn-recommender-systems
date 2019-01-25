from __future__ import division
from __future__ import print_function


def construct_feed_dict(placeholders, u_features, v_features, u_features_nonzero, v_features_nonzero,
                        support, support_t, labels, u_indices, v_indices, class_values,
                        dropout, u_features_side=None, v_features_side=None, E_start=None, E_end=None, E_start_nonzero=None, E_end_nonzero=None):
    """
    Function that creates feed dictionary when running tensorflow sessions.
    """

    feed_dict = dict()
    feed_dict.update({placeholders['u_features']: u_features})
    feed_dict.update({placeholders['v_features']: v_features})
    feed_dict.update({placeholders['u_features_nonzero']: u_features_nonzero})
    feed_dict.update({placeholders['v_features_nonzero']: v_features_nonzero})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['support_t']: support_t})

    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['user_indices']: u_indices})
    feed_dict.update({placeholders['item_indices']: v_indices})

    feed_dict.update({placeholders['dropout']: dropout})
    feed_dict.update({placeholders['class_values']: class_values})

    if (u_features_side is not None) and (v_features_side is not None):
        feed_dict.update({placeholders['u_features_side']: u_features_side})
        feed_dict.update({placeholders['v_features_side']: v_features_side})

    if E_start is not None and E_end is not None:
        for i in range(len(E_start)):
            feed_dict.update({placeholders['E_start_list'][i]: E_start[i]})
            feed_dict.update({placeholders['E_end_list'][i]: E_end[i]})
            feed_dict.update({placeholders['E_start_nonzero_list'][i]: E_start_nonzero[i]})
            feed_dict.update({placeholders['E_end_nonzero_list'][i]: E_end_nonzero[i]})
        # feed_dict.update({placeholders['E_start']: E_start})
        # feed_dict.update({placeholders['E_end']: E_end})

    return feed_dict
