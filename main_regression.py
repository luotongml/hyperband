#!/usr/bin/env python

"A regression example. Mostly the same, only importing from defs_regression."

import sys
# import cPickle as pickle
import pickle
from pprint import pprint

from hyperband import Hyperband
from defs_regression.meta import get_params, try_params
# Big logic is as follows:
# defs_regression.meta
# init space of regressors for sampling
# get_params will sample the regressor space
# try_params will run the sample parameter
# a tip here would be import get_params_{regressor_name} to delegate the get_params to the detailed implementation
# exec("from defs_regression.{} import get_params as get_params_{}".format("keras_ts", "keras_ts"))
# File structure as follows:
# meta{get_params, try_params};
# regressor {get_params_{regressor_name}, try_params_{regressor_name} }

try:
    output_file = sys.argv[1]
    if not output_file.endswith('.pkl'):
        output_file += '.pkl'
except IndexError:
    output_file = 'results.pkl'

print("Will save results to", output_file)

#

hb = Hyperband(get_params, try_params)
results = hb.run(skip_last=1)

print("{} total, best:\n".format(len(results)))

for r in sorted(results, key=lambda x: x['loss'])[:5]:
    print("loss: {:.2%} | {} seconds | {:.1f} iterations | run {} ".format(
        r['loss'], r['seconds'], r['iterations'], r['counter']))
    pprint(r['params'])
    print

print("saving...")

with open(output_file, 'wb') as f:
    pickle.dump(results, f)
