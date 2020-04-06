import argparse
from multiprocessing import Pool
import os

from data import load_dataset
from search import Searcher, f1_eval
from wknn import WeightedKNN

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from tqdm import tqdm


def weight_gen(ct, spacing):
    def grid_iter(ct, mass):
        if ct == 1:
            yield [mass]
        else:
            for cur_mass in list(np.arange(0, mass, 1 / spacing)) + (
                [mass] if mass / spacing == 0. else []):
                for weights in grid_iter(ct - 1, mass - cur_mass):
                    yield [cur_mass] + weights

    return list(grid_iter(ct, 1))


wknn = None

if not os.path.exists('results'):
    os.mkdir('results')


def real_exp():

    spacing = 2
    classes = 7
    grid_weights = weight_gen(classes, spacing)
    steps = 25

    ks = list(range(20, 200, 20))
    kwargs_dicts = {
        'coordinate': {
            'steps': steps,
            'step_size': 0.02
        },
        'grid': {
            'weights_list': grid_weights,
            'p': 20
        }
    }

    init_weights = np.ones(classes) / classes
    init_weights[-1] = 1 - np.sum(init_weights[:-1])

    searches = ['coordinate', 'linear', 'none']

    # Load datasets
    REAL_DIR = '/home/neilxu/work/xc_data/uci_data/covtype'
    train_X, train_y = load_dataset(os.path.join(REAL_DIR, 'covtype.train'))
    dev_X, dev_y = load_dataset(os.path.join(REAL_DIR, 'covtype.dev'))
    test_X, test_y = load_dataset(os.path.join(REAL_DIR, 'covtype.test'))

    results = []
    for k in tqdm(ks, desc='K neighbors'):
        wknn = WeightedKNN(wknn_weights=init_weights, wknn_rate_fn=lambda x: k)
        wknn.fit(train_X, train_y)

        for search in tqdm(searches, desc='Algorithm run', leave=False):
            wknn.set_weights(init_weights)
            if search in ['coordinate', 'grid']:
                kwargs = kwargs_dicts[search]
                emp_f1s, wts = zip(*(Searcher.search_dispatch(search)
                                     (wknn, dev_X, dev_y, **kwargs)))
                emp_f1, wt = emp_f1s[-1], wts[-1]
            elif search == 'none':
                emp_f1 = f1_eval(wknn.predict(dev_X), dev_y)
                wt = init_weights
            elif search == 'linear':
                model = SGDClassifier(loss='log', penalty='none')
                model.fit(train_X, train_y)
                emp_f1 = f1_eval(model.predict(dev_X), dev_y)
                wt = (model.coef_, model.intercept_)

            if search != 'linear':
                test_f1 = f1_eval(wknn.predict(test_X), test_y)
            else:
                test_f1 = f1_eval(model.predict(test_X), test_y)

            results.append({
                'Empirical F1': emp_f1,
                'True F1': test_f1,
                'Algorithm': search,
                'Weight': wt
            })

            df = pd.DataFrame.from_records(results)
            df.to_pickle('results/real_no_grid_search.pkl')


def grid_exp(k):
    print(f"Grid exp on {k} neighbors")
    spacing = 12
    classes = 7
    grid_weights = weight_gen(classes, spacing)

    kwargs = {'weights_list': grid_weights, 'p': 60}

    init_weights = np.ones(classes) / classes
    init_weights[-1] = 1 - np.sum(init_weights[:-1])

    # Load datasets
    REAL_DIR = 'uci_data/covtype'
    train_X, train_y = load_dataset(os.path.join(REAL_DIR, 'covtype.train'))
    dev_X, dev_y = load_dataset(os.path.join(REAL_DIR, 'covtype.dev'))
    test_X, test_y = load_dataset(os.path.join(REAL_DIR, 'covtype.test'))

    wknn = WeightedKNN(wknn_weights=init_weights, wknn_rate_fn=lambda x: k)
    wknn.fit(train_X, train_y)

    wknn.set_weights(init_weights)
    emp_f1s, wts = zip(*(
        Searcher.search_dispatch('grid')(wknn, dev_X, dev_y, **kwargs)))
    emp_f1, wt = emp_f1s[-1], wts[-1]
    test_f1 = f1_eval(wknn.predict(test_X), test_y)

    results = [{
        'Empirical F1': emp_f1,
        'True F1': test_f1,
        'Algorithm': 'grid',
        'Weight': wt
    }]

    df = pd.DataFrame.from_records(results)
    df.to_pickle(f'results/real_grid_search_{k}.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', type=str, help='type of experiment to run')
    parser.add_argument('--k', type=int, help='number of neighbors')
    args = parser.parse_args()
    if args.exp == 'grid':
        grid_exp(args.k)
    else:
        real_exp()
