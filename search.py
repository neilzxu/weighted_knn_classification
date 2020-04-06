from multiprocessing import Pool
import numpy as np
from sklearn.metrics import f1_score
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


def f1_eval(X, y):
    return f1_score(X, y, average='macro', zero_division=0)


@classmethod
def search_dispatch(cls, name):
    return cls._REGISTRY[name]


_global_wknn = None


class SearcherRegistry:
    _REGISTRY = {}

    def search_dispatch(name):
        return SearcherRegistry._REGISTRY[name]

    def register_search(name=None):
        def _register(fn):
            if name is None:
                SearcherRegistry._REGISTRY[fn.__name__] = fn
            else:
                SearcherRegistry._REGISTRY[name] = fn
            return fn

        return _register


class Searcher(SearcherRegistry):
    def greedy_search(wknn, X, y, steps, make_wt_fn, eval_fn):
        best_metric, best_wt = eval_fn(wknn, X, y), wknn.wknn_weights
        history = [(best_metric, best_wt)]
        for step in range(steps):
            candidate_wts = make_wt_fn(best_wt)
            for idx in range(candidate_wts.shape[0]):
                wknn.set_weights(candidate_wts[idx])
                cur_metric = eval_fn(wknn, X, y)
                best_metric, best_wt = max([(cur_metric, candidate_wts[idx]),
                                            (best_metric, best_wt)],
                                           key=lambda x: x[0])
            history.append((best_metric, best_wt))
        wknn.set_weights(best_wt)
        return history

    @SearcherRegistry.register_search('random')
    def random_search(wknn, X, y, steps, step_size, samples):
        class_ct = wknn.knn.classes_.shape[0]

        def make_wt_fn(wt):
            random_dirs = np.random.uniform(0, 1, size=(samples, class_ct))
            random_dirs = random_dirs / np.linalg.norm(
                random_dirs, order=1, axis=1) * step_size
            out_wt = random_dirs + wt
            out_wt[out_wt < 0] = 0
            out_wt = out_wt / np.sum(out_wt, axis=1)
            return out_wt

        def eval_fn(wknn, X, y):
            return f1_eval(wknn.predict(X), y)

        return Searcher.greedy_search(wknn, X, y, steps, make_wt_fn, eval_fn)

    @SearcherRegistry.register_search('coordinate')
    def coordinate_search(wknn, X, y, steps, step_size):
        class_ct = wknn.knn.classes_.shape[0]

        def make_wt_fn(wt):
            out_wts = step_size * np.vstack(
                [np.eye(class_ct), -np.eye(class_ct)])
            out_wts += wt
            out_wts[out_wts < 0] = 0
            out_wts = out_wts / np.sum(out_wts, axis=1)[:, np.newaxis]
            return out_wts

        def eval_fn(wknn, X, y):
            return f1_eval(wknn.predict(X), y)

        return Searcher.greedy_search(wknn, X, y, steps, make_wt_fn, eval_fn)

    @staticmethod
    def grid_task(data):
        (X, y), weights = data
        weights = np.array(weights)
        _global_wknn.set_weights(weights)
        cur_f1 = f1_eval(_global_wknn.predict(X), y)
        return cur_f1

    @SearcherRegistry.register_search('grid')
    def grid_search(wknn, X, y, weights_list, p=None):
        best_wt = None
        best_f1 = None
        global _global_wknn
        _global_wknn = wknn
        f1s = []
        data_list = zip([(X, y) for _ in range(len(weights_list))],
                        weights_list)
        if p is not None:
            pool = Pool(p)
            task_iter = pool.imap(Searcher.grid_task, data_list, chunksize=10)
            f1s = list(task_iter)
            pool.close()
            pool.join()
        else:
            f1s = [Searcher.grid_task(item) for item in data_list]

        best_idx = max(list(enumerate(f1s)), key=lambda x: x[1])[0]
        best_weights = weights_list[best_idx]
        best_f1 = f1s[best_idx]

        return [(best_f1, best_weights)]
