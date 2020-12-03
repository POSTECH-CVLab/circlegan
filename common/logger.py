import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import time
import collections
import pickle
from datetime import datetime
from scipy.signal import savgol_filter

class Logger:

    def __init__(self):

        self.since_beginning = collections.defaultdict(lambda: {})
        self.since_last_flush = collections.defaultdict(lambda: {})

        self.iter = 0
        self.logfile = None
        self.logtime = True
        self.save_folder = ""

    def clear(self):
        self.since_beginning = collections.defaultdict(lambda: {})
        self.since_last_flush = collections.defaultdict(lambda: {})

    def tick(self, iter):
        self.iter = iter

    def enable_logscale_plot(self):
        self.logscale = True

    def set_dir(self, folder):
        self.save_folder = folder
        self.logfile = open(self.save_folder + "log.txt", 'a')

    def set_casename(self, name):
        self.casename = name

    def info(self, name, value):
        self.since_last_flush[name][self.iter] = value

    def log(self, str):
        if self.logfile is not None:
            self.logfile.write(str + '\n')
        print(str)

    def plot(self, bSmooth=False):

        for name in np.sort(list(self.since_beginning.keys())):

            x_vals = np.sort(list(self.since_beginning[name].keys()))
            y_vals = [self.since_beginning[name][x] for x in x_vals]

            if np.std(y_vals) == 0.0:
                pass

            def smooth(y, winsize):
                if winsize % 2 == 0:
                    winsize -= 1
                if winsize <= 5:
                    return y
                return savgol_filter(y, winsize, 1, mode='mirror')

            plt.clf()
            plt.rcParams['agg.path.chunksize'] = 20000
            plt.plot(x_vals, smooth(y_vals, len(y_vals) // 1000) if bSmooth else y_vals)
            plt.xlabel('iteration')
            plt.ylabel(name)
            plt.savefig(self.save_folder + name.replace(' ', '_') + '.pdf')

            if np.std(y_vals[-1000:]) > 0 and (np.max(y_vals) - np.min(y_vals)) / np.std(y_vals[-1000:]) > 1e3:
                plt.yscale('log')
                plt.savefig(self.save_folder + name.replace(' ', '_') + '_log.pdf')

    def flush(self):

        prints = []
        for name in np.sort(list(set(self.since_last_flush.keys()).union(set(self.since_beginning.keys())))):
            if self.since_last_flush.get(name) is not None:
                vals = self.since_last_flush.get(name)
                self.since_beginning[name].update(vals)

                if np.std(list(vals.values())) == 0.0:
                    if np.mean(list(vals.values())) == 0.0:
                        pass
                    else:
                        prints.append("{}: {:.4f}".format(name, np.mean(list(vals.values()))))
                else:
                    prints.append("{}: {:.4f}±{:.4f}".format(name, np.mean(list(vals.values())), np.std(list(vals.values()))))
            else:
                vals = self.since_beginning.get(name)
                prints.append("{}: {:.4f}".format(name, vals[np.sort(list(vals.keys()))[-1]]))

        loginfo = "{}  ITER: {}, {}".format((datetime.now().strftime('%Y-%m-%d %H:%M:%S  ') if self.logtime else '') + self.casename, self.iter, ", ".join(prints))
        self.log(loginfo)

        self.since_last_flush.clear()
        self.logfile.flush()

    def save(self):
        with open(self.save_folder + 'log.pkl', 'wb') as f:
            pickle.dump(dict(self.since_beginning), f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.save_folder + 'log.pkl', 'rb') as f:
            self.since_beginning.update(pickle.load(f))
