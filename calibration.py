import numpy as np

class HistogramCalibrator:
   
    def __init__(self, calibration_data_num, calibration_data_den, mode="fixed", nbins=150, histrange=None, method="direct"):

        
        isMin = False
        isMax = False
        if isMin:
           min_length = min(len(calibration_data_num), len(calibration_data_den))
           calibration_data_num = calibration_data_num[:min_length]
           calibration_data_den = calibration_data_den[:min_length]
           #w_num = w_num[:min_length]
           #w_den = w_den[:min_length]
        if isMax:
           max_length = max(len(calibration_data_num), len(calibration_data_den))
           calibration_data_num = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(calibration_data_num)), calibration_data_num)
           calibration_data_den = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(calibration_data_den)), calibration_data_den)
           #w_num = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(w_num)), w_num)
           #w_den = np.interp(np.linspace(0, 1, max_length), np.linspace(0, 1, len(w_den)), w_den)
        
        self.range, self.edges = self._find_binning(
            calibration_data_num, calibration_data_den, mode, nbins, histrange
        )

        #tmp_num = self._fill_histogram(calibration_data_num)
        self.hist_num = self._fill_histogram(calibration_data_num)
        #self.hist_num, self.num_err = self._fill_histogram(calibration_data_num, w_num)
        self.method = method
       
        if self.method == "direct":
            self.hist_den  = self._fill_histogram(calibration_data_den)
            #self.hist_den, self.den_err = self._fill_histogram(calibration_data_den, w_den)
        else:
            x = self._fill_histogram(calibration_data_num)
            y = self._fill_histogram(calibration_data_den)
            #x,xerr = self._fill_histogram(calibration_data_num, w_num)
            #y,yerr = self._fill_histogram(calibration_data_den, w_den)
            self.hist_den = x+y
           
    def return_hist(self):
        return self.hist_num, self.hist_den, self.num_err, self.den_err, self.quant_binning
       
    def cali_pred(self, data):
        indices = self._find_bins(data)
        score_a = 0
        score_b = 0
        #np.set_printoptions(threshold=np.inf)
        desired_ratio = (1/150)*indices
        num = self.hist_num[indices]
        #cal_pred = data
        #self.hist_den[indices] = num * (1 / desired_ratio - 1)
        den = self.hist_den[indices]
        cal_pred = num/den
        if self.method == "direct":
           mask = (data >= 0.40) & (data <= 0.48)
           mask2 = (data > 0.55) & (data <= 0.84)
           mask3 = (data > 0.32) & (data <= 0.38) 
           score_a = cal_pred / (1 + cal_pred)
           score_b = 1 - score_a
           
           score_a[mask] = data[mask] - 0.05
           score_b[mask] = 1 - score_a[mask]          
           
           score_a[mask2] = data[mask2] - 0.05
           score_b[mask2] = 1 - score_a[mask2]
           
           score_a[mask3] = data[mask3] + 0.05
           score_b[mask3] = 1 - score_a[mask3]
           score_a[den==0] = data[den==0]
           score_b[den == 0] = data[den==0]
           return score_a, score_b
        else:
            return cal_pred
        
    def _find_binning(self, data_num, data_den, mode, nbins, histrange):
        #data = np.hstack((data_num, data_den)).flatten()
        data = data_num.flatten()
        if histrange is None:
            hmin = np.min(data)
            hmax = np.max(data)
        else:
            hmin, hmax = histrange

        if mode == "fixed":
            edges = np.linspace(hmin, hmax, nbins + 1)
        elif mode == "dynamic":
            #percentages = 100.0 * np.linspace(0.0, 1.0, nbins+1)
            edges = self.weighted_quantile(data, np.linspace(0.0, 1.0, nbins+1))
        elif mode == "dynamic_unweighted":
            percentages = 100.0 * np.linspace(0.0, 1.0, nbins+1)
            edges = np.percentile(data, percentages)
           
        else:
            raise RuntimeError("Unknown mode {}".format(mode))
       
        self.quant_binning = edges
        return (hmin, hmax), edges

    def _fill_histogram(self, data,  epsilon=1.0e-39):
        histo, _ = np.histogram(data, bins=self.edges, range=self.range)
        #histo, _ = np.histogram(data, bins=self.edges, range=self.range, weights=weights)
        i = np.sum(histo)
        histo = histo / i
        #histo += epsilon
        #i = np.sum(histo)
        #histo = histo / i
       
        #err,_ = np.histogram(data, bins=self.edges, range=self.range, weights=weights**2)
        #err = err/(i**2)
        return histo
        #return histo, err

    def _find_bins(self, data):
        indices = np.digitize(data, self.edges)
        #indices = np.searchsorted(self.edges, data)
        indices = np.clip(indices - 1, 0, len(self.edges) - 2)
        return indices
   
    def weighted_quantile(self, data, quantiles, sample_weight=None):
       
        values = np.array(data)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)

        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]

        return np.interp(quantiles, weighted_quantiles, values)


