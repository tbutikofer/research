import python.base as base
import numpy as np
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import shutil
import tensorflow as tf

def calc_covariance(time_series, offset):
    if offset > 0:
        feature_count = time_series.shape[0]
        cov = np.cov(m=time_series[offset:], y=time_series[:-offset])[:feature_count,-feature_count:]
    else:
        cov = np.cov(time_series)
    return cov

def calc_stress(time_series, offset):
    if time_series.shape[1] > 1:
        if offset > 0:
            series_count = time_series.shape[1]
            corr = np.corrcoef(x=time_series[offset:], y=time_series[:-offset],rowvar=False)[:series_count,-series_count:]
        else:
            corr = np.corrcoef(time_series,rowvar=False)
            stress = corr - np.diag(np.ones(len(corr)))
    else:
        stress = None
    return stress

def calc_data_change(marketcode, days, offset, today):
    data_array, date_list, isin_list, issuer_list = base.get_clean_data(marketcode, days=3 * days, today_str=today)
    data = np.array(data_array)
    data_change = np.array([[]])
    if len(date_list) > 1:
        date_list = date_list[1:]
        data_change = np.log(data[1:] / data[:-1])
        # remove days without trading (data_change = 0 for all securities)
        trading_days = ~np.all(data_change == 0, axis=1)
        data_change = data_change[trading_days,:]
        date_list = date_list[trading_days]
        data_change = data_change[-days:,:]
        date_list = date_list[-days:]

        if offset > 0:
            securities_traded = np.logical_and(
                ~np.all(data_change[offset:] == 0, axis=0),
                ~np.all(data_change[:-offset] == 0, axis=0)
            )
        else:
            securities_traded = ~np.all(data_change == 0, axis=0)
        data_change = data_change[:,securities_traded]

        isin_list = isin_list[securities_traded]
        issuer_list = issuer_list[securities_traded]

    return data_change, date_list, isin_list, issuer_list

def calc_PCA_market_dynamics(marketcode, start_date_str, end_date_str, series_length, step_days, offset, num_components):
    start_date = datetime.strptime(start_date_str, base.DATE_FORMAT)
    end_date = datetime.strptime(end_date_str, base.DATE_FORMAT)
    date_range = (start_date + timedelta(x) for x in range(0,(end_date-start_date).days+1,step_days))
#    data_csv = [['date','securities','components'] + [str(x+1) for x in range(num_components)]]
    data_csv = [['date'] + ["PC"+str(x+1) for x in range(num_components)]]
    for date in date_range:
        date_str = date.strftime(base.DATE_FORMAT)
        print(date_str)
        data_change, _,  _, _ = calc_data_change(marketcode, series_length, offset, date_str)
        cov = calc_covariance(data_change, offset)
        eigen_value, _ = np.linalg.eigh(cov)
        eigen_value_proportion = np.flip(eigen_value, axis=0) / np.sum(eigen_value)
        relevant_component_num = len(eigen_value_proportion[eigen_value_proportion>0.01])
#        data_csv.append([date_str, str(len(cov)), str(relevant_component_num)]+[str(x) for x in eigen_value_proportion[:num_components]])
        data_csv.append([date_str]+[str(x) for x in eigen_value_proportion[:num_components]])
    filename = "market_pca_{}_{}-{}_{}".format(marketcode, base.date_stamp(start_date_str),base.date_stamp(end_date_str),series_length)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')
    return filename

def calc_momentum(stress, axis=None):
    momentum = [np.dot(stress[i],stress[i]) for i in range(len(stress))]
    if axis is None:
        momentum = np.sum(momentum)
    momentum = np.sqrt(momentum)/len(stress)
    return momentum

def calculate_market_dynamics(marketcode, correlation_length, offset, date_series):
    momentum_series = []
    for date in date_series:
        date_str = date.strftime(base.DATE_FORMAT)
        data_change, _, _, _ = calc_data_change(marketcode, correlation_length, offset, date_str)
        stress = calc_stress(data_change, offset)
        if stress is not None:
            momentum = calc_momentum(stress)
        else:
            momentum = 0.0
        momentum_series.append(momentum)
        security_count = stress.shape[0] if stress is not None else 0
        print("{} {} {:.6f}".format(date.strftime(base.DATE_FORMAT), security_count, momentum))

    return momentum_series

def calc_market_stress_series(marketcode, start_date_str, end_date_str, series_lengths, step_days, offset):
    date_series = base.create_date_series(start_date_str, end_date_str, step_days)
    header = ['date']
    data_series = np.array([[date.strftime(base.DATE_FORMAT) for date in date_series]])
    for correlation_length in series_lengths:
        header.append(str(correlation_length))
        momentum_series = calculate_market_dynamics(marketcode, correlation_length, offset, date_series)
        data_series = np.append(data_series, [momentum_series], axis=0)
    #    data_series.append({'id':correlation_length, 'data':[{'data':momentum_series}]})
    data_csv = np.append(np.array([header]),np.transpose(data_series),axis=0)   
    filename = "market_stress_{}_{}-{}".format(marketcode, base.date_stamp(start_date_str),base.date_stamp(end_date_str))
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')
    return filename

def cluster_data(data, num_clusters):
    def input_fn():
        return tf.data.Dataset.from_tensors(
            tf.convert_to_tensor(data, dtype=tf.float32)
            ).repeat(count=1)
    #    return tf.train.limit_epochs(tf.convert_to_tensor(data, dtype=tf.float32), num_epochs=1)

    shutil.rmtree('tmp_tf')
    kmeans = tf.contrib.factorization.KMeansClustering(num_clusters=num_clusters, use_mini_batch=False, model_dir='tmp_tf')    

    # train
    score_prev = 1.0
    while True:
        kmeans.train(input_fn)
    #    cluster_centers = kmeans.cluster_centers()
        score = kmeans.score(input_fn)
        if np.abs(1.0 - score/score_prev) < 0.000001:
            break
        score_prev = score

    cluster_indices = np.array(list(kmeans.predict_cluster_index(input_fn)))

    #reorder cluster indices
    cluster_norm = []
    for i in range(num_clusters):
        aa = data[cluster_indices==i,:][:,cluster_indices==i]
        cluster_norm.append(calc_momentum(aa))
    index_order = np.argsort(np.negative(cluster_norm))
    cluster_indices_sorted = np.zeros([len(cluster_indices)])
    for i in range(num_clusters):
        cluster_indices_sorted[cluster_indices==index_order[i]] = i
    return cluster_indices_sorted.astype(int)

def calc_market_stress(marketcode, date_str, correlation_length, num_clusters=0, offset=0):
    data_change, _, isin_list, _ = calc_data_change(marketcode, correlation_length, offset, date_str)
    corr = calc_stress(data_change, offset)

    if num_clusters>0:
        cluster_idx = cluster_data(corr, num_clusters)
        norm = np.average(corr, axis=0)
        data_sorted_idx = np.lexsort([np.negative(norm), cluster_idx])
        corr = corr[data_sorted_idx,:][:,data_sorted_idx]
        isin_list = isin_list[data_sorted_idx]

    header = ['isin'] + [isin for isin in isin_list]
    data_csv = np.array([header])
    for isin_idx, isin in enumerate(isin_list):
        data_csv = np.append(data_csv, np.concatenate(([[isin]], [corr[isin_idx]]), axis=1), axis=0)
    filename = "market_stress_{}_{}-{}".format(marketcode, base.date_stamp(date_str), correlation_length)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')
    return filename

def find_base_securities(marketcode, events, date_from, date_to, correlation_length):
    securities = base.get_securities(marketcode)
    date0 = datetime.strptime(date_from, base.DATE_FORMAT)
    date1 = datetime.strptime(date_to, base.DATE_FORMAT)

    offset = 0
    num_clusters = 4
    isin_cluster_events = []
    for event in events[1:]:
        date = datetime.strptime(event[1], base.DATE_FORMAT)
        if date0 <= date <= date1:
            isin_cluster = np.empty(len(securities)) * np.nan
            data_change, _, isin_list, _ = calc_data_change(marketcode, correlation_length, offset, event[1])
            corr = calc_stress(data_change, offset)
            cluster_idx = cluster_data(corr, num_clusters=num_clusters)
            for i in range(num_clusters):
                isin_cluster[np.in1d(securities[:,0], isin_list[cluster_idx == i])] = i
            isin_cluster_events.append(isin_cluster)
    isin_cluster_events = np.transpose(np.array(isin_cluster_events))
    all_clusters_defined = np.logical_not(np.any(np.isnan(isin_cluster_events), axis=1))
    securities = securities[all_clusters_defined]
    isin_cluster_events = isin_cluster_events[all_clusters_defined]

    header = ['isin','issuer']
    data_csv = np.array([header])
    for row in securities[np.all(isin_cluster_events == 0, axis=1)]:
        data_csv = np.append(data_csv, [row], axis=0)
    filename = "market_base_{}_{}-{}-{}".format(marketcode, base.date_stamp(date_from), base.date_stamp(date_to), correlation_length)
    np.savetxt('csv/'+filename+'.csv', data_csv, fmt='%s', delimiter=',')
    return filename

def quote_change_statistics_real(marketcode, change_range, trading_days, end_date):
    data_change, date_list, _, _ = calc_data_change(marketcode, trading_days, 0, end_date)
    data_change = data_change.flatten()
    hist, bins = np.histogram(data_change,bins=100, range = change_range, density=True)

    date_start = date_list[0].strftime(base.DATE_FORMAT)
    date_end = date_list[-1].strftime(base.DATE_FORMAT)
    bins_avg = (bins[1:]+bins[:-1])/2

    filename = "quote_change_statistics_real_{}_{}-{}".format(marketcode, base.date_stamp(date_start), base.date_stamp(date_end))
    np.savetxt('csv/'+filename+'.csv', np.vstack((bins_avg,hist)).T, fmt='%s', delimiter=',')
    return filename, (date_start, date_end), len(data_change)

def baseline_stress_quote_simulation(trading_days_list, securities, simulation_runs, dist_params, change_range):
    """
    Calculates baseline stress from synthetic quote changes.
    Determines base.distribution_correlation for synthetic correlation matrix.
    """
    def prob_dist_T1(x, a, b):
        return np.power(x, a) * b

    def prob_dist_T2(x, a, b):
        return a * x + b

    quote_change = base.quote_change_simulation_generator(dist_params, change_range)

    data_record = ['trading days', 'average stress', 'std stress']
    data_record = np.concatenate((data_record, ["fit param {}".format(i+1) for i in range(2)]))
    data = np.array([data_record])
    for trading_days in trading_days_list:
        stress_elements = []
        momentum_series = []
        for _ in range(simulation_runs):
            data_change = [next(quote_change) for _ in range(trading_days*securities)]
            data_change = np.reshape(data_change, (trading_days,securities))
            stress = calc_stress(data_change, offset=0)
            for i in range(securities-1):
                for j in range(i+1, securities):
                    stress_elements.append(stress[i,j])
#            stress_elements = np.concatenate((stress_elements, stress.flatten()))
            momentum = calc_momentum(stress)
            momentum_series.append(momentum)
        data_record = [trading_days, np.average(momentum_series), np.std(momentum_series)]
        hist, bins = np.histogram(stress_elements, bins=100, density=True)
        bins_avg = (bins[1:]+bins[:-1])/2
        popt, pcov = curve_fit(base.correlation_distribution, bins_avg, hist, p0=(1.9, 20.0))
        data_record = np.concatenate((data_record, popt))
        data = np.append(data, [data_record], axis=0)

    x = data[1:,0].astype(np.float)
    p1 = data[1:,3].astype(np.float)
    p2 = data[1:,4].astype(np.float)

    popt1, pcov1 = curve_fit(prob_dist_T1, x, p1, p0=(0.3, 0.5))
    popt2, pcov2 = curve_fit(prob_dist_T2, x, p2, p0=(0.4, -1.5))

    filename = "baseline_stress_quote_simulation_{}".format(["uniform","normal","","observed"][len(dist_params)])
    np.savetxt('csv/'+filename+'.csv', data, fmt='%s', delimiter=',')
    return filename, (popt1, np.sqrt(np.diag(pcov1))), (popt2, np.sqrt(np.diag(pcov2)))

def baseline_stress_correlation_simulation(securities, simulations, T):
    """ Calculates baseline stress from synthetic correlation matrix """
    correlation_generator = base.correlation_simulation_generator((0.52587211, 0.34470357, 0.40524949 , -1.46300512), T)
    momentum_series = []
    for _ in range(simulations):
        stress = np.zeros((securities, securities))
        for i in range(securities - 1):
            for j in range(i+1, securities):
                corr_value = next(correlation_generator)
                stress[i,j] = corr_value
                stress[j,i] = corr_value
        momentum = calc_momentum(stress)
        momentum_series.append(momentum)
    return np.average(momentum_series)
