# some parts of the code were created by Prof Stephen Ficaks, University of Oregon.
import pandas as pd
import random
import functools

def row_to_vect(row, features):
    vect = []
    for feature in features:
        vect.append(float(row[feature]))
    return tuple(vect)

def initialize_centroids(sample_table, features):
    k = len(sample_table)
    centroids = {}
    for i in range(k):
        row = sample_table.iloc[i]
        vector = row_to_vect(row, features)
        centroids[i] = {'centroid': vector, 'cluster': []}
        
    return centroids

def euclidean_distance(vect1, vect2):
    sum = 0
    for i in range(len(vect1)):
        sum += (vect1[i] - vect2[i])**2
    return sum**.5  # I claim that this square root is not needed in K-means - see why?

def closest_centroid(centroids, row, k):
    min_distance = euclidean_distance(centroids[0]['centroid'], row)
    min_centroid = 0
    for i in range(1,k):
        distance = euclidean_distance(centroids[i]['centroid'], row)
        if distance < min_distance:
            min_distance = distance
            min_centroid = i
    return (min_centroid, min_distance)

def phase_1(centroids, table, features, k):
    for i in range(k):
        centroids[i]['cluster'] = []  # starting new phase 1 so empty out values from prior iteration
    
    #Go through every row in Titanic table (or Loan table) and place in closest centroid cluster.
    for i in range(len(table)):
        row = table.iloc[i]
        vrow = row_to_vect(row, features)
        (index, dist) = closest_centroid(centroids, vrow, k)
        centroids[index]['cluster'].append(vrow)
    
    return centroids

#cluster is a list of points, i.e., a list of lists.
def compute_mean(cluster):
    if len(cluster) == 0:
        return []
    the_sum = cluster[0]  # use 0th point as starter
    
    #I am using zip to pair up all points then do addition
    for i in range(1,len(cluster)):
        the_sum = [pair[0]+pair[1] for pair in zip(the_sum, cluster[i])]
    n = len(cluster)*1.0
    the_mean_point = [x/n for x in the_sum]
    return the_mean_point


def phase_2(centroids, k, threshold):
    old_centroids = []
    
    stop = True
    #Compute k new centroids and check for stopping condition
    for i in range(k):
        current_centroid = centroids[i]['centroid']
        new_centroid = compute_mean(centroids[i]['cluster'])
        centroids[i]['centroid'] = new_centroid
        if euclidean_distance(current_centroid, new_centroid) > threshold:
            stop = False  # all it takes is one

    return (stop, centroids)


def k_means(table, features, k, hypers):
    n = 100 if 'n' not in hypers else hypers['n']
    threshold = 0.0 if 'threshold' not in hypers else hypers['threshold']
    
    centroid_table = table.sample(n=k, replace=False, random_state=100)  # only random choice I am making
    centroids = initialize_centroids(centroid_table, features)
    
    j = 0
    stop = False
    while( j < n and not stop):
        print('starting '+str(j+1))
        centroids = phase_1(centroids, table, features, k)
        (stop, centroids) = phase_2(centroids, k, threshold)
        j += 1
    print('done')
    return centroids

features_used = [u'duration',
        u'title_year', u'imdb_score', u'Action', u'Adventure',
       u'Animation', u'Biography', u'Comedy', u'Crime', u'Documentary',
       u'Drama', u'Family', u'Fantasy', u'Film-Noir', u'History', u'Horror',
       u'Music', u'Musical', u'Mystery', u'News', u'Romance', u'Sci-Fi',
       u'Sport', u'Thriller', u'War', u'Western', u'c_Afghanistan',
       u'c_Argentina', u'c_Aruba', u'c_Australia', u'c_Bahamas', u'c_Belgium',
       u'c_Brazil', u'c_Bulgaria', u'c_Cameroon', u'c_Canada', u'c_Chile',
       u'c_China', u'c_Colombia', u'c_Czech Republic', u'c_Denmark',
       u'c_Dominican Republic', u'c_Egypt', u'c_Finland', u'c_France',
       u'c_Georgia', u'c_Germany', u'c_Greece', u'c_Hong Kong', u'c_Hungary',
       u'c_Iceland', u'c_India', u'c_Indonesia', u'c_Iran', u'c_Ireland',
       u'c_Israel', u'c_Italy', u'c_Japan', u'c_Kyrgyzstan', u'c_Mexico',
       u'c_Netherlands', u'c_New Line', u'c_New Zealand', u'c_Norway',
       u'c_Official site', u'c_Panama', u'c_Peru', u'c_Philippines',
       u'c_Poland', u'c_Romania', u'c_Russia', u'c_Slovakia',
       u'c_South Africa', u'c_South Korea', u'c_Spain', u'c_Sweden',
       u'c_Taiwan', u'c_Thailand', u'c_UK', u'c_USA', u'c_West Germany',
       u'rating_NC-17', u'rating_PG', u'rating_PG-13', u'rating_R',
       u'rating_Unrated']

def compute_centroid_labels(centroids, focus_table, focus_column, features, k):
    for i in range(len(focus_table)):
        row = focus_table.iloc[i]
        vrow = row_to_vect(row, features)
        (minc, mind) = closest_centroid(centroids, vrow, k)
        if focus_column not in centroids[minc]:
            centroids[minc][focus_column] = [row[focus_column]*1.0]
        else:
            centroids[minc][focus_column].append(row[focus_column]*1.0)
    for ind in range(k): 
        if len(centroids[ind][focus_column]) == 0:
            centroids[ind]['mean_label'] = 0.0
        else:
            the_sum = centroids[ind][focus_column][0]
            for i in range(1, len(centroids[ind][focus_column])):
                the_sum += centroids[ind][focus_column][i]
            centroids[ind]['mean_label'] = the_sum/(len(centroids[ind][focus_column]) * 1.0)
            
    return centroids

def kmeans_fill(centroids, full_table, features, focus_column, k):
   
    def get_closest_clmn(row):
        vrow = row_to_vect(row, features)
        (minc, mind) = closest_centroid(centroids, vrow, k)
        
        return centroids[minc]['mean_label']
    
    new_table = pd.DataFrame(full_table)
    new_table['kmeans_'+focus_column] = new_table.apply(lambda row: get_closest_clmn(row) if pd.isnull(row[focus_column]) else row[focus_column], axis=1)
    
    return new_table



def forest_builder(table, column_choices, target, hypers):
    tree_n = 5 if 'total-trees' not in hypers else hypers['total-trees']
    m = int(len(column_choices)**.5) if 'm' not in hypers else hypers['m']
    k = hypers['max-depth'] if 'max-depth' in hypers else min(2, len(column_choices))
    gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0

    #build a single tree - call it multiple times to build multiple trees
    def iterative_build(k):
        train = table.sample(frac=1.0, replace=True)
        train = train.reset_index()
        left_out = table.loc[~table.index.isin(train['index'])]
        left_out = left_out.reset_index() # this gives us the old index in its own column
        oob_list = left_out['index'].tolist()  # list of row indices from original titanic table
        
        rcols = random.sample(column_choices, m)  # subspcace sampling
        columns_sorted = find_best_splitter(train, rcols, target)
        (best_column, gig_value) = columns_sorted[0]

        #Note I add _1 or _0 to make it more readable for debugging
        current_paths = [{'conjunction': [(best_column+'_1', build_pred(best_column, 1))],
                          'prediction': None,
                          'gig_score': gig_value},
                         {'conjunction': [(best_column+'_0', build_pred(best_column, 0))],
                          'prediction': None,
                          'gig_score': gig_value}
                        ]
        k -= 1  # we just built a level as seed so subtract 1 from k
        tree_paths = []  # add completed paths here

        while k>0:
            new_paths = []
            for path in current_paths:
                conjunct = path['conjunction']  # a list of (name, lambda)
                before_table = generate_table(train, conjunct)  #the subtable the current conjunct leads to
                rcols = random.sample(column_choices, m)  # subspace
                columns_sorted = find_best_splitter(before_table, rcols, target)
                (best_column, gig_value) = columns_sorted[0]
                if gig_value > gig_cutoff:
                    new_path_1 = {'conjunction': conjunct + [(best_column+'_1', build_pred(best_column, 1))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_1 ) #true
                    new_path_0 = {'conjunction': conjunct + [(best_column+'_0', build_pred(best_column, 0))],
                                'prediction': None,
                                 'gig_score': gig_value
                                 }
                    new_paths.append( new_path_0 ) #false
                else:
                    #not worth splitting so complete the path with a prediction
                    path['prediction'] = compute_prediction(before_table, target)
                    tree_paths.append(path)
            #end for loop

            current_paths = new_paths
            if current_paths != []:
                k -= 1
            else:
                break  # nothing left to extend so have copied all paths to tree_paths
        #end while loop

        #Generate predictions for all paths that have None
        for path in current_paths:
            conjunct = path['conjunction']
            before_table = generate_table(train, conjunct)
            path['prediction'] = compute_prediction(before_table, target)
            tree_paths.append(path)
        return (tree_paths, oob_list)
    
    #let's build the forest
    forest = []
    for i in range(tree_n):
        (paths, oob) = iterative_build(k)
        forest.append({'paths': paths, 'weight': None, 'oob': oob})
        
    return forest

def vote_taker(row, forest):
    votes = {0:0, 1:0}
    for tree in forest:
        prediction = tree_predictor(row, tree)
        votes[prediction] += 1
    winner = 1 if votes[1]>votes[0] else 0  #ties go to 0
    return winner

def find_best_splitter(table, choice_list, target):
    gig_scores = map(lambda col: (col, gig(table, col, target)), choice_list)
    gig_sorted = sorted(gig_scores, key=lambda item: item[1], reverse=True)
    return gig_sorted

def probabilities(counts):
    count_0 = 0 if 0 not in counts else counts[0]  #could have no 0 values
    count_1 = 0 if 1 not in counts else counts[1]
    total = count_0 + count_1
    probs = (0,0) if total == 0 else (1.0*count_0/total, 1.0*count_1/total)  #build 2-tuple
    return probs

def gini(counts):
    (p0,p1) = probabilities(counts)
    sum_probs = p0**2 + p1**2
    gini = 1 - sum_probs
    return gini

def gig(starting_table, split_column, target_column):
    
    #split into two branches, i.e., two sub-tables
    true_table = starting_table.loc[starting_table[split_column] == 1]
    false_table = starting_table.loc[starting_table[split_column] == 0]
    
    #Now see how the target column is divided up in each sub-table (and the starting table)
    true_counts = true_table[target_column].value_counts()  # Note using true_table and not titanic_table
    false_counts = false_table[target_column].value_counts()  # Note using true_table and not titanic_table
    starting_counts = starting_table[target_column].value_counts() 
    
    #compute the gini impurity for the 3 tables
    starting_gini = gini(starting_counts)
    true_gini = gini(true_counts)
    false_gini = gini(false_counts)

    #compute the weights
    starting_size = len(starting_table.index)
    true_weight = 0.0 if starting_size == 0 else 1.0*len(true_table.index)/starting_size
    false_weight = 0.0 if starting_size == 0 else 1.0*len(false_table.index)/starting_size
    
    #wrap it up and put on a bow
    gig = starting_gini - (true_weight * true_gini + false_weight * false_gini)
    
    return gig

def build_pred(column, branch):
    return lambda row: row[column] == branch

def generate_table(table, conjunct):
    result_table = functools.reduce(lambda accum, pair: accum.loc[pair[1]], conjunct, table)  # accum starts as table
    return result_table

def compute_prediction(table, target):
    counts = table[target].value_counts()  # counts looks like {0: v1, 1: v2}
    if 0 not in counts and 1 not in counts:
        raise LookupError('Prediction impossible - Empty tree on leaf')
    if 0 not in counts:
        prediction = 1
    elif 1 not in counts:
        prediction = 0
    elif counts[1] > counts[0]:  # ties go to 0 (negative)
        prediction = 1
    else:
        prediction = 0

    return prediction

def tree_predictor(row, tree):
    
    #go through each path, one by one (could use a map instead of for loop?)
    for path in tree['paths']:
        conjuncts = path['conjunction']
        result = map(lambda tuple: tuple[1](row), conjuncts)
        if all(result):
            return path['prediction']
    raise LookupError('No true paths found for row: ' + str(row))

def predictor_case(row, pred, target):
    actual = row[target]
    prediction = row[pred]
    if actual == 0 and prediction == 0:
        case = 'true_negative'
    elif actual == 1 and prediction == 1:
        case = 'true_positive'
    elif actual == 1 and prediction == 0:
        case = 'false_negative'
    else:
        case = 'false_positive'
    return case

def f1(cases):
    dict_cases = cases.to_dict()  # easier to work with dict than series
    #the heart of the matrix
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']  # use isin method if working with Series
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']
    total_pos = tp+fn
    total_pos_predict = tp+fp
    
    #other measures we can derive
    recall = 0.0 if total_pos == 0 else 1.0*tp/total_pos  # positive correct divided by total positive in the table
    precision = 0.0 if total_pos_predict == 0 else 1.0*tp/total_pos_predict # positive correct divided by all positive predictions made
    recall_div = 0.0 if recall == 0 else 1.0/recall
    precision_div = 0.0 if precision == 0 else 1.0/precision
    sum_f1 = recall_div + precision_div
    f1 = 0.0 if sum_f1 == 0 else 2.0/sum_f1
    return f1

def informedness(cases):
    dict_cases = cases.to_dict()  # easier to work with dict than series
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']
    total_pos = tp+fn
    total_neg = tn+fp

    recall = 0.0 if total_pos == 0 else 1.0*tp/total_pos  # positive correct divided by total positive in the table
    specificty = 0.0 if total_neg == 0 else 1.0*tn/total_neg # negative correct divided by total negative in the table
    J = (recall + specificty) - 1
    return J

def accuracy(cases):
    dict_cases = cases.to_dict()
    tp = 0 if 'true_positive' not in dict_cases else dict_cases['true_positive']
    fn = 0 if 'false_negative' not in dict_cases else dict_cases['false_negative']
    tn = 0 if 'true_negative' not in dict_cases else dict_cases['true_negative']
    fp = 0 if 'false_positive' not in dict_cases else dict_cases['false_positive']

    return 1.0*(tp + tn)/(tp+tn+fp+fn)  #assumes at least one case exists

def execute_rf(ref, table, column_choices, target, hypers):
    forest = forest_builder(table, column_choices, target, hypers=hypers)
    table['forest_'+str(ref)] = table.apply(lambda row: vote_taker(row, forest), axis=1)
    table['forest_'+str(ref)+'_type'] = table.apply(lambda row: predictor_case(row, pred='forest_'+str(ref), target=target), axis=1)
    return (table, forest)

def test_rf(ref, table):
    forest_types = table['forest_'+str(ref)+'_type'].value_counts()  # returns a series
    return (accuracy(forest_types), f1(forest_types), informedness(forest_types))

splitter_columns = [ u'Action', u'Adventure',
       u'Animation', u'Biography', u'Comedy', u'Crime', u'Documentary',
       u'Drama', u'Family', u'Fantasy', u'Film-Noir', u'History', u'Horror',
       u'Music', u'Musical', u'Mystery', u'News', u'Romance', u'Sci-Fi',
       u'Sport', u'Thriller', u'War', u'Western', u'c_Afghanistan',
       u'c_Argentina', u'c_Aruba', u'c_Australia', u'c_Bahamas', u'c_Belgium',
       u'c_Brazil', u'c_Bulgaria', u'c_Cameroon', u'c_Canada', u'c_Chile',
       u'c_China', u'c_Colombia', u'c_Czech Republic', u'c_Denmark',
       u'c_Dominican Republic', u'c_Egypt', u'c_Finland', u'c_France',
       u'c_Georgia', u'c_Germany', u'c_Greece', u'c_Hong Kong', u'c_Hungary',
       u'c_Iceland', u'c_India', u'c_Indonesia', u'c_Iran', u'c_Ireland',
       u'c_Israel', u'c_Italy', u'c_Japan', u'c_Kyrgyzstan', u'c_Mexico',
       u'c_Netherlands', u'c_New Line', u'c_New Zealand', u'c_Norway',
       u'c_Official site', u'c_Panama', u'c_Peru', u'c_Philippines',
       u'c_Poland', u'c_Romania', u'c_Russia', u'c_Slovakia',
       u'c_South Africa', u'c_South Korea', u'c_Spain', u'c_Sweden',
       u'c_Taiwan', u'c_Thailand', u'c_UK', u'c_USA', u'c_West Germany',
       u'rating_NC-17', u'rating_PG', u'rating_PG-13', u'rating_R',
       u'rating_Unrated', 'budget_low', 'is_new', 'is_long',
        'budget_medium', 'budget_high', 'gross_low', 'gross_medium', 'gross_high']

def vote_taker_oob(row, forest):
    votes = {0:0, 1:0}
    for tree in forest:
        if row['index'] in tree['oob']:
            prediction = tree_predictor(row, tree)
            votes[prediction] += 1
    winner = 1 if votes[1]>votes[0] else 0  #ties go to 0
    return winner

def oob_testing(table, forest):
    
    #first create a union of oobs
    oob_df = pd.DataFrame(columns=table.columns)
    for tree in forest:
        oob_df = oob_df.append([table.loc[[x]] for x in tree['oob'] if x not in oob_df.index])
    
    #create a new index column
    oob_df = oob_df.reset_index()
    
    return oob_df
    #for each tree

#total of the oobs in the forest
def total_oob(forest):
    total = 0
    for tree in forest:
        total += len(tree['oob'])
    return total

def list_oob(forest):
    oob = []
    for tree in forest:
        oob.extend(tree['oob'])
    return oob

def test_oob(table, forest, target):
    oob = oob_testing(table, forest)
    oob['forest_oob'] = oob.apply(lambda row: vote_taker_oob(row, forest), axis=1)
    oob['forest_oob_type'] = oob.apply(lambda row: predictor_case(row, pred='forest_oob', target=target), axis=1)
    forest_oob_types = oob['forest_oob_type'].value_counts()
    #print(forest_oob_types.sum())  # length of testing table
    return (accuracy(forest_oob_types), f1(forest_oob_types), informedness(forest_oob_types))


def build_tree_iter(table, choices, target, hypers={} ):

    k = hypers['max-depth'] if 'max-depth' in hypers else min(4, len(choices))
    gig_cutoff = hypers['gig-cutoff'] if 'gig-cutoff' in hypers else 0.0
    
    def iterative_build(k):
        columns_sorted = find_best_splitter(table, choices, target)
        (best_column, gig_value) = columns_sorted[0]
        
        #Note I add _1 or _0 to make it more readable for debugging
        current_paths = [{'conjunction': [(best_column+'_1', build_pred(best_column, 1))],
                          'prediction': None,
                          'gig_score': gig_value},
                         {'conjunction': [(best_column+'_0', build_pred(best_column, 0))],
                          'prediction': None,
                          'gig_score': gig_value}
                        ]
        k -= 1  # we just built a level as seed so subtract 1 from k
        tree_paths = []  # add completed paths here
        
        while k>0:
            new_paths = []
            for path in current_paths:
                conjunct = path['conjunction']  # a list of (name, lambda)
                before_table = generate_table(table, conjunct)  #the subtable the current conjunct leads to
                columns_sorted = find_best_splitter(before_table, choices, target)
                (best_column, gig_value) = columns_sorted[0]
                if gig_value > gig_cutoff:
                    new_path_1 = {'conjunction': conjunct + [(best_column+'_1', build_pred(best_column, 1))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_1 ) #true
                    new_path_0 = {'conjunction': conjunct + [(best_column+'_0', build_pred(best_column, 0))],
                                'prediction': None,
                                 'gig_score': gig_value}
                    new_paths.append( new_path_0 ) #false
                else:
                    #not worth splitting so complete the path with a prediction
                    path['prediction'] = compute_prediction(before_table, target)
                    tree_paths.append(path)
            #end for loop
            
            current_paths = new_paths
            if current_paths != []:
                k -= 1
            else:
                break  # nothing left to extend so have copied all paths to tree_paths
        #end while loop

        #Generate predictions for all paths that have None
        for path in current_paths:
            conjunct = path['conjunction']
            before_table = generate_table(table, conjunct)
            path['prediction'] = compute_prediction(before_table, target)
            tree_paths.append(path)
        return tree_paths

    return {'paths': iterative_build(k), 'weight': None}


def execute_dt(ref, table, column_choices, target, hypers):
    tree = build_tree_iter(table, column_choices, target, hypers)
    table['tree_'+str(ref)] = table.apply(lambda row: tree_predictor(row, tree), axis=1)
    table['tree_'+str(ref)+'_type'] = table.apply(lambda row: predictor_case(row, pred='tree_'+str(ref), target=target), axis=1)
    return table

def test_dt(ref, table):
    tree_types = table['tree_'+str(ref)+'_type'].value_counts()  # returns a series
    return (accuracy(tree_types), f1(tree_types), informedness(tree_types))

def compute_training(slices, left_out):
    training_slices = []
    for i in range(len(slices)):
        if i == left_out:
            continue
        training_slices.append(slices[i])
    return pd.concat(training_slices)

def caser(table, tree, target):
    scratch_table = pd.DataFrame(columns=['prediction', 'actual'])
    scratch_table['prediction'] = table.apply(lambda row: tree_predictor(row, tree), axis=1)
    scratch_table['actual'] = table[target]  # just copy the target column
    cases = scratch_table.apply(lambda row: predictor_case(row, pred='prediction', target='actual'), axis=1)
    return cases.value_counts()

def k_fold_random(table, k, target, hypers, candidate_columns):
    result_columns = ['name', 'true_positive', 'false_positive', 'true_negative', 'false_negative', 'accuracy', 'f1', 'informedness']
    k_fold_results_table = pd.DataFrame(columns=result_columns)

    total_len = len(table.index)
    split_size = int(total_len/(1.0*k))
    slices = [pd.DataFrame(columns=table.columns) for x in range(k)]

    #generate the slices
    for ind in range(total_len):
        rand = random.randint(0,k-1)
        slices[rand] = slices[rand].append(table.loc[[ind]])
    
    #generate test results
    for i in range(k):
        test_table = slices[i]
        train_table = compute_training(slices, i)
        fold_tree = build_tree_iter(train_table, candidate_columns, target, hypers)  # train
        fold_cases = caser(test_table, fold_tree, target)  # test

        k_fold_results_table = k_fold_results_table.append(fold_cases,ignore_index=True)
        end = k_fold_results_table.last_valid_index()
        k_fold_results_table.name.iloc[end] =  'fold '+str(i+1)+' test'
        k_fold_results_table.accuracy.iloc[end] =  accuracy(fold_cases)
        k_fold_results_table.f1.iloc[end] =  f1(fold_cases)
        k_fold_results_table.informedness.iloc[end] =  informedness(fold_cases)
        
    k_fold_results_table.__doc__ = str(hypers)  # adds comment to remind me of hyper params used
    return k_fold_results_table

