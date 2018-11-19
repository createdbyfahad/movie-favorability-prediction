# Many parts of the code were created by Prof Stephen Ficaks, University of Oregon.
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
