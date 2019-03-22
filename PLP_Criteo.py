# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 11:09:02 2018

@author: pmeziane
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import time
from pyspark.sql.types import *
import numpy as np
import pandas as pd
import statistics

# =============================================================================
# The idea of the code is to preprocess huge csv files sent by Criteo.
# Usually, the data is really dirty and processing it through Pandas takes hours
# (if it is able to fit in the memory).
# Hence, we have decided to use Spark (and do our PLP project on this subject as well).
# =============================================================================

# =============================================================================
# To understand the logic here, it is important to know that an user_id corresponds
# to a device (like a computer, a mobile phone, ...) and is the 'real' device identifier.
# The xd_id however is computed by Criteo and is supposed to represent the physical
# person behind. Hence, a single xd_id is composed of several user_id. The goal
# of the project is to perform anomaly detection on xd_ids (beause sometimes,
# user_id are missclassified as being part of the wrong xd_id). 
# This first approach focuses on gender anomalies (e.g. xd_id with 5 user_id
# considered as male and 1 as female is likely to be abnormal).
# =============================================================================

spark = SparkSession.builder.master('local[*]').getOrCreate()#.config("spark.executor.instances", "2").config("spark.executor.cores", "2").config('spark.sql.shuffle.partitions', '10').getOrCreate()

### LOADING DATA (nrows = 2000)
def load_dataset(path, delimiter = '\t', nrows = True):
    '''
    This function loads a csv into spark. It is possible to specify the number
    of rows to be loaded.
    
    Input: String (file Path)
    Output: Spark Dataframe
    '''
    
    schema = StructType([StructField('xd_id',StringType(),True),StructField('user_id',StringType(),True),StructField('product',StringType(),True),StructField('nb_views',StringType(),True)])
    df = spark.read.option('delimiter', delimiter).option('header', 'true').schema(schema).csv(path)
    if nrows == True:
        nrows = df.count()
    print('*** Loading {} rows of {} ***'.format(nrows, path))
    df = df.limit(nrows)
    return df

dset = load_dataset('C:/Users/Utilisateur/Documents/ECP/Projets/Criteo/Centrale XD Project/extract_dataset.csv')
#dset = load_dataset('C:/Users/Utilisateur/Documents/ECP/Projets/Criteo/test.csv')
### CLEANING DATA ###
def clean(dset):
    '''
    Cleans columns and filters products where there is not information available
    ('"[""null_null_null"",null,null,null]"' in product column).
    Extract informations from product column into several other columns.
    For the record, product column contains a list in string format.
    
    Input: Spark DataFrame
    Output: Spark DataFrame
    '''
    dset = dset.filter(dset['product'] != '[""null_null_null"",null,null,null]' ) #"[\"null_null_null\",null,null,null]"
    dset = dset.fillna({'nb_views':'1'}) # we consider that a product has been seen at least once

    split_col_1 = f.split(dset['product'], ',')
    dset = dset.withColumn('product_id', split_col_1.getItem(0))
    dset = dset.withColumn('product_gender', split_col_1.getItem(1))
    dset = dset.withColumn('product_gender_proba', split_col_1.getItem(2))
    dset = dset.withColumn('product_Gid', split_col_1.getItem(3))
    
    split_col_2 = f.split(dset['product_id'], '"')
    dset = dset.withColumn('product_id', split_col_2.getItem(1))
    
    #split_col_3 = f.split(dset['product_Gid'], ']')
    #dset = dset.withColumn('product_Gid', split_col_3.getItem(1))
    
    dset = dset.fillna({'product_Gid':'"null'})
    
    split_col_4 = f.split(dset['product_Gid'], '"')
    dset = dset.withColumn('product_Gid', split_col_4.getItem(1))
    
    dset = dset.filter('product_Gid is not null')
    
    split_col_5 = f.split(dset['product_gender_proba'], '"')
    dset = dset.withColumn('product_gender_proba', split_col_5.getItem(1))
    
    split_col_6 = f.split(dset['product_gender'], '"')
    dset = dset.withColumn('product_gender', split_col_6.getItem(1))
    
    #dset = dset.drop('_c0')
    
    return dset

dset = clean(dset)

### STATS ###
def compute_stats(dset):
    '''
    Display several statistics on the dataset distribution.
    
    Input: Spark DataFrame
    Output: None
    '''
    n_xd_id = dset.select('xd_id').distinct().count()
    avg_unique_produt_per_uid = dset.groupby('user_id').agg({'nb_views': 'sum'}).select(f.mean(f.col('sum(nb_views)')).alias('mean')).collect()[0]['mean']
    avg_produt_per_uid = dset.groupby('user_id').agg({'product': 'count'}).select(f.mean(f.col('count(product)')).alias('mean')).collect()[0]['mean']
    print ("numbre of cross_device on the sample : ", n_xd_id)
    print ("average number of views per user Id : ",avg_unique_produt_per_uid)
    print ("average unique products per user Id : ",avg_produt_per_uid)

#dset = compute_stats(dset)

#xd_id_group = dset.groupby('xd_id').agg(f.countDistinct('user_id'))
#product_group_per_views = dset.groupby('product_id').agg(f.sum('nb_views'))
#product_group_per_user_id = dset.groupby('product_id').agg(f.count('user_id'))

#xd_id_group_mean = xd_id_group.select(f.mean(f.col('count(DISTINCT user_id)')).alias('mean')).collect()[0]['mean']
#xd_id_group_max = xd_id_group.select(f.max(f.col('count(DISTINCT user_id)')).alias('max')).collect()[0]['max']
#xd_id_group_min = xd_id_group.select(f.min(f.col('count(DISTINCT user_id)')).alias('min')).collect()[0]['min']

### PLOTS ###
def plot(dset):
    '''
    Plot distributions. Extremely inefficient as it has to use pandas.
    
    Input: Spark DataFrame
    Output: None
    '''
    xd_id_group = dset.groupby('xd_id').agg(f.countDistinct('user_id'))
    product_group_per_views = dset.groupby('product_id').agg(f.sum('nb_views'))
    product_group_per_user_id = dset.groupby('product_id').agg(f.count('user_id'))
    xd_id_group.toPandas().plot()
    product_group_per_views.toPandas().plot()
    product_group_per_user_id.toPandas().plot()

#dset = plot(dset)

### OUTLIER CLEANING ###
def outlier_cleaning(dset):
    '''
    Removes outliers from a spark DataFrame, based on total number of view per
    products. Uses quantiles.
    
    Input: Spark DataFrame
    Output: Spark DataFrame
    '''
    product_group_per_views = dset.groupby('product_id').agg(f.sum('nb_views'))
    dset = dset.join(product_group_per_views, 'product_id')
    q1, qx = product_group_per_views.approxQuantile('sum(nb_views)', [0.25, 0.9], 0.05)
    seuil = qx + 1.5 * (qx - q1)
    dset = dset.filter(dset['sum(nb_views)'] < seuil)
    return dset

#dset = outlier_cleaning(dset)
 
### FEATURES EXTRACTION ###
def add_columns(dset):
    '''
    Add columns to spark DataFrame to perform other operations afterwards.
    Inputed columns are floats (else, compatibility problems with Apache Arrow).
    Inpute unknown gender products as unisex. Removes product without ponderation
    as it is not possible to perform gender detection on those.
    
    Input: Spark DataFrame
    Output: Spark DataFrame
    '''
    dset = dset.withColumn('ponderation', f.column('product_gender_proba') * f.column('nb_views'))
    dset = dset.withColumn('user_id_sex_MALE', f.lit(0.))
    dset = dset.withColumn('user_id_sex_FEMALE', f.lit(0.))
    dset = dset.withColumn('user_id_sex_UNISEX', f.lit(0.))
    dset = dset.fillna({'product_gender':'UNISEX'})
    dset = dset.filter('ponderation is not null')
    dset = dset.withColumn('validation_interval_MALE_b', f.lit(0.))
    dset = dset.withColumn('validation_interval_FEMALE_b', f.lit(0.))
    dset = dset.withColumn('validation_interval_UNISEX_b', f.lit(0.))
    dset = dset.withColumn('validation_interval_MALE_t', f.lit(1.))
    dset = dset.withColumn('validation_interval_FEMALE_t', f.lit(1.))
    dset = dset.withColumn('validation_interval_UNISEX_t', f.lit(1.))
    return dset

dset = add_columns(dset).cache()


@f.pandas_udf(dset.schema, f.PandasUDFType.GROUPED_MAP)
def set_uid_gender(pdf):
    '''
    Parallelized function to be applied on each user_id_group.
    The idea of this function is to compute how each user_id behaves: 
    x% like a male, y% like a female, z% like an unisex, with x+y+z = 100.
    We ponderate the weight of each product by its number of views (ponderation
    column's role).
    
    Input: Pandas DataFrame
    Output: Pandas DataFrame
    '''
    dataset_genred = pdf.groupby('product_gender').ponderation.sum()
    dataset_genred = dataset_genred.map(lambda x: min(float(x) / float(pdf.ponderation.sum()),1.))
    dic = dataset_genred.to_dict()
    for key in ['MALE', 'FEMALE', 'UNISEX']:
        pdf.loc[:,'user_id_sex_' + key] = dic.get(key, 0.)
    return pdf

def set_gender(py_dset):
    '''
    Parallelize set_uid_gender on each user_id group using Apache Arrow and
    vectorized pandas udf.
    
    Input: Spark DataFrame
    Output: Spark DataFrame
    '''
    uid_grouped = py_dset.groupby('user_id')
    return uid_grouped.apply(set_uid_gender)

dset = set_gender(dset).cache()

@f.pandas_udf(dset.schema, f.PandasUDFType.GROUPED_MAP)
def genderize(pdf):
    '''
    Parallelized function to be applied on each xd_id group.
    For each MALE, FEMALE, UNISEX, returns two values that correspond to extreme
    behaviour in the considered xd_id.
    If an user_id does not fit in this interval, it is likely to be classified
    in the wrong xd_id.
    
    Input: Pandas DataFrame
    Output: Pandas DataFrame
    '''
    def validation_interval(liste, seuil=2):
        '''
        Computes a confidence interval at 95% (if seuil=2), under the hypothesis
        that the list distribution is a gaussian.
        
        Input: List
        Output: List
        '''
        try:
            answer = [max(0.,statistics.mean(liste) - seuil * statistics.stdev(liste)), min(statistics.mean(liste) + seuil * statistics.stdev(liste),1.)]
        except:
            answer = [0.,1.]
        return answer
    pdf = pdf.drop_duplicates(subset = 'user_id')
    for key in ['MALE', 'FEMALE', 'UNISEX']:
        interval = validation_interval(pdf['user_id_sex_' + key])
        pdf.loc[:,'validation_interval_' + key +'_b'] = interval[0]
        pdf.loc[:,'validation_interval_' + key +'_t'] = interval[1]
    return pdf
    
def set_gender_reference(py_dset):
    '''
    Parallelize genderize on each xd_id group using Apache Arrow and
    vectorized pandas udf.
    
    Input: Spark DataFrame
    Output: Spark DataFrame
    '''
    xid_grouped = py_dset.groupby('xd_id')
    return xid_grouped.apply(genderize)
    
dset = set_gender_reference(dset).cache()

start = time.time()
dset.show()
stop = time.time()
print(stop-start)


### ANOMALY DETECTION

#dset_anomaly= dset.dropDuplicates(['user_id'])
#dset_anomaly = dset.filter((dset.user_id_sex_MALE < dset.validation_interval_MALE_b) | 
#        (dset.user_id_sex_MALE > dset.validation_interval_MALE_t) | 
#        (dset.user_id_sex_FEMALE < dset.validation_interval_FEMALE_b) | 
#        (dset.user_id_sex_FEMALE > dset.validation_interval_FEMALE_t) | 
#        (dset.user_id_sex_UNISEX < dset.validation_interval_UNISEX_b) | 
#        (dset.user_id_sex_UNISEX > dset.validation_interval_UNISEX_t))

#dset_anomaly.show()
#dset_anomaly.write.csv('pyspark_processed') # use spark.read.csv("pyspark_processed")