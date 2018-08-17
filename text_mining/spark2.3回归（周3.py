import warnings
warnings.filterwarnings('ignore')
from pyspark import SparkContext,SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler,VectorIndexer
from pyspark.ml.regression import LinearRegression,GBTRegressor,RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
#创建sparksession
sparkConf = SparkConf()
sc = SparkContext(conf=sparkConf)
spark = SparkSession(sparkContext=sc).builder.appName('test').getOrCreate()
#读入数据集，后面这个必须写
data = spark.read.csv('hour.csv',header=True)
#删除列 就得这样一个个删除
data = data.drop('instant').drop('dteday').drop('yr').drop('casual').drop('registered')
#打印数据类型
# print(data.printSchema())
#修改数据类型  循环每一列通过col选中每一列中的每个值 转为double类型
data = data.select([col(i).cast('double') for i in data.columns])
# print(data.printSchema())

#划分数据集 随机划分
train,test = data.randomSplit([0.6,0.4])
# train.cache()
# test.cache()
#特征助理
features = data.columns[:-1]
#应该是 将多个列的特征转化为一列特征 列名为features
vectorAssembler = VectorAssembler(inputCols=features,outputCol='feature')
print('---------------------------------------------------------------------------------')
#建立模型    需要实际值的列 和 转化为一列后的特征
clf = LinearRegression(labelCol='cnt',featuresCol='feature')
#建立管道       放入将多列转化为一列的方式 以及 模型
line = Pipeline(stages=[vectorAssembler,clf])
#训练模型
model = line.fit(train)
#模型预测
result = model.transform(test)
#计算评估指标 要实际值
evluator = RegressionEvaluator(labelCol='cnt')
#放入实际值
print('均方根误差是：',evluator.evaluate(result))
print('---------------------------------------------------------------------------------')

# 使用随机森林
#应该是 要把一列特征 的每一个区分一定的类别并且给一个索引 随机森林得用这种！！！
vectorIndexer = VectorIndexer(inputCol='feature',
                              outputCol='features')   # 分类特征处理
rf = RandomForestRegressor(labelCol='cnt',featuresCol= 'features')
#管道里 先用转成一列的再用一列給分类索引的！！！
line = Pipeline(stages=[vectorAssembler,vectorIndexer,rf])
model = line.fit(train)
result = model.transform(test)
print('随机森林均方根误差是：',evluator.evaluate(result))

# 使用GBDT    同随机森林！！！！！！！！
gbt = GBTRegressor(labelCol='cnt',featuresCol='features')
line = Pipeline(stages=[vectorAssembler,vectorIndexer,gbt])
model = line.fit(train)
result = model.transform(test)
print('GBDT均方根误差是：',evluator.evaluate(result))

# 使用网格搜索
paramGrid = ParamGridBuilder().addGrid(gbt.maxDepth,[5,10]).build()

cv = CrossValidator(estimator=gbt, evaluator=evluator,
                                  estimatorParamMaps=paramGrid, numFolds=3)
pipeline = Pipeline(stages=[vectorAssembler,vectorIndexer,cv])
pipelineModel = pipeline.fit(train)
predicted = pipelineModel.transform(test)

print(predicted.select('cnt','prediction').show(10))
print('网格搜索后的均方根误差是：',evluator.evaluate(predicted))
