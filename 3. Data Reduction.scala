// /opt/spark-2.1.3/bin/spark-shell --packages JMailloH:kNN_IS:3.0,djgarcia:SmartReduction:1.0


import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}


// Load Train & Test

val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
val rawDataTrain = sc.textFile(pathTrain)

val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
val rawDataTest = sc.textFile(pathTest)

// Train & Test RDDs

val train = rawDataTrain.map{line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}

val test = rawDataTest.map { line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}


// Encapsulate Learning Algorithms

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

def trainDT(train: RDD[LabeledPoint], test: RDD[LabeledPoint], maxDepth: Int = 5): Double = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxBins = 32

    val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    val labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    testAcc
}


import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.evaluation._
import org.apache.spark.rdd.RDD


def trainKNN(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int = 3): Double = {

    val numClass = train.map(_.label).distinct().collect().length
    val numFeatures = train.first().features.size

    val knn = kNN_IS.setup(train, test, k, 2, numClass, numFeatures, train.getNumPartitions, 2, -1, 1)
    val predictions = knn.predict(sc)
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision

    precision

}


/*****Instance Selection*****/

// FCNN_MR

import org.apache.spark.mllib.feature._

val k = 3 //number of neighbors

val fcnn_mr_model = new FCNN_MR(train, k)

val fcnn_mr = fcnn_mr_model.runPR()

fcnn_mr.persist()

fcnn_mr.count()


trainDT(fcnn_mr, test)

trainKNN(fcnn_mr, test)


// RMHC_MR

import org.apache.spark.mllib.feature._

val p = 0.1 // Percentage of instances (max 1.0)
val it = 100 // Number of iterations
val k = 3 // Number of neighbors

val rmhc_mr_model = new RMHC_MR(train, p, it, k, 48151623)

val rmhc_mr = rmhc_mr_model.runPR()

rmhc_mr.persist()

rmhc_mr.count()

trainDT(rmhc_mr, test)

trainKNN(rmhc_mr, test)


// SSMA-SFLSDE_MR

import org.apache.spark.mllib.feature._

val ssmasflsde_mr_model = new SSMASFLSDE_MR(train) 

val ssmasflsde_mr = ssmasflsde_mr_model.runPR()

ssmasflsde_mr.persist()

ssmasflsde_mr.count()

trainDT(ssmasflsde_mr, test)

trainKNN(ssmasflsde_mr, test)


/*****Discretization*****/

// /opt/spark-2.1.3/bin/spark-shell --packages JMailloH:kNN_IS:3.0,sramirez:spark-MDLP-discretization:1.3,sramirez:spark-infotheoretic-feature-selection:1.4.4



import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}


// Load Train & Test

val pathTrain = "file:///home/spark/datasets/susy-10k-tra.data"
val rawDataTrain = sc.textFile(pathTrain)

val pathTest = "file:///home/spark/datasets/susy-10k-tst.data"
val rawDataTest = sc.textFile(pathTest)

// Train & Test RDDs

val train = rawDataTrain.map{line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}

val test = rawDataTest.map { line =>
    val array = line.split(",")
    var arrayDouble = array.map(f => f.toDouble) 
    val featureVector = Vectors.dense(arrayDouble.init) 
    val label = arrayDouble.last 
    LabeledPoint(label, featureVector)
}


// Encapsulate Learning Algorithms

import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD

def trainDT(train: RDD[LabeledPoint], test: RDD[LabeledPoint], maxDepth: Int = 5): Double = {
    val numClasses = 2
    val categoricalFeaturesInfo = Map[Int, Int]()
    val impurity = "gini"
    val maxBins = 32

    val model = DecisionTree.trainClassifier(train, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins)

    val labelAndPreds = test.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testAcc = 1 - labelAndPreds.filter(r => r._1 != r._2).count().toDouble / test.count()
    testAcc
}


import org.apache.spark.mllib.classification.kNN_IS.kNN_IS
import org.apache.spark.mllib.evaluation._
import org.apache.spark.rdd.RDD


def trainKNN(train: RDD[LabeledPoint], test: RDD[LabeledPoint], k: Int = 3): Double = {

    val numClass = train.map(_.label).distinct().collect().length
    val numFeatures = train.first().features.size

    val knn = kNN_IS.setup(train, test, k, 2, numClass, numFeatures, train.getNumPartitions, 2, -1, 1)
    val predictions = knn.predict(sc)
    val metrics = new MulticlassMetrics(predictions)
    val precision = metrics.precision

    precision

}


// MDLP

import org.apache.spark.mllib.feature.MDLPDiscretizer

val categoricalFeat: Option[Seq[Int]] = None
val nBins = 25
val maxByPart = 10000

val mdlpDiscretizer = MDLPDiscretizer.train(train, categoricalFeat, nBins, maxByPart)
mdlpDiscretizer.thresholds

val mdlpTrain = train.map(i => LabeledPoint(i.label, mdlpDiscretizer.transform(i.features)))
mdlpTrain.first()

val mdlpTest = test.map(i => LabeledPoint(i.label, mdlpDiscretizer.transform(i.features)))
mdlpTest.first()


trainDT(mdlpTrain, mdlpTest)

trainKNN(mdlpTrain, mdlpTest)


/*****Feature Selection*****/

//ChiSq

import org.apache.spark.mllib.feature.ChiSqSelector

val numFeatures = 5
val selector = new ChiSqSelector(numFeatures)
val transformer = selector.fit(train)

val chisqTrain = train.map { lp => 
  LabeledPoint(lp.label, transformer.transform(lp.features)) 
}

val chisqTest = test.map { lp => 
  LabeledPoint(lp.label, transformer.transform(lp.features)) 
}

chisqTrain.first.features.size // 5

trainDT(chisqTrain, chisqTest)

trainKNN(chisqTrain, chisqTest)


// PCA

import org.apache.spark.mllib.feature.PCA

val numFeatures = 5

val pca = new PCA(5).fit(train.map(_.features))

val projectedTrain = train.map(p => p.copy(features = pca.transform(p.features)))
val projectedTest = test.map(p => p.copy(features = pca.transform(p.features)))

projectedTrain.first.features.size // 5
projectedTest.first.features.size // 5


trainDT(projectedTrain, projectedTest)

trainKNN(projectedTrain, projectedTest)


// mRMR

import org.apache.spark.mllib.feature._

val criterion = new InfoThCriterionFactory("mrmr")
val nToSelect = 5
val nPartitions = mdlpTrain.getNumPartitions

val featureSelector = new InfoThSelector(criterion, nToSelect, nPartitions).fit(mdlpTrain)

val reducedTrain = mdlpTrain.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))
reducedTrain.first()

val reducedTest = mdlpTest.map(i => LabeledPoint(i.label, featureSelector.transform(i.features)))


trainDT(reducedTrain, reducedTest)

trainKNN(reducedTrain, reducedTest)
