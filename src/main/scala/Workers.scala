import akka.actor.{Actor, ActorLogging}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object Workers {
  case class Gradient (id:Int, grad: INDArray)
  case class Loaded()
  case class LoadData(listoffiles: Array[String], fileToLabel: collection.mutable.Map[String, Int])
  case class GetWorkerAccuracy(thetas: INDArray)
}

class Workers(inputpath: String) extends Actor with ActorLogging {

  import ParameterServer._
  import Workers._

  var xbatch: INDArray = _
  var ybatch: INDArray = _

  def backward(WXT:INDArray, thetas: INDArray): INDArray = {
    var N = xbatch.shape().toSeq(0)
    var k = thetas.shape().toSeq(0)
    var ones = Nd4j.ones(k,N)
    var delta = Nd4j.zeros(k,N)
    for (i <- 0 to N-1) {
      var j = ybatch.getInt(i,0)
      delta.putScalar(Array[Int](j, i), 1)
    }

    var intermediate = ones.sub(delta).add(WXT)
    var maxArgVal = Nd4j.zeros(N)
    var c = intermediate.columns()
    var r = intermediate.rows()

    for (j <- 0 to c-1) {
      var maxi = intermediate.getDouble(0,j)
      var index = 0
      for (i <- 1 to r-1) {
        if (maxi < intermediate.getDouble(i,j)) {
          maxi = intermediate.getDouble(i,j)
          index = i
        }
      }
      //print(index)
      maxArgVal.putScalar(Array[Int](j), index)
    }

    //var maxArgVal = Nd4j.getExecutioner().exec(new IAMax(ones.sub(delta).add(WXT)),0)
    //println(maxArgVal)

    var grad = Nd4j.zeros(k, xbatch.shape().toSeq(1))
    for (i <- 0 to k-1) {
      grad.putRow(i, thetas.getRow(i))
      if (i == 0) {
        //println(grad.getDouble(0,784))
      }
      var pos = 0
      var neg = 0
      //println(grad.getRow(i))
      for (j <- 0 to N-1) {

        if (maxArgVal.getInt(j) == i) {
          var intermediate = grad.getRow(i).add(xbatch.getRow(j))
          grad.putRow(i, intermediate)
          pos += 1
        }
        if (ybatch.getInt(j,0) == i) {
          var intermediate = grad.getRow(i).sub(xbatch.getRow(j))
          grad.putRow(i, intermediate)
          neg -= 1
        }


      }

    }
    grad
  }


  def forward(thetas: INDArray): INDArray = {
    var WXT = thetas.mmul(xbatch.transpose())
    WXT
  }


  def classify(thetas: INDArray): INDArray = {
    var prob = thetas.mmul(xbatch.transpose())

    var tags = Nd4j.zeros(xbatch.rows(), 1)
    var c = prob.columns()
    var r = prob.rows()

    for (j <- 0 to c-1) {
      var maxi = prob.getDouble(0,j)
      var index = 0
      for (i <- 1 to r-1) {
        if (maxi < prob.getDouble(i,j)) {
          maxi = prob.getDouble(i,j)
          index = i
        }
      }
      tags.putScalar(Array[Int](j,0), index)
    }
    tags
  }

  def accuracy(y: INDArray): Int = {
    var count = 0
    for (i <- 0 to y.size(0)-1) {
      if (y.getDouble(i,0) == ybatch.getDouble(i,0)) {
        count += 1
      }
    }
    count
  }

  def computeCost(thetas: INDArray, C: Double): Double  = {
    var magnitude = 0.5*Nd4j.diag(thetas.mmul(thetas.transpose())).sumNumber().doubleValue()

    var N = xbatch.shape().toSeq(0)
    var k = thetas.shape().toSeq(0)
    var ones = Nd4j.ones(k,N)
    var delta = Nd4j.zeros(k,N)
    for (i <- 0 to N-1) {
      var j = ybatch.getInt(i,0)
      delta.putScalar(Array[Int](j, i), 1)
    }

    var WXT = thetas.mmul(xbatch.transpose())
    var intermediate = ones.sub(delta).add(WXT)
    var maxArgVal = Nd4j.zeros(N)
    var c = intermediate.columns()
    var r = intermediate.rows()

    for (j <- 0 to c-1) {
      var maxi = intermediate.getDouble(0,j)
      var index = 0
      for (i <- 1 to r-1) {
        if (maxi < intermediate.getDouble(i,j)) {
          maxi = intermediate.getDouble(i,j)
          index = i
        }
      }
      //print(index)
      maxArgVal.putScalar(Array[Int](j), maxi)
    }

    var losspart2 = maxArgVal.sumNumber().doubleValue()

    var wd = WXT.mul(delta)
    var losspart3 = wd.sumNumber().doubleValue()
    //println(magnitude, losspart2, losspart3, magnitude + C*(losspart2-losspart3), C)
    magnitude + C*(losspart2-losspart3)
  }

  def receive = {
    case LoadData(listoffiles, maptofiles) => {
      var labels = Array[Double]()
      var total = 0
      for (f <- listoffiles) {
        labels = labels :+ maptofiles(f).toDouble
        var allnumbers = Array[Double]()
        var allfiles =  Source.fromFile(inputpath+f).getLines
        for (f <- allfiles) {
          allnumbers = allnumbers :+ f.toString().toDouble
        }
        if (total == 0) {
          xbatch = Nd4j.create(allnumbers)
          //print(f, allnumbers(0), xbatch)
        }
        else {
          xbatch = Nd4j.concat(0,xbatch,Nd4j.create(allnumbers))
        }
        total += 1
      }
      val bias = Nd4j.ones(xbatch.rows(), 1)
      xbatch =  Nd4j.concat(1, xbatch, bias)
      ybatch = Nd4j.create(labels, Array[Int](labels.size, 1))
      //print(ybatch.shape().toSeq, xbatch.shape().toSeq)
    }

    case NextBatch(id, thetas) => {
      val grad = backward(forward(thetas), thetas)
      sender() ! Gradient(id, grad)
    }

    case GetWorkerAccuracy(thetas) => {
      var score = classify(thetas)
      var correct = accuracy(score)
      var loss = computeCost(thetas, 1.0)
      sender() ! Accuracy(correct, loss)
    }
  }

}
