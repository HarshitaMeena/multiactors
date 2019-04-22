import akka.actor.{Actor, ActorRef, Props}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j

import scala.io.Source

object MultiClassSVMActor {
  case class fit(labelfile: String, inputpath: String, w_init: String, lr: Float, iters: Int, minibatch: Int, nw: Int, K: Int)
  case class DistributeData (iter: Int)
}

class MultiClassSVMActor extends Actor {

  import ParameterServer.UpdatedParameter
  private var combinedData: INDArray = _
  private var noOfWorkers: Int = _
  private var inputs: INDArray = _
  private var labels: INDArray = _
  private var batchSize, lr: Int = _
  var lrServer : ActorRef = _
  var t1 = 0.0
  var weights: INDArray = _
  var fileToLabel = collection.mutable.Map[String, Int]()

  import MultiClassSVMActor._

  def getDimensions(labelfile: String, inputfile: String): Int = {
    var labels = Array[Double]()
    var images = Array[String]()
    var total = 0
    var allfiles =  Source.fromFile(labelfile).getLines

    for (f <- allfiles) {
      var inputwithlabel = f.split(" ")
      fileToLabel += (inputwithlabel(0) -> inputwithlabel(1).toDouble.toInt)
    }
    val (key, v) = fileToLabel.head
    var allval = Source.fromFile(inputfile+key).getLines

    return allval.length
  }

  def receive = {
    case fit(labelfile: String, inputpath: String, w_init: String, lr: Float, iters: Int, minibatch: Int, nw: Int, k: Int) => {
      t1 = System.nanoTime
      val ndims = getDimensions(labelfile, inputpath)
      noOfWorkers = nw
      if (w_init == "zeros") {
        weights = Nd4j.zeros(ndims+1, k)
      } else if (weights == "ones") {
        weights = Nd4j.ones(ndims+1, k)
      }

      print(fileToLabel.size, ndims)
      lrServer = context.actorOf(Props(new ParameterServer(ndims, k, fileToLabel, inputpath, batchSize, lr, noOfWorkers)))
      println("  Entering the parameter server ")
      lrServer ! DistributeData(iters)
      context.become(waitForIterations)
    }
  }


  def waitForIterations: Receive = {
    case UpdatedParameter(theta, accuracy, loss) => {
      weights = theta
      println("Accuracy and loss is ", accuracy*100, loss)
      val duration = (System.nanoTime - t1) / 1e9
      context.unbecome()
    }
  }

}
