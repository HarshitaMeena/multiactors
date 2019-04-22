import akka.actor.{Actor, ActorRef, Props}
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j


object ParameterServer {
  case class UpdatedParameter (theta: INDArray, accuracy: Float, loss: Double)
  case class NextBatch(id: Int, thetas: INDArray)
  case class Accuracy(correct: Int, loss: Double)
}

class ParameterServer(ndims: Int, k: Int, labeltofile: collection.mutable.Map[String, Int], inputpath: String, batchsize: Int, lr: Float, now: Int)
  extends Actor {

  import MultiClassSVMActor._
  import ParameterServer._
  import Workers._

  var noOfWorkers = now
  val partitionActors: Array[ActorRef] = new Array[ActorRef](noOfWorkers)
  for (i <- 0 to noOfWorkers-1) {
    partitionActors(i) = context.actorOf(Props(new Workers(inputpath)))
  }
  var totalbatches = math.ceil(labeltofile.size / batchsize.toFloat).toInt
  var batchesProcessed = 0
  var batchesSent = 0
  var parent: ActorRef = _
  var iterations: Int = _
  var dataload = 0
  var correctClassified = 0
  var totalloss = 0.0
  var t1 = 0L

  noOfWorkers = math.min(totalbatches, noOfWorkers)

  private var thetas = Nd4j.zeros(k,ndims+1)


  def shuffleData(x: INDArray, y: INDArray): INDArray = {
    val stackedData = Nd4j.hstack(x,y)
    //Nd4j.shuffle(stackedData, 1)
    stackedData
  }

  def startWorkers(): Unit = {
    //combinedData = shuffleData(input, labels)

    var i = 0
    val batchsize = (labeltofile.size / noOfWorkers).toInt


    var allinputfiles = labeltofile.keySet.toArray


    while (i < noOfWorkers) {

      var lastIndex = (i+1)*batchsize
      if (i == noOfWorkers - 1) {
        lastIndex = allinputfiles.length
      }
      partitionActors(i) ! LoadData(allinputfiles.slice(i*batchsize, lastIndex), labeltofile)
      partitionActors(i) ! NextBatch(i, thetas)
      i += 1
      batchesSent += 1
    }
    println("Entered worker mode")

  }


  def receive = {
    case DistributeData(iter) => {
      iterations = iter
      //log.info(iter.toString + " " + combinedData.length().toString)
      println(iterations)

      startWorkers()
      parent = sender()
      t1 = System.nanoTime
      context.become(waitForWorkers)
    }
  }


  def waitForWorkers: Receive = {
    case Gradient(id, grad) => {
      thetas = thetas.add(grad.mul(-lr))
      iterations -= 1
      if (iterations < 0 ) {
        context.become(getAccuracy)
        for (worker <- partitionActors) {
          worker ! GetWorkerAccuracy(thetas)
        }
      } else {
        sender() ! NextBatch(id, thetas)
      }
    }
  }

  def getAccuracy: Receive = {
    case Accuracy(correct, loss) => {
      noOfWorkers -= 1
      correctClassified += correct
      totalloss += loss
      if (noOfWorkers == 0) {
        //println(correctClassified)
        parent ! UpdatedParameter(thetas, correctClassified.toFloat/labeltofile.size, totalloss)
        val duration = (System.nanoTime - t1) / 1e9
        println("TOTAL TIME FOR PROCESSING is ", duration)
      }
    }
  }
}
