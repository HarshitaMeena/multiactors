import MultiClassSVMActor.fit
import akka.actor.{ActorSystem, Props}

object main extends App {

  //var (x,y) = readFromAFile("cod-rna.txt")
  ///*
  val labelfile = "src/main/resources/mnist-labels-5k.txt"
  val inputpath = "src/main/resources/mnist/"
  val system = ActorSystem("Main")
  val decay = 0.001f
  val now = 3
  val lr = 0.00000005f
  val iterations = 2000
  val labels = 10
  //*/

  //var (x,y) = readFromAFile("cod-rna.txt")
  /*
  val labelfile = "src/main/resources/satimage-labels.txt"
  val inputpath = "src/main/resources/satimage/"
  val system = ActorSystem("Main")
  val decay = 0.001f
  val now = 3
  val lr = 0.005f
  val iterations = 10000
  val labels = 6
  */

  val ac = system.actorOf(Props[MultiClassSVMActor])
  ac ! fit(labelfile, inputpath, "zeros", lr, iterations, 0, now, labels)

}

