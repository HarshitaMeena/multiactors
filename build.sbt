name := "MultiClassActors"

version := "0.1"
scalaVersion := "2.11.8"
resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

val nd4jVersion = "0.8.0"
lazy val akkaVersion = "2.5.21"

libraryDependencies += "org.deeplearning4j" % "deeplearning4j-core" % nd4jVersion

libraryDependencies += "org.nd4j" % "nd4j-native-platform" % nd4jVersion

libraryDependencies += "org.nd4j" %% "nd4s" % nd4jVersion
libraryDependencies += "org.datavec" %  "datavec-api" % nd4jVersion

libraryDependencies ++= Seq(
  "com.typesafe.akka" %% "akka-actor" % akkaVersion,
  "com.typesafe.akka" %% "akka-testkit" % akkaVersion,
  "org.scalatest" %% "scalatest" % "3.0.5" % "test"
)
