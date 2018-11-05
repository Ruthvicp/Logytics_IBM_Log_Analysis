import org.apache.spark.{SparkConf, SparkContext}

object Stats {

  def main(args: Array[String]) {
    // directory for hadoop substitute
    System.setProperty("hadoop.home.dir", "C:\\winutils")

    val sparkConf = new SparkConf().setAppName("Stats").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)

    // retrieve data for files
    val inputf = sc.wholeTextFiles("data", 10)

    // get the log message data
    val wordData = inputf.flatMap(line => line._2.split("\\n")).map(line => line.split(",").drop(1).head.trim).cache()

    // map the data for a word count
    val wordCount = wordData.map(line => (line, 1))

    // map type and log message
    val wordDataTitle = inputf.flatMap(line => line._2.split("\\n")).map(line => (line.split(",").head.trim, line.split(",").drop(1).head.trim)).cache()

    // save count of types
    wordDataTitle.map(line => (line._1, 1)).reduceByKey(_+_).coalesce(1, shuffle = true).sortBy(line => line._2, false).saveAsTextFile("TitleCount")

    // save count of error messages
    wordDataTitle.map(line => if(line._1.toLowerCase.contains("err")){
      (line._2, 1)
    }
    else
    {
      ("", 0)
    }).reduceByKey(_+_).filter(line => line._1.compareTo("") != 0).coalesce(1, shuffle = true).sortBy(line => line._2, false).saveAsTextFile("ErrorCount")

    // save word count
    wordCount.reduceByKey(_+_).coalesce(1, shuffle = true).sortBy(line => line._2, false).saveAsTextFile("WordCount")

    // save distinct messages
    wordData.distinct().coalesce(1, shuffle = true).sortBy(line => line, false).saveAsTextFile("UniqueWords")
  }
}