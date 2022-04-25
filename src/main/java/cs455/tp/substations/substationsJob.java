package cs455.aqi.substations;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.input.TextInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;
import java.io.IOException;

public class substationsJob{
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "substations");
        // current class
        job.setJarByClass(substationsJob.class);
        // Mapper
        job.setMapperClass(substationsMapper.class);
        // Combiner
        //job.setCombinerClass(IntSumReducer.class);
        // Reducer
        job.setReducerClass(substationsReducer.class);
        // Outputs from the Mapper
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        // Outputs from the Reducer
        job.setOutputKeyClass(Text.class);
        job.setOutputKeyClass(DoubleWritable.class);
        // set number of tasks
        job.setNumReduceTasks(1); 
        System.out.println(args[0]);
        FileInputFormat.addInputPath(job, new Path(args[1]));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }      
}