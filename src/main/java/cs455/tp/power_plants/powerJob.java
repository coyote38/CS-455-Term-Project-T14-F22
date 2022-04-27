package cs455.tp.power_plants;

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

public class powerJob{
    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "powerJob");
        // current class
        job.setJarByClass(powerJob.class);
        // Mapper
        job.setMapperClass(powerMapper.class);
        // Combiner
        //job.setCombinerClass(IntSumReducer.class);
        // Reducer
        job.setReducerClass(powerReducer.class);
        // Outputs from the Mapper
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        // Outputs from the Reducer
        job.setOutputKeyClass(Text.class);
        job.setOutputKeyClass(DoubleWritable.class);
        // set number of tasks
        job.setNumReduceTasks(1); 
        System.out.println(args[0]);
        System.out.println(args[1]);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }      
}