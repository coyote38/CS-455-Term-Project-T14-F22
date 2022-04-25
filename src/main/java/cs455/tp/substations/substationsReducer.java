package cs455.aqi.substations;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.Text;
import java.util.TreeMap;
import java.util.Map;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/*
    Receives: <Month, AQI Scores>
    Sums up AQI scores for each key (month)
    Finds mean by dividing sum by number of entries for that month
    Returns means for month as <Month, mean> 
*/

public class substationsReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
        // sum of all aqi scores
        long sum = 0;
        // keeps track of number of items for key
        int num = 0;
        long avg = 0;
        for(IntWritable val : values){
            num += 1;
            sum += val.get();
        }
        avg = sum / num;
        context.write(new Text(key.toString()), new DoubleWritable(avg));
    }
}