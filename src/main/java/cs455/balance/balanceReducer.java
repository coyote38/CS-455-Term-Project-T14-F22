package cs455.tp.balance;

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
    Receives: <Power Region, #Substation lines>
    Sums up length of lines for each key region
    Returns sums for power region
*/

public class balanceReducer extends Reducer<Text, IntWritable, Text, DoubleWritable> {
    @Override
    public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
        // sum of all aqi scores
        double sum = 0;
        // keeps track of number of items for key
        double num = 0;
        double avg = 0;
        for(IntWritable val : values){
            num += 1;
            sum += val.get();
        }
        // System.out.println("sum" + sum);
        // System.out.println("num" + num);

        avg = sum / num;
        // System.out.println("avg" + avg);
        context.write(new Text(key.toString()), new DoubleWritable(avg));
    }
}