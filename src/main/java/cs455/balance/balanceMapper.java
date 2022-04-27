package cs455.tp.balance;

import java.io.IOException;
import java.util.StringTokenizer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import java.util.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

/*
    Mapper: Read each line of CSV data
    Grab ID, Power Region, and length of lines
    Returns: <Power Region, length of lines lines>
*/ 

public class balanceMapper extends Mapper<Object, Text, Text, IntWritable> {

    private TreeMap<Integer, String> treeMap;
 
    @Override
    public void setup(Context context) throws IOException, InterruptedException{
        treeMap = new TreeMap<Integer, String>();
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
        String input = value.toString();
        String[] lineSplit = input.split(",");
        String Id = lineSplit[0];
        // String date = lineSplit[1];
        Date date = new Date(lineSplit[1]);
        Calendar calendar = Calendar.getInstance();
        calendar.setTime(date);
        // String mon = Integer.toString(calendar.get(Calendar.MONTH));
        String mon = String.format("%02d", calendar.get(Calendar.MONTH));
        String year = Integer.toString(calendar.get(Calendar.YEAR));
        String output_date = year + "-" + mon;
        

        int net_gen = Integer.parseInt(lineSplit[2]);
        int total_interchange = Integer.parseInt(lineSplit[3]);
        String powerRegion = lineSplit[4];


        context.write(new Text(powerRegion + output_date), new IntWritable(net_gen));
    }

}