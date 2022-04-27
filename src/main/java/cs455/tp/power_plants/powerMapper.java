package cs455.tp.power_plants;

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
    Grab ID, Power Region, and Power Source
    Returns: <Power Region + Power Source, 1>
*/ 

public class powerMapper extends Mapper<Object, Text, Text, IntWritable> {

    private TreeMap<Integer, String> treeMap;
 
    @Override
    public void setup(Context context) throws IOException, InterruptedException{
        treeMap = new TreeMap<Integer, String>();
    }

    @Override
    public void map(Object key, Text value, Context context) throws IOException, InterruptedException{
        System.out.println(value);
        String input = value.toString();
        System.out.println(input);
        String[] lineSplit = input.split(",");
        System.out.println("0" + lineSplit[0]);
        System.out.println("1" + lineSplit[1]);
        System.out.println("1" + lineSplit[2]);
        // int plantId = Integer.parseInt(lineSplit[0]);
        String powerRegion = lineSplit[1];
        String powerSource = lineSplit[2];

        context.write(new Text(powerRegion + powerSource), new IntWritable(1));
    }

}