import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SMO;
import weka.classifiers.functions.supportVector.RBFKernel;
import java.util.ArrayList;
import java.util.Random;
import weka.core.stemmers.SnowballStemmer;
import weka.classifiers.functions.supportVector.*;
import weka.classifiers.meta.FilteredClassifier;
import java.io.*;

public class Train_learn {

    Instances trainData;
    StringToWordVector filter;
    FilteredClassifier classifier;
    ArrayList<String> stopWords;

    public void loadDataset(String fileName) {

        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            trainData = new Instances(reader);
            reader.close();
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileName);
            e.printStackTrace();
        }
    }

    public void loadStopWords(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            stopWords = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                stopWords.add(line.toLowerCase().trim());
            }
            //System.out.println("STOPWORDS : " + stopWords);
            reader.close();
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileName);
            e.printStackTrace();
        }

    }

    public void evaluate() {
        try {

            trainData.setClassIndex(0);
            filter = new StringToWordVector();
            String[] options  = {"-L"}; //lower case
            filter.setOptions(options);
            filter.setAttributeIndices("last");
            //filter.setWordsToKeep(10000); //Num Features

            //Stemming
            SnowballStemmer stemmer = new SnowballStemmer();
            stemmer.setStemmer("english");
            filter.setStemmer(stemmer);

            //Filtering Stop Words
            filter.setStopwordsHandler(new StopTest(stopWords));

            //TFIDF
            filter.setTFTransform(true);
            filter.setIDFTransform(true);

            classifier = new FilteredClassifier();
            classifier.setFilter(filter);

            //SMO Parameters
            double cValue = 1;
            double gammaValue = -6;
            Kernel kernelValue = new RBFKernel();
            double c = Math.pow(2,cValue);
            double gamma = Math.pow(2, gammaValue);

            // Classification Model
            SMO myClassifier = new SMO();
            myClassifier.setKernel(kernelValue);
            myClassifier.setC(c);
            ((RBFKernel) kernelValue).setGamma(gamma);

            classifier.setClassifier(myClassifier);

            System.out.println("============EVALUATION============");

            Evaluation eval = new Evaluation(trainData);
            eval.crossValidateModel(classifier,trainData,10,new Random(4));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString());

        }
        catch (Exception e) {
            e.printStackTrace();
            System.out.println("Problem found when evaluating");
        }
    }


    public void learn() {
        try {
            trainData.setClassIndex(0);
            filter = new StringToWordVector();
            String[] options  = {"-L"}; //lower case
            filter.setOptions(options);
            filter.setAttributeIndices("last");
            //filter.setWordsToKeep(10000); //Num Features

            //Stemming
            SnowballStemmer stemmer = new SnowballStemmer();
            stemmer.setStemmer("english");
            filter.setStemmer(stemmer);

            //Filtering Stop Words
            filter.setStopwordsHandler(new StopTest(stopWords));

            //TFIDF
            filter.setTFTransform(true);
            filter.setIDFTransform(true);

            classifier = new FilteredClassifier();
            classifier.setFilter(filter);

            //SMO Parameters
            double cValue = 1;
            double gammaValue = -6;
            Kernel kernelValue = new RBFKernel();
            double c = Math.pow(2,cValue);
            double gamma = Math.pow(2, gammaValue);

            // Classification Model
            SMO myClassifier = new SMO();
            myClassifier.setKernel(kernelValue);
            myClassifier.setC(c);
            ((RBFKernel) kernelValue).setGamma(gamma);

            classifier.setClassifier(myClassifier);
            classifier.buildClassifier(trainData);

            //Write Classifier
            //FileWriter wr = new FileWriter("withTfidf");
            //wr.write(classifier.toString());
            //wr.close();

        }
        catch (Exception e) {
            System.out.println("Problem found when training");
        }
    }

    public void saveModel(String fileName) {
        try {
            ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
            out.writeObject(classifier);
            out.close();
            System.out.println("===== Saved model: " + fileName + " =====");
        }
        catch (IOException e) {
            System.out.println("Problem found when writing: " + fileName);
        }
    }

    public static void main (String[] args) {

        Train_learn learner;
        learner = new Train_learn();
        learner.loadDataset("data/train.arff");
        learner.loadStopWords("data/stopset.txt");
        learner.evaluate();
        learner.learn();
        learner.saveModel("SMOmodel");

    }
}