import weka.core.*;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Attribute;
import weka.classifiers.meta.FilteredClassifier;
import java.util.ArrayList;
import java.io.*;

public class Classify {

    ArrayList<String> text = new ArrayList<>();
    ArrayList<String> predResults = new ArrayList<>();
    ArrayList<String> actualValues;
    Instances instances;
    FilteredClassifier classifier;

    public void load(String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String line;
            while ((line = reader.readLine()) != null) {
                text.add(line.toLowerCase().trim());
            }
            System.out.println("===== Loaded text data: " + fileName + " =====\n");
            reader.close();
        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileName);
        }
    }


    public void loadModel(String fileName) {
        try {
            ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
            Object tmp = in.readObject();
            classifier = (FilteredClassifier) tmp;
            in.close();
            System.out.println("===== Loaded model: " + fileName + " =====\n");
        }
        catch (Exception e) {
            System.out.println("Problem found when reading: " + fileName);
        }
    }


    //Create Instances to be classified
    public void makeInstance() {

        // Create the attributes, class and text
        ArrayList<String> classVal = new ArrayList<>(8);
        for(int i = 1; i<= 8; i++){
            classVal.add(Integer.toString(i));
        }
        Attribute attribute1 = new Attribute("class",classVal);
        Attribute attribute2 = new Attribute("text",(ArrayList) null);

        // Create list of instances with one element
        ArrayList<Attribute> myAttributes = new ArrayList<Attribute>(2);
        myAttributes.add(attribute1);
        myAttributes.add(attribute2);
        instances = new Instances("Test relation",myAttributes,1);

        // Set class index
        instances.setClassIndex(0);

        // Create and add the instance
        for (String inst : text){
            Instance instance = new DenseInstance(2);
            instance.setValue(attribute2,inst);
            instances.add(instance);
        }

    }

    public ArrayList<String> classify() {
        try {
            System.out.println("===== Classified instances =====");
            for(int i=0; i<instances.numInstances();i++) {
                double pred = classifier.classifyInstance(instances.instance(i));
                String currentResult = instances.classAttribute().value((int) pred);
                //System.out.println("Class predicted: " + instances.classAttribute().value((int) pred) + " : " + instances.instance(i));
                predResults.add(currentResult);
            }
            return predResults;
        }
        catch (Exception e) {
            System.out.println("Problem found when classifying the text");
            return predResults;
        }
    }

    public double getAccuracy(ArrayList<String> predValues,String fileName) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            actualValues = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                actualValues.add(line.trim());
            }
            reader.close();
            double countCorrect = 0;

            for (int i = 0; i<actualValues.size();i++){
                if ((actualValues.get(i).toString().equals(predValues.get(i).toString()))){
                    countCorrect += 1;
                }
            }
            double accuracy = (countCorrect/actualValues.size()) * 100;
            return accuracy;

        }
        catch (IOException e) {
            System.out.println("Problem found when reading: " + fileName);
            e.printStackTrace();
            return 0;
        }

    }

    public static void main (String[] args) {

        Classify classifier;
        classifier = new Classify();
        classifier.load("data/test.txt");
        classifier.loadModel("SMOmodel");
        classifier.makeInstance();
        ArrayList<String> predict_classes = classifier.classify();
        System.out.println("\nAccuracy : " + classifier.getAccuracy(predict_classes,"data/actual_output.txt")+" %");

    }
}