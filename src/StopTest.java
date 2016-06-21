import weka.core.stopwords.StopwordsHandler;
import java.io.Serializable;
import java.util.ArrayList;

public class StopTest implements StopwordsHandler,Serializable {

    ArrayList<String> stopWords;

    public StopTest(ArrayList<String> s){
        stopWords = s;
    }
    @Override
    public boolean isStopword(String s) {
        return stopWords.contains(s);
    }
}
