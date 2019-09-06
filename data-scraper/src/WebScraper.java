import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.select.Elements;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class WebScraper {

	public static void main(String[] args)throws Exception {
		// TODO Auto-generated method stub
		Document document;
		try {

			document = Jsoup.connect("https://www.baseball-reference.com/boxes/CIN/CIN201903280.shtml").get();
			String title = document.title();
			System.out.println(" Title: " +title);
		}
		catch (IOException e) {
			e.printStackTrace();	
		}
	}

}
