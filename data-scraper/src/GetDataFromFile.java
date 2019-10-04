import java.io.*;
import java.util.*;
public class GetDataFromFile {

        public static void main(String[] args)throws Exception {
        //File file = new File("C:\\gl2010_18\\GL2010.txt");
        //BufferedReader br = new BufferedReader(new FileReader(file));
        //File csv = new File("NYY2010.csv");
        String[] teamAbbr = new String[] {"CHN","PHI","PIT", "CIN", "SLN", "BOS", "CHA",
                                          "CLE", "DET", "NYA", "BAL", "LAN", "SFN", "MIN",
                                          "HOU", "NYN", "ATL", "OAK", "KCA", "SDN", "TEX",
                                          "TOR", "SEA", "FLO", "COL", "ANA", "TBA", "ARI", 
                                          "MIL", "WAS", "MIA"};
        for(int year=2010;year<2019;year++)
        {
            File file = new File("C:\\gl2010_18\\GL"+year+".txt");
            for(int teamNum=0; teamNum<=29;teamNum++)
            {
                BufferedReader br = new BufferedReader(new FileReader(file));
                String csvFileName = teamAbbr[teamNum] + year + ".csv";
                File csv = new File(csvFileName);
                FileWriter csvWriter = new FileWriter(csvFileName);

                csvWriter.append("Date");
                csvWriter.append(",");
                csvWriter.append("Visting Team");
                csvWriter.append(",");
                csvWriter.append("League");
                csvWriter.append(",");
                csvWriter.append("Visiting Team Game Number");
                csvWriter.append(",");
                csvWriter.append("Home Team");
                csvWriter.append(",");
                csvWriter.append("League");
                csvWriter.append(",");
                csvWriter.append("Home Team Game Number");
                csvWriter.append(",");
                csvWriter.append("Visiting Team Score");
                csvWriter.append(",");
                csvWriter.append("Home Team Score");
                csvWriter.append(",");
                csvWriter.append("Length of Game in Outs");
                csvWriter.append(",");
                csvWriter.append("Park ID");
                csvWriter.append(",");
                csvWriter.append("Attendance");
                csvWriter.append(",");
                csvWriter.append("Visiting Team Line Numbers");
                csvWriter.append(",");
                csvWriter.append("Home Team Line Numbers");
                csvWriter.append(",");
                csvWriter.append("Visting Team At-Bats");
                csvWriter.append(",");
                csvWriter.append("Visting Team Hits");
                csvWriter.append(",");
                csvWriter.append("Visting Team Doubles");
                csvWriter.append(",");
                csvWriter.append("Visting Team Triples");
                csvWriter.append(",");
                csvWriter.append("Visting Team Home-Runs");
                csvWriter.append(",");
                csvWriter.append("Visting Team RBI");
                csvWriter.append(",");
                csvWriter.append("Visiting Team Sac Hits");
                csvWriter.append(",");
                csvWriter.append("Visting Team Sac Flys");
                csvWriter.append(",");
                csvWriter.append("Visting Team HBP");
                csvWriter.append(",");
                csvWriter.append("Visting Team Walks");
                csvWriter.append(",");
                csvWriter.append("Visting Team Int Walks");
                csvWriter.append(",");
                csvWriter.append("Visting Team Strikeouts");
                csvWriter.append(",");
                csvWriter.append("Visting Team Stolen Bases");
                csvWriter.append(",");
                csvWriter.append("Visting Team Caught Stealing");
                csvWriter.append(",");
                csvWriter.append("Visting Team G Double Play");
                csvWriter.append(",");
                csvWriter.append("Visting Team Awarded First on Interference");
                csvWriter.append(",");
                csvWriter.append("Visiting Team LOB");
                csvWriter.append(",");
                csvWriter.append("Visting Team Pitchers Used");
                csvWriter.append(",");
                csvWriter.append("Visting Team Ind ER");
                csvWriter.append(",");
                csvWriter.append("Visting Team Team ER");
                csvWriter.append(",");
                csvWriter.append("Visting Team Wild Pitches");
                csvWriter.append(",");
                csvWriter.append("Visting Team Balks");
                csvWriter.append(",");
                csvWriter.append("Visting Team Put-outs");
                csvWriter.append(",");
                csvWriter.append("Visting Team Assists");
                csvWriter.append(",");
                csvWriter.append("Visting Team Errors");
                csvWriter.append(",");
                csvWriter.append("Visting Team Passed Balls");
                csvWriter.append(",");
                csvWriter.append("Visting Team Double Plays");
                csvWriter.append(",");
                csvWriter.append("Visting Team Triple Plays");
                csvWriter.append(",");

                csvWriter.append("Home Team At-Bats");
                csvWriter.append(",");
                csvWriter.append("Home Team Hits");
                csvWriter.append(",");
                csvWriter.append("Home Team Doubles");
                csvWriter.append(",");
                csvWriter.append("Home Team Triples");
                csvWriter.append(",");
                csvWriter.append("Home Team Home-Runs");
                csvWriter.append(",");
                csvWriter.append("Home Team RBI");
                csvWriter.append(",");
                csvWriter.append("Home Team Sac Hits");
                csvWriter.append(",");
                csvWriter.append("Home Team Sac Flys");
                csvWriter.append(",");
                csvWriter.append("Home Team HBP");
                csvWriter.append(",");
                csvWriter.append("Home Team Walks");
                csvWriter.append(",");
                csvWriter.append("Home Team Int Walks");
                csvWriter.append(",");
                csvWriter.append("Home Team Strikeouts");
                csvWriter.append(",");
                csvWriter.append("Home Team Stolen Bases");
                csvWriter.append(",");
                csvWriter.append("Home Team Caught Stealing");
                csvWriter.append(",");
                csvWriter.append("Home Team G Double Play");
                csvWriter.append(",");
                csvWriter.append("Home Team Awarded First on Interference");
                csvWriter.append(",");
                csvWriter.append("Home Team LOB");
                csvWriter.append(",");
                csvWriter.append("Home Team Pitchers Used");
                csvWriter.append(",");
                csvWriter.append("Home Team Ind ER");
                csvWriter.append(",");
                csvWriter.append("Home Team Team ER");
                csvWriter.append(",");
                csvWriter.append("Home Team Wild Pitches");
                csvWriter.append(",");
                csvWriter.append("Home Team Balks");
                csvWriter.append(",");
                csvWriter.append("Home Team Put-outs");
                csvWriter.append(",");
                csvWriter.append("Home Team Assists");
                csvWriter.append(",");
                csvWriter.append("Home Team Errors");
                csvWriter.append(",");
                csvWriter.append("Home Team Passed Balls");
                csvWriter.append(",");
                csvWriter.append("Home Team Double Plays");
                csvWriter.append(",");
                csvWriter.append("Home Team Triple Plays");
                csvWriter.append(",");
                csvWriter.append("Winning Pitcher ID");
                csvWriter.append(",");
                //csvWriter.append("Winning Pitcher Name");
                //csvWriter.append(",");
                csvWriter.append("Losing Pitcher ID");
                csvWriter.append(",");
                //csvWriter.append("Losing Pitcher Name");
                //csvWriter.append(",");
                csvWriter.append("Saving Pitcher ID");
                csvWriter.append(",");
                //csvWriter.append("Saving Pitcher Name");
                //csvWriter.append(",");
                csvWriter.append("Visiting Starter Pitcher ID");
                csvWriter.append(",");
                //csvWriter.append("Visiting Starter Pitcher Name");
                //csvWriter.append(",");
                csvWriter.append("Home Starter Pitcher ID");
                csvWriter.append(",");
                //csvWriter.append("Home Starter Pitcher Name");
                csvWriter.append("\n");
                String st;
                int i=0;
                while ((st = br.readLine()) != null)
                {
                    //System.out.println(st);
                    String[] data = st.split(",");
                    System.out.println(data[3]);
                    System.out.println(teamAbbr[teamNum]);
                    if(year>2011 && teamAbbr[teamNum].equals("FLO"))
                    {
                        if(data[3].equals("\"MIA\"") || data[6].equals("\"MIA\""))
                        {
                            i++;
                            System.out.println(i);
                            csvWriter.append(data[0]);
                            csvWriter.append(",");
                            csvWriter.append(data[3]);
                            csvWriter.append(",");
                            csvWriter.append(data[4]);
                            csvWriter.append(",");
                            csvWriter.append(data[5]);
                            csvWriter.append(",");
                            csvWriter.append(data[6]);
                            csvWriter.append(",");
                            csvWriter.append(data[7]);
                            csvWriter.append(",");
                            csvWriter.append(data[8]);
                            csvWriter.append(",");
                            csvWriter.append(data[9]);
                            csvWriter.append(",");
                            csvWriter.append(data[10]);
                            csvWriter.append(",");
                            csvWriter.append(data[11]);
                            csvWriter.append(",");
                            csvWriter.append(data[16]);
                            csvWriter.append(",");
                            csvWriter.append(data[17]);
                            csvWriter.append(",");

                            for(int j=19;j<=76;j++)
                            {
                                csvWriter.append(data[j]);
                                csvWriter.append(",");
                            }
                            for(int k=93;k<=98;k=k+2)
                            {
                                csvWriter.append(data[k]);
                                csvWriter.append(",");
                            }
                            for(int l=101;l<=104;l=l+2)
                            {
                                csvWriter.append(data[l]);
                                csvWriter.append(",");
                            }
                            csvWriter.append("\n");
                        }
                    }
                    else if(data[3].equals("\""+teamAbbr[teamNum]+"\"") || data[6].equals("\""+teamAbbr[teamNum]+"\""))
                    {
                        i++;
                        System.out.println(i);
                        csvWriter.append(data[0]);
                        csvWriter.append(",");
                        csvWriter.append(data[3]);
                        csvWriter.append(",");
                        csvWriter.append(data[4]);
                        csvWriter.append(",");
                        csvWriter.append(data[5]);
                        csvWriter.append(",");
                        csvWriter.append(data[6]);
                        csvWriter.append(",");
                        csvWriter.append(data[7]);
                        csvWriter.append(",");
                        csvWriter.append(data[8]);
                        csvWriter.append(",");
                        csvWriter.append(data[9]);
                        csvWriter.append(",");
                        csvWriter.append(data[10]);
                        csvWriter.append(",");
                        csvWriter.append(data[11]);
                        csvWriter.append(",");
                        csvWriter.append(data[16]);
                        csvWriter.append(",");
                        csvWriter.append(data[17]);
                        csvWriter.append(",");
                        for(int j=19;j<=76;j++)
                        {
                            csvWriter.append(data[j]);
                            csvWriter.append(",");
                        }
                        for(int k=93;k<=98;k=k+2)
                        {
                            csvWriter.append(data[k]);
                            csvWriter.append(",");
                        }
                        for(int l=101;l<=104;l=l+2)
                        {
                            csvWriter.append(data[l]);
                            csvWriter.append(",");
                        }
                        csvWriter.append("\n");
                    }   
                }
            csvWriter.flush();
            csvWriter.close();
            br.close();

            }
        //FileWriter csvWriter = new FileWriter("NYY2010.csv");
        }
    }
}
