# -*- coding: utf-8 -*-
import glob, sys, os, re, string, time, random, dateparser
from dateparser.search import search_dates
#!/usr/sfw/bin/python

"""
    dateparser_export_to_BIEO_format, a script to call dateparser on
    sentences in French and export the result to the BIEO format
    (Beginning / Inside / End / Outside).
    
    Copyright (C) 2021 Philippe Gambette

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# get the current folder   
folder = os.path.abspath(os.path.dirname(sys.argv[0]))

# for all csv files in the corpus folder
for file in glob.glob(os.path.join(os.path.join(folder, "corpus"), "*.csv")):
   # open the input file, which should be a csv file containing:
   # * sentences starting with #
   # * followed by the list of tokens number, each followed, 
   # on the same line, by a tabulation and by the token itself
   input = open(file, "r", encoding="utf-8")
   
   # save the result in this output file
   output = open(file+".results.tsv", "w", encoding="utf-8")
   
   beginLabels = []
   linesToWrite = []
   savedLines = []
   for line in input:
      # Look for lines with a tabulation separator
      res = re.search("^(.*)\t(.*)$", line)
      if res:
         # part1 should contain either a # followed by a sentence or a token id
         part1 = res.group(1)
         # part2 should be empty or contain a token
         part2 = res.group(2)
         # Rem
         res2 = re.search("^([^\r\n]*[\r\n]*).*$", part2)
         if res2:
            part2 = res2.group(1)
         if len(part1)>0 and len(part2)==0:
            # Found a sentence: look for dates inside
            print("Parsed sentence: " + part1)
            # Looking for dates with the dateparser search_dates function
            foundDates = search_dates(part1, languages=['fr'])
            if str(type(foundDates)) != "<class 'NoneType'>" and len(foundDates)>0:
               if len(foundDates)>1:
                  print("   → dateparser found "+str(len(foundDates))+" dates: "+str(foundDates))
               else: 
                  print("   → dateparser found a date: "+str(foundDates))
               # start matching tokens with the date found by dateparser
               # the matched tokens and their label will be added to the list linesToWrite
               beginLabels = []
               linesToWrite = []
               # beginLabels will store the beginnings of dates found in the text when we progressively read tokens
               for d in foundDates:
                  beginLabels.append("")
                  linesToWrite.append([])
            else:
               # no date found in the sentence
               beginLabels = []
               linesToWrite = []
            output.writelines(line)
         if len(part1)>0 and len(part2)>0:
            # Assume that the new token is not part of a date
            savedLines.append(part1+"\t"+part2+"\tO\n")
            for i in range(0,len(beginLabels)):
               foundBeginning = False
               if beginLabels[i] + part2 == foundDates[i][0][0:len(beginLabels[i]+part2)]:
                  # The token, not preceded by a whitespace, is part of a date
                  if beginLabels[i] == "":
                     #The token is the start of a date
                     linesToWrite[i].append(part1+"\t"+part2+"\tBT\n")
                     if part2 == foundDates[i][0]:
                        #It is also an end of a date: write it directly with its label BT
                        output.writelines(linesToWrite[i][0])
                        linesToWrite[i] = []
                        # Reinitialize savedLines as the new token was actually part of a date
                        savedLines = []
                     beginLabels[i] += part2
                  else:
                     # The token belongs to a date
                     beginLabels[i] += part2
                     #print("Matched a following!")
                     if beginLabels[i] == foundDates[i][0]:
                        # If the token is the end of a date, add it with label ET to the lines to write,
                        # then write all the lines in linesToWrite to the output file 
                        linesToWrite[i].append(part1+"\t"+part2+"\tET\n")
                        savedLines = []
                        for l in linesToWrite[i]:
                           output.writelines(l)
                        beginLabels[i] = ""
                        linesToWrite[i] = []
                        # Reinitialize savedLines as the new token was actually part of a date
                        savedLines = []
                     else:
                        # If the token is inside a date, add it with label IT to the lines to write
                        linesToWrite[i].append(part1+"\t"+part2+"\tIT\n")
                  foundBeginning = True
               if beginLabels[i] + " " + part2 == foundDates[i][0][0:len(beginLabels[i]+" "+part2)]:
                     # The token, preceded by a whitespace, is part of a date
                     beginLabels[i] += " " + part2
                     # The token belongs to a date
                     if beginLabels[i] == foundDates[i][0]:
                        # If the token is the end of a date, add it with label ET to the lines to write,
                        # then write all the lines in linesToWrite to the output file 
                        linesToWrite[i].append(part1+"\t"+part2+"\tET\n")
                        savedLines = []
                        for l in linesToWrite[i]:
                           output.writelines(l)
                        beginLabels[i] = ""
                        linesToWrite[i] = []
                        # Reinitialize savedLines as the new token was actually part of a date
                        savedLines = []
                     else:
                        # If the token is inside a date, add it with label IT to the lines to write
                        linesToWrite[i].append(part1+"\t"+part2+"\tIT\n")
                     foundBeginning = True
               # The current token does not correspond to this beginning of a date
               if not(foundBeginning):
                  beginLabels[i] = ""
                  linesToWrite[i] = []
            # Check if we are still matching a date,  
            # that is if beginLabels still contain the beginning of a date 
            # matching a sequence made of the latest seen tokens
            stillMatchingADate = False
            for i in range(0,len(beginLabels)):
               if len(beginLabels[i])>0:
                  stillMatchingADate = True
            # If we are not currently matching a date, write the latest token to the output file with label O
            if not(stillMatchingADate):
               for l in savedLines:
                  output.writelines(l)
               savedLines = []
         if len(part1)==0 and len(part2)==0:
            # The line is empty, we reproduce it directly
            output.writelines(line)

   input.close()
   output.close()