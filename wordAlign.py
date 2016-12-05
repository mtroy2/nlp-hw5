import os
import numpy as np
import math


class wordAligner(object):

    def __init__(self):
        
        inFile = open(os.getcwd() + "/data-a/episode1.zh-en")
        self.fileLines = inFile.readlines()
        self.lambdaDict = {}
        self.engLines = []
        self.chiLines = []
    def readLines(self):

       for line in self.fileLines:
            line = line.strip('\n')
            splitLine = line.split('\t')
            chinese = splitLine[0]
            english = splitLine[1]
            english = "NULL " + english
            chinese = chinese.split()
            english = english.split()
            self.chiLines.append(chinese)
            self.engLines.append(english)
            for character in chinese:
                if character in self.lambdaDict.keys():
                   for word in english:
                        if word not in self.lambdaDict[character].keys():
                            self.lambdaDict[character][word] = 0
                     
                else:
                    self.lambdaDict[character] = {}
                    for word in english:
                        if word not in self.lambdaDict[character].keys():
                            self.lambdaDict[character][word] = 0


    def computeProb(self, chiLine, engLine):
        
        m = len(chiLine)
        l = len(engLine)
        prob = 0
        for j, chiWord in enumerate(chiLine):
                
            engSum = 0
            for engWord in engLine:
                engSum += self.getTProb( chiWord, engWord )

            nullParam = self.getTProb(chiWord, 'NULL')
            prob += math.log((1/(l+1)) * (nullParam + engSum))

        if prob == 0 :
            return 1
        else:
            return prob

    def getTProb(self,chiSym,engSym):
        num =  math.exp( self.lambdaDict[chiSym][engSym] )
        
        probSum = 0
        for chiKey, engDict in self.lambdaDict.items()  :
            if engSym in engDict.keys():  
                probSum += math.exp(engDict[engSym])
        
        if probSum == 0:
            return 1
        else:
            return (num / probSum)
    def fiveLines(self):
        fiveEng = self.engLines[:5]
        fiveChin = self.chiLines[:5]
        for i,chin in enumerate(fiveChin):
            
            prob = self.computeProb(chin,fiveEng[i])
            print('Log prob of line ' + str(i) + ' = ' + str(prob))
            
if __name__ == "__main__":
    model = wordAligner()
    model.readLines()
    model.fiveLines()
