import os
import numpy as np
import math
import random
import operator
class wordAligner(object):

    def __init__(self):
        
        inFile = open(os.getcwd() + "/data-a/episode1.zh-en")
        self.fileLines = inFile.readlines()
        self.lambdaDict = {}
        self.engLines = []
        self.chiLines = []
        self.wordAlignments = []
        self.tProbs = {}
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
            for engWord in english :
        
                if engWord in self.lambdaDict.keys():
                   for chiWord in chinese:
                        if chiWord not in self.lambdaDict[engWord].keys():
                            self.tProbs[engWord][chiWord] = 0
                            self.lambdaDict[engWord][chiWord] = 0
                     
                else:
                    self.lambdaDict[engWord] = {}
                    self.tProbs[engWord] = {}
                    for chiWord in chinese:
                        #if chiWord not in self.lambdaDict[engWord].keys():
                            self.lambdaDict[engWord][chiWord] = 0
                            self.tProbs[engWord][chiWord] = 0
        self.popTProbs()
    def popTProbs(self):
        for engWord, chiDict in self.lambdaDict.items():
            probSum = sum(np.exp(list(chiDict.values())))
            for chiWord, prob in chiDict.items():
                #print('t( ' + engWord + ' | ' + chiWord + ' ) = ' + str(math.exp(prob)) + ' / ' + str(probSum))
                self.tProbs[engWord][chiWord] = math.exp(prob) / probSum


    def computeProb(self,chiLine,engLine):
        m = len(chiLine)
        l = len(engLine)
        ll = l
        prob = 0
        forward = [0]*(m+1)     
        forward[0] = 1
        for j,chiWord in enumerate(chiLine):
            for i,engWord in enumerate(engLine):
                forward[j+1] += forward[j]*(1/ll)*self.tProbs[engWord][chiWord]
            forward[j+1] += forward[j] * (1/ll)*self.tProbs['NULL'][chiWord]
        #for j in range(1,m+1):
        #    for i in range(1,l):
        #        forward[j] += forward[j-1]*(1/ll)*self.tProbs[engLine[i]][chiLine[j-1]]
        #    forward[j] += forward[j-1]*(1/ll)*self.tProbs['NULL'][chiLine[j-1]]
        return (forward[m]/100 )
            


    def updateTProbs(self,engWord):
        probSum = sum(np.exp(list(self.lambdaDict[engWord].values())))
        for chiWord,lam in self.lambdaDict[engWord].items():
            self.tProbs[engWord][chiWord] = math.exp(lam) / probSum

    def gradDescent(self, T):
        shuffleLines = sorted(self.fileLines, key=lambda k: random.random())
        for t in range(1,T+1):
            n = 1/t
            LL = 0
            shuffleLines = sorted(shuffleLines, key=lambda k: random.random())
            for line in shuffleLines:
                
                line = line.rstrip()
                splitLine = line.split('\t')
                chinese = splitLine[0]
                english = splitLine[1]
                #english = "NULL " + english
                chinese = chinese.split()
                english = english.split()
                LL += math.log(self.computeProb(chinese, english))
                
                for i,chiWord in enumerate(chinese):
                    Z = self.tProbs['NULL'][chiWord]
                
                    for engWord in english:
                        Z += self.tProbs[engWord][chiWord]
                    for engWord in english:
                        
                        prob = self.tProbs[engWord][chiWord] / Z # prob that fj's partner is ei
                        self.lambdaDict[engWord][chiWord] += n*prob
                        self.updateTProbs(engWord)
                        for f in self.lambdaDict[engWord].keys():
                            self.lambdaDict[engWord][f] -= (n*prob)*self.tProbs[engWord][f] 
            
                        
            print('Pass ' + str(t) + ' through training data')
            print('Log probability = ' + str(LL))  
    def test(self,testData):
        self.wordAlignments.clear()
        for line in testData:
            line = line.rstrip()
            line = line.split('\t')
            english = line[1]
            english = 'NULL ' + english
            chinese = line[0]
            english = english.split()
            chinese = chinese.split()
            maxAlignments = []
            alignmentMatrix = []
    
            for i,chiWord in enumerate(chinese):
                tVector = []
                for j, engWord in enumerate(english):
                    tVector.append( self.tProbs[engWord][chiWord] )
                max_index, max_value = max(enumerate(tVector), key=operator.itemgetter(1))
                # NULL alignment will be -1
                maxAlignments.append(max_index-1)
            self.wordAlignments.append(maxAlignments)   
     
    def fiveLines(self):
        fiveEng = self.engLines[:5]
        fiveChin = self.chiLines[:5]
        for i,chin in enumerate(fiveChin):
            
            prob = math.log(self.computeProb(chin,fiveEng[i]))
            print('Log prob of line ' + str(i) + ' = ' + str(prob))
    def testFive(self):
        fiveLines = self.fileLines[:5]
        self.test(fiveLines)
        for lineAlignment in self.wordAlignments:
            for i , wordAlignment in enumerate(lineAlignment):             
                print(str(i) + '-' + str(wordAlignment) + ' ', end="")
            print("")
    def testAll(self):
        self.test(self.fileLines)
        outFile = open('myAlignments.align', 'w')
        for j,lineAlignment in enumerate(self.wordAlignments):
            #outFile.write(self.fileLines[j] + '\n')  
            for i, wordAlignment in enumerate(lineAlignment):
                # if not aligned w/ NULL
                if wordAlignment != -1:
                    outFile.write( str(i) + '-' + str(wordAlignment) + ' ')
            outFile.write('\n')
    def getTProb(self,chiWord,engWord):
        
        probSum = sum(np.exp(list(self.lambdaDict[engWord].values())))
        prob = math.exp(self.lambdaDict[engWord][chiWord])
        return (prob/probSum)
    def wordPairs(self):
        jediTVal      = self.tProbs[ 'jedi']['绝地' ]
        droidTVal     = self.tProbs[ 'droid']['机械人' ]
        forceTVal     = self.tProbs[ 'force'][ '原力']  
        midichlorTVal = self.tProbs[ 'midi-chlorians']['原虫']
        yousaTVal     = self.tProbs[ 'yousa']['你']
        print('t( 绝地 | jedi) = ' + str(jediTVal))
        print('t( 机械人 | droid ) = ' + str(droidTVal))
        print('t( 原力 | force ) = ' + str(forceTVal))
        print('t( 原虫 | midi-chlorians ) = ' + str(midichlorTVal))
        print('t( 你 | yousa ) = ' + str(yousaTVal))
if __name__ == "__main__":
    model = wordAligner()
    model.readLines()
    model.fiveLines()
    model.gradDescent(10)
    model.wordPairs()
    model.testFive()
    model.testAll()
