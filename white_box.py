
import sys
from jacobian_matrix import getJacobianMatrix
from classifier_F import getClassifier
from generate_bugs import generateBugs
from operator import itemgetter
from collections import OrderedDict
import random
import pprint
from gensim.models import Word2Vec








class WhiteBox():

    def __init__(self, X, y, F, epsilon): 
        self.X = X
        self.y = y
        self.F = F
        self.epsilon = epsilon
        self.textbuggerWhiteBox(X,y,F,epsilon)

    def textbuggerWhiteBox(self, X, y, F, epsilon):
        print("tb whitebox")
        Xlist = X.split(" ")
        Xdict = {}


        # Lines 2-4: Compute importance C
        for word in Xlist:
            Xdict[word] = self.getImportanceC(F, X, word)



        # Line 5: Sort words in terms of importance C
        sortedXDict = OrderedDict(sorted(Xdict.items(), key = itemgetter(1), reverse = True))
        # print(sortedXDict)
        sortedXList = list(sortedXDict)
        print(sortedXList)

        # Lines 6-14: SelectBug and Iterate
        for word in sortedXList:
            bug = self.selectBug(word, X, y, F)









    def selectBug(self, word, X, y, F):
        bugs = generateBugs(word)
        # print('59')
        # print(bugs)
        for bug_type, bk in bugs.items():
            candidate = self.getCandidate(word, bk, X)
            print(candidate)



        return bugs

    def getImportanceC(self, F, X, word):
        Xlist = X.split(" ")
        return random.randint(0, len(Xlist)-1)


    def getCandidate(self, word, bug, X):
        Xlist = X.split(" ")
        for i in range(0, len(Xlist)):
            # print(Xlist[i])
            if Xlist[i] == word:
                Xlist[i] = bug
                # print(Xlist)
                return ' '.join(Xlist)




if __name__ == "__main__":
    # if len(sys.argv)<2:
    #     print("python process_data.py [text]")
    #     # print(len(sys.argv))
    #     sys.exit(1)




    text = "I dislike all vegetables"
    truth = "positive"
    F = None
    epsilon = 0.5

    whiteBox = WhiteBox(text, truth, F, epsilon)



