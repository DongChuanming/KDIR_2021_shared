#This script is designed for transferring the labels assigned on tokens in a sentence tokenized in one way, 
#to labels assigned on tokens in the same sentence tokenized in another way.
# for example, we want the annotation on camemBERT tokens in this sentence : 
#_La(BO) _SARL(IO) _L(EO) U(EO) CI(EO) ANI(EO) _a(BN) _été(IN) _mise(IN) _en(IN) _liquidation(EN) _le(BT) _31(ET) /08/(ET) 2006(ET) .
#to annnotation on tree-tagger tokenization :
#La(BO) SARL(IO) LUCIANI(EO) a(BN) été(IN) mise(IN) en(IN) liquidation(EN) le(BT) 31/08/2006(ET) .(O)
#this means we already have the sentence tokenized in two ways, and we also know the annotation on one of the tokenization
#The idea is to glue the letters together to form an index string,and correspond each letter to a label, 
#and this way when the letters reorganized, the labels will be reorganized with them.

#our text is annotated in tsv format, with each line representing a word and its label. The words compose sentences, and each sentence are 
#seperated from each other with one or more empty line. To illustrate, here are two extracted sentences:

#-----------------------------------------------------------------------------------------------------
# #- Les derniers déchets ont été éliminés début 2004.
# 1	-	        O		O
# 2	Les	        BO		BU
# 3	derniers	IO		IU
# 4	déchets	    EO		EU
# 5	ont	        BN		O
# 6	été	        IN		O
# 7	éliminés	EN		O
# 8	début	    BT		O
# 9	2004	    ET		O
# 10.	        O		O
					
					
#1	-	        O		O
#2	Les	        BO		O
#3	bâtiments	EO		O
#4	ont	        BN		O
#5	été	        IN		O
#6	démolis	    EN		O
#7	.	        O		O

#-----------------------------------------------------------------------------------------------------

#So the first step is to read the file and extract words of sentences in a list
#for that we designed a fonction read_conllu_phrases(t,ind), with t for text file name, ind for the colonne


def read_conllu(t,ind):
    import re
    #we would like to create a list that contains multiple tuples. Each tuple represents a aentence, containing a list of word form, and a list of label
    #there for the first step is to seperate the tsv file by sentences. In our file, the sentences are seperated by empty lines,
    #but given that the file may be edited in google sheet, The empty line in the file may contains tab symbol, so we include tabs in our seperator "sep" 
    sep=re.compile("\n\t*\n\t*\n*\t*")
    #read file
    with open(t, "r", encoding="utf-8") as p:
        t=p.read()
        #seperate into a list of sentences
        phs=sep.split(t)
        #we creat a list 'corpus' to contain the reorganised information of the sentences
        corpus=list()
        #begin to prossess each sentence
        for s in phs:
            #we create a list 'phrase' to contain the information of the words
            phrase=list()
            for i in range(len(ind)):
                phrase.append(list())
            #we seperate each line to make a list of word features
            ws=s.split('\n')
            for w in ws:
                #we don't want to process the line that begins with "#" since it is not a annotation of word
                if len(w.strip("\t"))>0:
                    if w[0]!="#":
                        lt=w.split("\t")
                        #we extract the features in columns that we indicated in the "ind"
                        for i in range(len(ind)):
                            index=ind[i]
                            phrase[i].append(lt[index])
	    #then the lists that contain word features get reorganised into a tuple, which is then inserted into the list "corpus" created before.
            if len(phrase[0])>0:
                corpus.append(tuple(phrase))
	#the output is the list corpus
        return corpus

# In[18]:
# we read the annotated corpus with camembert tokenization
corpus=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Results/token_annot_compare.tsv",[1,2,3])

# In[11]:
# now the words will be glued together, and each letter will inherate the word's label
correspondance=list()
for a,c,d in corpus:
    concat=""
    indexe1=list()
    indexe2=list()
    for i in range(len(a)):
        w=a[i]
        t=c[i]
        p=d[i]
        lw=len(w)
        lt=[t]*lw
        lp=[p]*lw
        concat+=w
        indexe1+=lt
        indexe2+=lp
    # we insert the glued words and the lists that contain their letters' labels into a list called "correspondance"      
    correspondance.append((concat,indexe1,indexe2))
        
        


# In[12]:
# this fonction is designed to solve the problem that if the tokenization changes, the new token may correspond more than one old tokens, and inherit 
# more than one labels. The solution is to choose the label that covers the most number of letters in the new token. 

def maxelement(l):
    dl=dict()
    for i in l:
        dl[i]=dl.get(i,0)+1
    ldl=sorted(dl, key=lambda x:dl[x], reverse=True)
    if len(ldl)>1:
        if ldl[0]=="O":
            tag=ldl[1]
        else:
            tag=ldl[0]
    else:
        tag=ldl[0]
        
    return tag


# In[13]:
# now the file with the new tokenization is read. This file and the last file contain the same sentences,
# in the same order, but with different tokenization.
# now we read the file of tokenized sentences that we want to assign labels to
tokenized=read_conllu("part3_dev.tsv",[1,2,3])


# In[15]:
# the final step is to correspond the letters in object tokens to the glued letters, and let the former inherit
# the label of the later, then assign the token with the label that belong to the most of the letters in the token

token_tags=list()
for i in range(len(correspondance)):
    w=tokenized[i][0]
    v1=tokenized[i][1]
    v2=tokenized[i][2]
    l=correspondance[i][0] 
    t1=correspondance[i][1]
    t2=correspondance[i][2]
    count=0
    for iw in range(len(w)):
	# "▁" is a special symbole of CamemBERT, that normally corresponds a space in sentence.  
	# to extract token from a glued text without space, it's better to exclude all the symbol "▁" 
        nw=w[iw].strip("▁") 
        if len(nw)>0:
            if count >= len(t1):
                rt1=maxelement(t1[-1:])
                rt2=maxelement(t2[-1:])
            elif count+len(nw) >= len(t1):
                rt1=maxelement(t1[count:])
                rt2=maxelement(t2[count:])
            else:  
                rt1=maxelement(t1[count:count+len(nw)])
                rt2=maxelement(t2[count:count+len(nw)])
            # once the token corresponds the letters in the glued text, the labels of the letter of each token will be processed by the function maxelement()
	    # this function will choose the label that occupies most letters in the token (variable rt1 and rt2), and assign it to the token
            v1[iw]=rt1
            v2[iw]=rt2
            count+=len(nw)
    token_tags.append((w,v1,v2))
# finally we have a list token_tags that contains the new tokens and their tags
        
            
            
        


# In[16]:
# the new tokens and their tags will be stocked in a tsv file with same format 

with open("recombined.tsv", "w", encoding="utf-8") as r:
    for w,v1,v2 in token_tags:
        for i in range(len(w)):
            r.write(str(i+1)+"\t"+w[i]+"\t"+v1[i]+"\t"+v2[i]+"\n")
        r.write("\n\n")

	


#  

# we can also use the same strategy to corrspond a treetagger tokenized sentence to a CamemBERT tokenized sentence



# In[17]:
# reading the camembert file that contains labels for CamemBERT tokens

corpus=read_conllu("Part3_Camembert_ONTAL_model123_1200_Automatique.tsv",[1,2])


# In[18]:
# we generate the glued text with CamemBERT tokenized sentences

correspondance=list()
for a,b in corpus:
    concat=""
    indexe=list()
    for i in range(len(a)):
	# the symbol "▁" should also be ignored here
        w=a[i].strip("▁")
        t=b[i]
        lw=len(w)
        lt=[t]*lw
        concat+=w
        indexe+=lt
        
    correspondance.append((concat,indexe))


# In[20]:
# reading the file tokenized by tree-tagger

tokenized=read_conllu("part3_dev_combined.tsv",[1,2])


# In[23]:
# same way of corresponding tokens to letters, and inheritating their labels

token_tags=list()
for i in range(len(correspondance)):
    w=tokenized[i][0]
    v=tokenized[i][1]
    l=correspondance[i][0]
    t=correspondance[i][1]
    count=0
    for iw in range(len(w)):
        nw=w[iw].strip("▁")
        if len(nw)>0:
            if count >= len(t):
                rt=maxelement(t[-1:])
            elif count+len(nw) >= len(t):
                rt=maxelement(t[count:])
            else:
                rt=maxelement(t[count:count+len(nw)])
            v[iw]=rt
            count+=len(nw)
    token_tags.append((w,v))


# In[24]:
# we save the tokenized results in a new file.
with open("Part3_Camembert_ONTAL_model123_1200_Automatique_recombined.tsv",'w',encoding="utf-8") as p:
    for a,b in token_tags:
        for i in range(len(a)):
            p.write(str(i+1)+"\t"+a[i]+'\t'+b[i]+'\t'+"_"+"\n")
        p.write("\n\n")




