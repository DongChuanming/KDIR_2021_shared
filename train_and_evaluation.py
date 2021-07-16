import time
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from transformers import CamembertModel, CamembertTokenizer
from transformers import CamembertConfig
torch. no_grad()

#loading camembert language model
# You can replace "camembert-large" with any other camembert model, e.g. "camembert/camembert-base".
tokenizer = CamembertTokenizer.from_pretrained("camembert/camembert-large")
config = CamembertConfig.from_pretrained("camembert/camembert-large", output_hidden_states=True)
camembert = CamembertModel.from_pretrained("camembert/camembert-large", config=config)
camembert.eval()  # disable dropout (or leave in train mode to finetune)




def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix, outfeats):
    sents=[]
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score +         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


# In[23]:
# Bi-LASTM network setting

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, indexation):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.indexation=indexation
        self.wordembeds = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds=torch.tensor(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score +                 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        #print(sentence)
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


# In[66]:
# definition of the function that reads a conll file

import re
sep=re.compile("\n\t*\n\t*\n*\t*")
sac_mot=set()
#with open("/media/cdong/Elements/these/Projet/Extractor/Data/TRAIN_ONATLIDSU.tsv", "r", encoding="utf-8") as p:    #print(p.read())
def read_conllu(t,ind):
    with open(t, "r", encoding="utf-8") as p:
        t=p.read()
        phs=sep.split(t)
        #.split("\n\n")
        corpus=list()
        #dpos=dict()
        for s in phs:
            phrase=list()
            #tup1=list()
            #tup2=list()
            #tup3=list()
            for i in range(len(ind)):
                phrase.append(list())
            ws=s.split('\n')
            #print(ws)
            for w in ws:
                if len(w.strip("\t"))>0:
                    if w[0]!="#":
                        lt=w.split("\t")
                        for i in range(len(ind)):
                            index=ind[i]
                            phrase[i].append(lt[index])
            if len(phrase[0])>0:
                #if set(phrase[1])!={"O"}:
                corpus.append(phrase)
        return corpus 


# In[67]:

# read training corpus, but only register the second and third column (the second column is word forms,
# and the third column contains the annotation of the first group labels)
corpus=read_conllu("train.conll",[1,2])

# read dev corpus, same column as the train corpus
dev_corpus=read_conllu("dev.conll",[1,2])

# definition of pre-embedding function, we give the camembert vector of each word directly to the neural network as input
# to do this we need a function can vectorize the words before running the network
def preembed(corpus):
    for i in range(len(corpus)):
        a=corpus[i][0]
        encoded_sentence = tokenizer.encode(a)
        encoded_sentence = torch.tensor(encoded_sentence).unsqueeze(0)
        # for a semantic annotation, it's better to use the vectors generted in middle layers of camembert model, therefore here we set "11"
        # the layer to extract (since camembert has 25 layers)
        # otherwise for a more syntactic annotation, the last layers are more appropriate, "camembert(encoded_sentence)[0]" gives you directly 
        # the last layer embedding
        #lc=camembert(encoded_sentence)[0].tolist()
        lc=camembert(encoded_sentence)[2][11].tolist()
        emb=[[c] for c in lc[0][1:-1]]
        corpus[i].append(emb)
    return corpus


# In[71]:

# the vectors of each word will be inserted into the list that contains the word forms and their labels, for both train corpus and dev corpus
corpus=preembed(corpus)
dev_corpus=preembed(dev_corpus)


# this is a function that can generate butch indexation according to your configuration

def index_finder(l,p,b):
    e=b+p
    if e<l:
        return (b,e)
    else:
        d=e-l
        b=b-l
        return b,d
        
# the labels will be organized into a dictionary
cats=set()
START_TAG = "<START>"
STOP_TAG = "<STOP>"
for e in corpus:
#for x,y in corpus:
    if "" in e[1]:
        print(e[0])
    cats=cats|set(e[1])
#print(sorted(cats))
dc={}
for e in sorted(cats):
    if e not in dc:
        dc[e]=len(dc)+1
dc["O"]=0
dc[START_TAG]=len(dc)
dc[STOP_TAG]=len(dc)
print(dc)
cd = dict(zip(dc.values(), dc.keys()))


# In[75]:


with open("fr100.txt", "r", encoding="utf-8") as v:
    lv=v.read().split("\n")
    print(len(lv))
dw=dict()
for w in lv:
    if len(w)>100:
        lw=w.split(" ")
        dw[lw[0]]=[float(e) for e in lw[1:]]
#print(dw)


# In[76]:


with open("dic_tokens.txt", "r", encoding="utf-8") as v:
    t=v.read()
    exec("dw="+t)


# In[77]:


word_to_ix = {}
for w in dw:
    if w not in word_to_ix:
        word_to_ix[w] = len(word_to_ix)


# In[78]:


for m in sac_mot:
    if m not in word_to_ix:
        word_to_ix[m] = len(word_to_ix)


# In[79]:


ix_to_word = {}
for k,v in word_to_ix.items():
    ix_to_word[v]=k




# In[88]:
# now the neural network will be run to train the model

start = time.time()
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 1024
HIDDEN_DIM = 18 # we set hidden dimension to 18 to generate a complex model

# usage of the train corpus
training_data =corpus
dev=dev_corpus
gold=list()
for a,b,c in dev:
    gold+=b

tag_to_ix = dc

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM,ix_to_word)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-3)

# Check predictions before training
with torch.no_grad():
    #print(len(training_data[0][0]))
    #precheck_sent = prepare_sequence(training_data[0][0], word_to_ix,training_data[0][0])
    precheck_sent=training_data[0][2]
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    #outf=training_data[0][2].copy()
    #for i in range(len(precheck_sent)):
    #    outf[i]=outf[i].append(precheck_sent[i])
    #print(precheck_sent)
    print(model(precheck_sent))
l=len(training_data)
span=20
begin=0
# Make sure prepare_sequence from earlier in the LSTM section is loaded
model.zero_grad()
for epoch in range(
        34001):  # again, normally you would NOT do 300 epochs, it is toy data
    print(epoch, end="\r")
    begin,end=index_finder(l,span,begin)
    for sentence, tags, emb in training_data[begin:end]:
    #for sentence, tags, emb in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        #sentence_in = prepare_sequence(sentence, word_to_ix, sentence)
        sentence_in=emb
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        #print(targets)

        # Step 3. Run our forward pass.
        #print(sentence_in)
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
    #model.zero_grad()
    
    begin=end
    
    # an evaluation every 100 epoch
    if (epoch % 1700)==0:
        d_la=dict()
        d_lg=dict()
        d_ag=dict()
        #print(model(precheck_sent))
        la=0
        lg=0
        ag=0
        anno = list()
        with torch.no_grad():
            for a,b,c in dev:
                #li=list()
                precheck_sent = c
                for i in model(precheck_sent)[1]:
                    anno.append(cd[i])
        for i in range(len(anno)):
            a=anno[i]
            g=gold[i]
            if len(a)>1:
                la+=1
                d_la[a[1]]=d_la.get(a[1],0)+1
                if len(g)>1:
                    if a[1]==g[1]:
                        ag+=1
                        d_ag[a[1]]=d_ag.get(a[1],0)+1
            if len(g)>1:
                lg+=1
                d_lg[g[1]]=d_lg.get(g[1],0)+1
            
        try:
            prec=ag/la
        except:
            prec=1
        try:
            rap=ag/lg
        except:
            rap=1
        try:
            fscore=2*(prec*rap)/(prec+rap)
        except:
            fscore=1
        print(fscore)
        print(d_la)
        print(d_lg)
        print(d_ag)
        print("\n")
        torch.save(model, '/your_directory/'+str(epoch)+'.pkl')
            
        
# Evaluation part

# reload a model
model = torch.load('/your_directory/'+the_name_of_your_model_trained_just_now+'.pkl')

# test the model with test corpus
# read test corpus, same column as the corpus before
test_corpus=read_conllu("test.conllu",[1,2])


# In[82]:

# get the embedding of words in test corpus
gold=preembed(dev_corpus)


# In[83]:
# using the model on the test corpus
annotation = list()
with torch.no_grad():
    for a,b,c in gold:
        li=list()
        precheck_sent = c
        for i in model(precheck_sent)[1]:
            li.append(cd[i])
            #try:
            #    li.append(cd[i-1])
            #except:
            #    li.append(cd[i])
        # save the result in "annotation"
        annotation.append((a,li))
            


# save the annotation in a local file, in conll format
with open("annotated.conllu", "w", encoding="utf-8") as w:
    for ph in annotation:
        p=ph[0]
        a=ph[1]
        for i in range(len(p)):
            w.write(str(i+1)+"\t"+p[i]+"\t"+a[i]+"\t"+"_"+"\n")
        w.write("\n\n")


# 3 ways to evaluate the results

# including the non-label "O", calculate precision and recall on labels of each word (including border evaluation)
prec=dict()
rap=dict()
for i in range(len(gold)):
    g=gold[i][1]
    a=annotation[i][1]
    for e in range(len(g)):
        t1=g[e]
        t2=a[e]
        #prec[t2]
        l=prec.get(t2,[0,0])
        m=rap.get(t1,[0,0])
        if t2==t1:
            l[0]+=1
            l[1]+=1
            m[0]+=1
            m[1]+=1
            prec[t2]=l
            rap[t1]=m
        else:
            l[0]+=1
            m[0]+=1
            prec[t2]=l
            rap[t1]=m


#do not include "o", calculate precision and recall on labels of each word (including border evaluation)
prec=dict()
rap=dict()
for i in range(len(gold)):
    g=gold[i][1]
    a=annotation[i][1]
    for e in range(len(g)):
        t1=g[e]
        t2=a[e]
        #prec[t2]
        
        l=prec.get(t2,[0,0])
        m=rap.get(t1,[0,0])
        if t2==t1:
            if t1 != "O" :
                l[0]+=1
                l[1]+=1
                m[0]+=1
                m[1]+=1
                prec[t2]=l
                rap[t1]=m
        else:
            if t2 != "O":
                l[0]+=1
            if t1 != "O":
                m[0]+=1
            prec[t2]=l
            rap[t1]=m


# do not include "o", calculate precision and recall on category of each word (do not evaluate border information)
prec=dict()
rap=dict()
for i in range(len(gold)):
    g=gold[i][1]
    a=annotation[i][1]
    #g=gold[i][2]
    #a=gold[i][1]
    
    for e in range(len(g)):
        
        t1=g[e]
        try:
            t2=a[e]
        except:
            print(gold[i][0])
            print(annotation[i][0])
            continue
        #prec[t2]
        #if len(t2)>1:
        #    l=prec.get(t2[-1],[0,0])
        #if len(t1)>1:
        #    m=rap.get(t1[-1],[0,0])
        try:
            if t2[-1]==t1[-1]:
                if t1 != "O" and t2 != "O":
                    l=prec.get(t2[-1],[0,0])
                    m=rap.get(t1[-1],[0,0])
                    l[0]+=1
                    l[1]+=1
                    m[0]+=1
                    m[1]+=1
                    prec[t2[-1]]=l
                    rap[t1[-1]]=m
            else:
                if t2 != "O":
                    l=prec.get(t2[-1],[0,0])
                    l[0]+=1
                    prec[t2[-1]]=l
                if t1 != "O":
                    m=rap.get(t1[-1],[0,0])
                    m[0]+=1
                    rap[t1[-1]]=m
        except:
            print(t1+"----")
            print(t2+"----")


# In[135]:


print(rap)


# In[136]:


print(prec)



# calculate precisiona and recall in total (do not differentiate categories)
allg=0
alla=0
rappel=dict()
for k, v in rap.items():
    #if k!="O":
    alla+=v[1]
    allg+=v[0]
    nv=v[1]/v[0]
    rappel[k]=nv
all_rappel=alla/allg
print(all_rappel)


# In[131]:


allg=0
alla=0
precision=dict()
for k, v in prec.items():
    #if k!="O":
    alla+=v[0]
    allg+=v[1]
    nv=v[1]/v[0]
    precision[k]=nv
all_precision=allg/alla
print(all_precision)


# In[132]:


rappel


# In[133]:


precision



