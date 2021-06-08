
import re
sep=re.compile("\n\t*\n\t*\n*\t*")
sac_mot=set()
#with open("/media/cdong/Elements/these/Projet/Extractor/Data/TRAIN_ONATLIDSU.tsv", "r", encoding="utf-8") as p:    #print(p.read())
def read_conllu_phrases(t,ind):
    with open(t, "r", encoding="utf-8") as p:
        t=p.read()
        phs=sep.split(t)
        #.split("\n\n")
        corpus=list()
        phrases=list()
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
                    if w[0]=="#":
                        phrases.append(w.strip("#").strip())
                    if w[0]!="#":
                        lt=w.split("\t")
                        for i in range(len(ind)):
                            index=ind[i]
                            phrase[i].append(lt[index])
                        #tup1.append(lt[1])
                        #sac_mot.add(lt[1])
                        #tup2.append(lt[8])
                        #if lt[1].isupper():
                        #    case=3
                        #elif lt[1].istitle():
                        #    case=2
                        #else:
                        #    case=1
                        #tup3.append([])
                        #if lt[3] in dpos:
                        #    tup3.append([dpos[lt[3].strip()], case])
                        #else:
                        #    tup3.append([29, case])
            #phrase=(tup1,tup2)
            #dpos[tuple(tup1)]=tup3
            if len(phrase[0])>0:
                #if set(phrase[1])!={"O"}:
                corpus.append(tuple(phrase))
        return corpus, phrases


# In[9]:


corpus, phrases=read_conllu_phrases("/media/cdong/Elements/these/Projet/Extractor/Data/compare_date.tsv", [1,8])


# In[11]:


import dateparser


# In[10]:


corpus


#  

#  

# 分解词汇到CamemBERT分词方式

#   

#   

# In[1]:


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
                        #tup1.append(lt[1])
                        #sac_mot.add(lt[1])
                        #tup2.append(lt[8])
                        #if lt[1].isupper():
                        #    case=3
                        #elif lt[1].istitle():
                        #    case=2
                        #else:
                        #    case=1
                        #tup3.append([])
                        #if lt[3] in dpos:
                        #    tup3.append([dpos[lt[3].strip()], case])
                        #else:
                        #    tup3.append([29, case])
            #phrase=(tup1,tup2)
            #dpos[tuple(tup1)]=tup3
            if len(phrase[0])>0:
                #if set(phrase[1])!={"O"}:
                corpus.append(tuple(phrase))
        return corpus 


# In[28]:


corpus=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Results/TEST_compare.txt",[1,2,3])


# In[29]:


len(corpus)


# In[30]:


count=0
for a,b,c in corpus:
    for e in c:
        if len(e)>1:
            if e[1]=="T":
                count+=1


# In[31]:


count


# In[18]:


corpus=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Results/token_annot_compare.tsv",[1,2,3])


# In[19]:


corpus


# In[11]:


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
        
    correspondance.append((concat,indexe1,indexe2))
        
        


# In[12]:


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


tokenized=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Data/part3_dev.tsv",[1,2,3])


# In[14]:


tokenized


# In[15]:


token_tags=list()
for i in range(len(correspondance)):
    w=tokenized[i][0]
    v1=tokenized[i][1]
    v2=tokenized[i][2]
    l=correspondance[i][0]
    
    t1=correspondance[i][1]
    t2=correspondance[i][2]
    #if 'GRAPH' in l:
        #print(t2)
    count=0
    for iw in range(len(w)):
        nw=w[iw].strip("▁")
        if len(nw)>0:
            #if nw==l[i:i+len(nw)]:
            if count >= len(t1):
                rt1=maxelement(t1[-1:])
                rt2=maxelement(t2[-1:])
            elif count+len(nw) >= len(t1):
                rt1=maxelement(t1[count:])
                rt2=maxelement(t2[count:])
            else:  
                rt1=maxelement(t1[count:count+len(nw)])
                rt2=maxelement(t2[count:count+len(nw)])
                #if nw=="GRAPH":
                #    print
                
            v1[iw]=rt1
            v2[iw]=rt2
            
            count+=len(nw)
    #print(w)
    #print(v)
    token_tags.append((w,v1,v2))
    #print("\n\n")
        
            
            
        


# In[16]:


with open("/media/cdong/Elements/these/Projet/Extractor/Data/Part_3_dev_ONTALIDSU_corrigee_camembert.tsv", "w", encoding="utf-8") as r:
    for w,v1,v2 in token_tags:
        for i in range(len(w)):
            r.write(str(i+1)+"\t"+w[i]+"\t"+v1[i]+"\t"+v2[i]+"\n")
        r.write("\n\n")


# In[97]:


stoken=set()
with open("dic_tokens.txt","w", encoding="utf-8") as d:
    for a,b in tokenized:
        for e in a:
            stoken.add(e)
    d.write(str(stoken))


# In[31]:


with open("/media/cdong/Elements/these/Projet/Extractor/correcte_camembert.conllu", "r", encoding="utf-8") as r:
    with open("/media/cdong/Elements/these/Projet/Extractor/Data/Part_1_2_camembert.tsv", "w", encoding="utf-8") as w:
        lr=r.readlines()
        b=0
        e=0
        for i in range(len(lr)):
            #if lr[i].strip()=="#Le site de Champagne sur Seine a accueilli a priori une usine fabriquant du gaz à partir de la distillation de la houille (ce qui devra être confirmé par une étude historique).":
            #    b=i
            if lr[i].strip()=='''#Un impact dans les eaux souterraines est toutefois identifiés du coté du quartier Liegard.''':
                e=i+18
                break
        for l in lr[b:e]:
            w.write(l)


# In[ ]:





#  

# 合并成词

#  

# In[3]:


#corpus=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Résultat/Part3_Camembert_IDSU_Automatique.tsv",[1,2])


# In[17]:


corpus=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Results/Part3_Camembert_ONTAL_model123_1200_Automatique.tsv",[1,2])


# In[18]:


correspondance=list()
for a,b in corpus:
    concat=""
    indexe=list()
    for i in range(len(a)):
        w=a[i].strip("▁")
        t=b[i]
        lw=len(w)
        lt=[t]*lw
        concat+=w
        indexe+=lt
        
    correspondance.append((concat,indexe))
        
        


# In[19]:


len(correspondance)


# In[20]:


tokenized=read_conllu("/media/cdong/Elements/these/Projet/Extractor/Data/part3_dev_combined.tsv",[1,2])


# In[21]:


len(tokenized)


# In[22]:


correspondance


# In[23]:


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
            #if nw==l[i:i+len(nw)]:
            if count >= len(t):
                rt=maxelement(t[-1:])
            elif count+len(nw) >= len(t):
                rt=maxelement(t[count:])
            else:
                rt=maxelement(t[count:count+len(nw)])
            v[iw]=rt
            count+=len(nw)
    #print(w)
    #print(v)
    token_tags.append((w,v))
    #print("\n\n")


# In[24]:


token_tags


# In[25]:


with open("/media/cdong/Elements/these/Projet/Extractor/Results/Part3_Camembert_ONTAL_model123_1200_Automatique_recombined.tsv",'w',encoding="utf-8") as p:
    #p.write(str(token_tags))
    for a,b in token_tags:
        for i in range(len(a)):
            p.write(str(i+1)+"\t"+a[i]+'\t'+b[i]+'\t'+"_"+"\n")
        p.write("\n\n")


# In[ ]:




