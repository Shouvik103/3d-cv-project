"""Bound matching work while retaining temporal overlap, revisits and wraparound."""
import numpy as np


def select_pairs(images,neighbors=6,retrieval=6,features=None):
    import cv2
    n=len(images);pairs=set()
    for i in range(n):
        for step in range(1,min(neighbors,n-1)+1):pairs.add(tuple(sorted((i,(i+step)%n))))
    descriptors=[]
    for image in images:
        hsv=cv2.cvtColor(cv2.resize(image,(64,64)),cv2.COLOR_BGR2HSV)
        hist=cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).ravel();hist/=max(np.linalg.norm(hist),1e-9);descriptors.append(hist)
    if features is not None and sum(map(len,features))>=64:
        # Small, per-input visual vocabulary distinguishes texture revisits from
        # merely similar colors; no pretrained retrieval model or GPU required.
        samples=np.concatenate([d[np.linspace(0,len(d)-1,min(250,len(d)),dtype=int)] for d in features if len(d)]).astype('f4')
        cv2.setRNGSeed(0)
        _,_,vocabulary=cv2.kmeans(samples,64,None,(cv2.TERM_CRITERIA_MAX_ITER,15,.01),1,cv2.KMEANS_PP_CENTERS)
        matcher=cv2.BFMatcher(cv2.NORM_L2);histograms=[]
        for d in features:
            assignments=[m.trainIdx for m in matcher.match(d.astype('f4'),vocabulary)] if len(d) else []
            histograms.append(np.bincount(assignments,minlength=64))
        descriptors=np.asarray(histograms,float)
        descriptors*=np.log((n+1)/(1+(descriptors>0).sum(0)))+1
        descriptors=np.sqrt(descriptors);descriptors/=np.maximum(np.linalg.norm(descriptors,axis=1,keepdims=True),1e-9)
    similarities=np.asarray(descriptors)@np.asarray(descriptors).T
    for i in range(n):
        for j in np.argsort(-similarities[i],kind='stable'):
            if i!=j:pairs.add(tuple(sorted((i,int(j)))))
            if sum(1 for a,b in pairs if i in (a,b) and min(abs(a-b),n-abs(a-b))>neighbors)>=retrieval:break
    return sorted(pair for pair in pairs if pair[0]!=pair[1])
