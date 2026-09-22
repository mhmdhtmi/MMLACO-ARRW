"""Run repeated MMLACO-ARRW experiments and evaluate with ML-kNN."""
import argparse, os, sys, time
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import hamming_loss, label_ranking_loss, label_ranking_average_precision_score, f1_score
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mmlaco_arrw import compute_mutual_information, mmlaco

def evaluate_mlknn(selected, data, labels, seed, k=10, test_size=0.30):
    from skmultilearn.adapt import MLkNN
    X=data.iloc[:,np.asarray(selected,dtype=int)]
    X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=test_size,random_state=seed)
    clf=MLkNN(k=k); clf.fit(X_train.values,y_train.values); pred=clf.predict(X_test.values).toarray(); true=y_test.values
    return {"Hamming Loss":hamming_loss(true,pred),"Ranking Loss":label_ranking_loss(true,pred),
            "LRAP":label_ranking_average_precision_score(true,pred),
            "F1-micro":f1_score(true,pred,average="micro",zero_division=0),
            "F1-macro":f1_score(true,pred,average="macro",zero_division=0)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True); p.add_argument("--features",type=int,required=True)
    p.add_argument("--labels",type=int,required=True); p.add_argument("--selected",type=int,default=40)
    p.add_argument("--runs",type=int,default=10); p.add_argument("--start-seed",type=int,default=0)
    p.add_argument("--mode",choices=["fixed","arrw"],default="arrw"); p.add_argument("--mlknn-k",type=int,default=10)
    p.add_argument("--test-size",type=float,default=0.30); p.add_argument("--output",default="results/repeated_runs.csv")
    args=p.parse_args(); df=pd.read_csv(args.data,header=None)
    data=df.iloc[:,:args.features]; labels=df.iloc[:,args.features:args.features+args.labels]
    ants=data.shape[1] if data.shape[1]<100 else 100; mif,mil=compute_mutual_information(data,labels); rows=[]
    for run in range(args.runs):
        seed=args.start_seed+run; t=time.time()
        selected,tau,history=mmlaco(data,mif,mil,n_ants=ants,n_selected=args.selected,mode=args.mode,random_state=seed)
        metrics=evaluate_mlknn(selected,data,labels,seed,k=args.mlknn_k,test_size=args.test_size)
        rows.append({"run":run+1,"seed":seed,"mode":args.mode,"runtime_seconds":time.time()-t,**metrics})
    out=pd.DataFrame(rows); os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True); out.to_csv(args.output,index=False)
    cols=["Hamming Loss","Ranking Loss","LRAP","F1-micro","F1-macro"]
    pd.DataFrame({"mean":out[cols].mean(),"std":out[cols].std(ddof=1)}).to_csv(os.path.splitext(args.output)[0]+"_summary.csv")
    print(out.to_string(index=False)); print("\nMean ± SD"); print(pd.DataFrame({"mean":out[cols].mean(),"std":out[cols].std(ddof=1)}))
if __name__=="__main__": main()
