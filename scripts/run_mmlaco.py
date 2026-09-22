"""Run one MMLACO-ARRW experiment on a prepared headerless CSV."""
import argparse, json, os, sys, time
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mmlaco_arrw import compute_mutual_information, mmlaco

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", required=True); p.add_argument("--features", type=int, required=True)
    p.add_argument("--labels", type=int, required=True); p.add_argument("--selected", type=int, default=40)
    p.add_argument("--ants", type=int, default=None); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["fixed","arrw"], default="arrw"); p.add_argument("--output", default="results/run.json")
    args=p.parse_args()
    df=pd.read_csv(args.data, header=None)
    data=df.iloc[:,:args.features]; labels=df.iloc[:,args.features:args.features+args.labels]
    if labels.shape[1] != args.labels: raise ValueError("Feature/label counts do not match CSV.")
    ants=args.ants if args.ants is not None else (data.shape[1] if data.shape[1] < 100 else 100)
    t=time.time(); mif,mil=compute_mutual_information(data,labels)
    selected,tau,history=mmlaco(data,mif,mil,n_ants=ants,n_selected=args.selected,mode=args.mode,random_state=args.seed)
    payload={"data":args.data,"n_samples":int(len(data)),"n_features":int(data.shape[1]),"n_labels":int(labels.shape[1]),
             "n_selected":int(args.selected),"n_ants":int(ants),"mode":args.mode,"seed":int(args.seed),
             "runtime_seconds":time.time()-t,"selected_features":selected.tolist(),
             "alpha_first":float(history.iloc[0]["alpha"]),"gamma_first":float(history.iloc[0]["gamma"]),
             "alpha_last":float(history.iloc[-1]["alpha"]),"gamma_last":float(history.iloc[-1]["gamma"])}
    os.makedirs(os.path.dirname(args.output) or ".",exist_ok=True)
    with open(args.output,"w",encoding="utf-8") as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))
if __name__=="__main__": main()
