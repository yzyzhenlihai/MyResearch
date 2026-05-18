import pickle
import os
import sys

ROOTPATH = "data/KuaiRec"
DATAPATH = os.path.join(ROOTPATH, "data_raw")
PRODATAPATH = os.path.join(ROOTPATH, "data_processed")

def main():
    filepath = os.path.join(PRODATAPATH, 'DM_KuaiEnv-v0_data.pkl')
    if os.path.isfile(filepath):
        # 已经计算过训练数据了
        with open(filepath,'rb') as f:
            trajectories = pickle.load(f)  
            print("load training trajectories")
        print("trajectories[0]:", trajectories[0])
        for trajectory in trajectories:
            trajectory["actions"] = trajectory["actions"].detach().cpu().numpy()
            trajectory["rewards"] = trajectory["rewards"].flatten()
            trajectory["terminals"] = trajectory["terminals"].flatten()

        with open(filepath, "wb") as f:
            pickle.dump(trajectories, f)
    else:
        print("文件不存在")



if __name__ == "__main__":
    main()